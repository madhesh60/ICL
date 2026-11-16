"""A tiny stand-in for a vLLM server, for testing the HTTP client for real.

It speaks just enough of the OpenAI chat API, can be told to fail a number of
times before succeeding, records every request body, and imitates a prefix
cache by reporting how many prompt characters it has seen before.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class FakeLLMServer:
    def __init__(self) -> None:
        self.fail_statuses: list[int] = []  # returned (in order) before any success
        self.requests: list[dict] = []
        self.reply_text = "OK"
        self.include_think_inline = False
        self.use_reasoning_field = True
        self._seen_prefixes: list[str] = []
        self._lock = threading.Lock()
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args) -> None:  # keep test output quiet
                pass

            def _send(self, status: int, body: dict) -> None:
                data = json.dumps(body).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self) -> None:
                if self.path.endswith("/models"):
                    self._send(200, {"data": [{"id": "Qwen3-4B"}]})
                else:
                    self._send(404, {"error": "not found"})

            def do_POST(self) -> None:
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                with outer._lock:
                    outer.requests.append(body)
                    status = outer.fail_statuses.pop(0) if outer.fail_statuses else 200
                if status != 200:
                    self._send(status, {"error": f"simulated {status}"})
                    return
                self._send(200, outer._answer(body))

        self._httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.port = self._httpd.server_address[1]
        self.base_url = f"http://127.0.0.1:{self.port}/v1"
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)

    def _answer(self, body: dict) -> dict:
        prompt = "".join(m["content"] for m in body["messages"])
        with self._lock:
            cached = 0
            for old in self._seen_prefixes:
                shared = 0
                for a, b in zip(prompt, old):
                    if a != b:
                        break
                    shared += 1
                cached = max(cached, shared)
            self._seen_prefixes.append(prompt)
        structured = body.get("structured_outputs") or {}
        choice = structured.get("choice") or body.get("guided_choice")
        thinking = (body.get("chat_template_kwargs") or {}).get("enable_thinking", True)
        text = choice[0] if choice else self.reply_text
        message: dict = {"role": "assistant", "content": text}
        if thinking:
            if self.include_think_inline:
                message["content"] = f"<think>\nreasoning here\n</think>\n\n{text}"
            elif self.use_reasoning_field:
                message["reasoning"] = "reasoning here"
        choices = []
        for index in range(body.get("n", 1)):
            entry: dict = {"index": index, "message": message, "finish_reason": "stop"}
            if body.get("logprobs"):
                entry["logprobs"] = {
                    "content": [
                        {
                            "token": text[:1],
                            "logprob": -0.1,
                            "top_logprobs": [
                                {"token": "Sad", "logprob": -0.4},
                                {"token": "Not", "logprob": -1.2},
                                {"token": "The", "logprob": -6.0},
                            ][: body.get("top_logprobs", 3)],
                        }
                    ]
                }
            choices.append(entry)
        return {
            "choices": choices,
            "usage": {
                "prompt_tokens": len(prompt) // 4,
                "completion_tokens": 2,
                "prompt_tokens_details": {"cached_tokens": cached // 4},
            },
        }

    def __enter__(self) -> FakeLLMServer:
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._httpd.shutdown()
        self._httpd.server_close()
