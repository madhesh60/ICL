"""Build the E003/E004 Kaggle comparison from the current working tree."""

import base64
import hashlib
import json
import textwrap
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/kaggle-e004-delivery"
DATA = Path("OpenSeek/openseek/competition/LongContext-ICL-Annotation/data")
BASELINE = Path("dev_results_now")


def add(archive, name, content):
    # Fixed metadata makes two packages of the same inputs byte-identical.
    header = zipfile.ZipInfo(name)
    header.compress_type = zipfile.ZIP_DEFLATED
    archive.writestr(header, content)


def build_zip(path):
    files = [
        Path("configs/default.yaml"),
        Path("configs/serve/flagscale_qwen3_4b.yaml"),
        Path("scripts/kaggle_e004.sh"),
        Path("scripts/compare_e003.py"),
        Path("OpenSeek/hf_openseek/tokenizer/qwen.tiktoken"),
        DATA / "openseek-2_count_nouns_verbs.json",
        DATA / "openseek-7_jeopardy_answer_generation_all.json",
        DATA / "openseek-8_kernel_generation.json",
        BASELINE / "task8/openseek-8.jsonl",
    ]
    for task in (2, 7):
        for name in ("dev.journal.jsonl", "dev.report.json"):
            files.append(BASELINE / f"task{task}" / name)
    files.extend(p.relative_to(ROOT) for p in (ROOT / "cairn").rglob("*.py"))
    files.append(Path("cairn/tasks/triton_api.json"))
    manifest = []
    with zipfile.ZipFile(path, "w") as archive:
        for relative in sorted(files):
            content = (ROOT / relative).read_bytes()
            if relative.suffix == ".sh":
                content = content.replace(b"\r\n", b"\n")
            name = relative.as_posix()
            manifest.append(f"{hashlib.sha256(content).hexdigest()}  {name}\n")
            add(archive, name, content)
        add(archive, "e004-files.sha256", "".join(manifest).encode())


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    archive = OUT / "cairn-e004.zip"
    build_zip(archive)
    payload = archive.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    encoded = base64.b64encode(payload).decode("ascii")
    cell = textwrap.dedent('''\
        # Select GPU T4 x2 and enable Internet. Run this cell once.
        # The embedded ZIP contains the updated Cairn code, official inputs and E003 predictions.
        import base64
        import hashlib
        import io
        import subprocess
        import tempfile
        import zipfile
        from pathlib import Path

        def run_comparison(encoded, expected_sha256):
            payload = base64.b64decode(encoded)
            if hashlib.sha256(payload).hexdigest() != expected_sha256:
                raise ValueError("The embedded snapshot is incomplete. Reimport the notebook.")
            with tempfile.TemporaryDirectory() as work:
                with zipfile.ZipFile(io.BytesIO(payload)) as archive:
                    archive.extractall(work)
                subprocess.run(["bash", str(Path(work) / "scripts/kaggle_e004.sh")], check=True)

        # Snapshot bytes follow. No separate dataset upload is needed for this notebook.
        run_comparison(b"""
        ''')
    # Avoid uploading thousands of notebook lines for binary data.
    cell += "\n".join(encoded[i : i + 8192] for i in range(0, len(encoded), 8192))
    cell += f'\n""", "{digest}")\n'
    notebook = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        },
        "cells": [
            {
                "cell_type": "code",
                "id": "paired-e004",
                "metadata": {},
                "execution_count": None,
                "outputs": [],
                "source": cell.splitlines(keepends=True),
            }
        ],
    }
    book = OUT / "cairn-e004.ipynb"
    book.write_text(json.dumps(notebook, indent=1) + "\n", encoding="utf-8")
    print(f"{book}\n{archive}\nZIP SHA-256 {digest}")


if __name__ == "__main__":
    main()
