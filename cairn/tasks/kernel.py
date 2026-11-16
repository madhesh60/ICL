"""Task 8 helpers: read the requested function signature, and check generated code
without running it.

A small model writing GPU code fails in a few typical ways:

    - the wrapper gets a different name or different parameters than requested
    - it calls a Triton function that does not exist (`tl.cidv`, `tl.relu`, ...)
    - it writes a kernel but never launches it
    - it uses PyTorch inside the kernel, where only Triton operations work

All four can be seen by reading the code as a tree (the "abstract syntax tree"),
in milliseconds, with no GPU.  Each finding becomes one plain sentence that is
handed back to the model as a repair instruction.

The list of real Triton function names in `triton_api.json` was not typed from
memory.  It was read from the installed `triton` package (version recorded in
the file).  Fun fact found that way: one of the dataset's own reference kernels
calls `tl.cidv`, a typo of `cdiv` that no Triton version has.
"""

from __future__ import annotations

import ast
import inspect
import json
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

_ENTRY = re.compile(r"Wrapper Entry Information:\s*(?:def\s+)?([A-Za-z_][\w.]*)\s*\(")
_API_FILE = Path(__file__).with_name("triton_api.json")


@dataclass(frozen=True)
class Param:
    name: str
    kind: str  # "positional" | "keyword_only" | "var_positional" | "var_keyword"
    default: str | None = None  # source text of the default value, None = required


@dataclass(frozen=True)
class WrapperSignature:
    full_name: str  # as written in the task, may contain dots ("linalg.svd")
    params: tuple[Param, ...]
    returns: str = "Tensor"

    @property
    def py_name(self) -> str:
        """The name a Python `def` can carry: the part after the last dot."""
        return self.full_name.split(".")[-1]

    def render(self) -> str:
        parts: list[str] = []
        star_done = False
        for p in self.params:
            if p.kind == "var_positional":
                parts.append(f"*{p.name}")
                star_done = True
                continue
            if p.kind == "var_keyword":
                parts.append(f"**{p.name}")
                continue
            if p.kind == "keyword_only" and not star_done:
                parts.append("*")
                star_done = True
            parts.append(p.name if p.default is None else f"{p.name}={p.default}")
        return f"def {self.py_name}({', '.join(parts)}):"


def _balanced(text: str, start: int) -> int:
    """Index just after the bracket that closes the one opened right before `start`."""
    depth, quote = 1, ""
    for i in range(start, len(text)):
        ch = text[i]
        if quote:
            if ch == quote:
                quote = ""
        elif ch in "\"'":
            quote = ch
        elif ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
            if depth == 0:
                return i + 1
    return -1


def _params_from_ast(source: str) -> tuple[Param, ...]:
    args = ast.parse(f"def _f({source}): pass").body[0].args  # type: ignore[attr-defined]
    positional = [*args.posonlyargs, *args.args]
    defaults = [None] * (len(positional) - len(args.defaults)) + list(args.defaults)
    out = [
        Param(a.arg, "positional", ast.unparse(d) if d is not None else None)
        for a, d in zip(positional, defaults, strict=True)
    ]
    if args.vararg:
        out.append(Param(args.vararg.arg, "var_positional"))
    out.extend(
        Param(a.arg, "keyword_only", ast.unparse(d) if d is not None else None)
        for a, d in zip(args.kwonlyargs, args.kw_defaults, strict=True)
    )
    if args.kwarg:
        out.append(Param(args.kwarg.arg, "var_keyword"))
    return tuple(out)


def _params_by_hand(source: str) -> tuple[Param, ...]:
    """Fallback for parameter lists that are not valid Python (free-form docs)."""
    pieces, depth, current = [], 0, ""
    for ch in source:
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        if ch == "," and depth == 0:
            pieces.append(current)
            current = ""
        else:
            current += ch
    pieces.append(current)
    out: list[Param] = []
    keyword_only = False
    for piece in (p.strip() for p in pieces):
        if piece == "*":
            keyword_only = True
            continue
        name, _, default = piece.partition("=")
        name = name.split(":")[0].strip().lstrip("*")
        if name.isidentifier():
            kind = "keyword_only" if keyword_only else "positional"
            out.append(Param(name, kind, default.strip() or None if "=" in piece else None))
    return tuple(out)


def parse_wrapper_signature(task_input: str) -> WrapperSignature | None:
    """Find `name(params) -> returns` after "Wrapper Entry Information:"."""
    match = _ENTRY.search(task_input)
    if not match:
        return None
    end = _balanced(task_input, match.end())
    if end < 0:
        return None
    source = task_input[match.end() : end - 1]
    try:
        params = _params_from_ast(source)
    except SyntaxError:
        params = _params_by_hand(source)
    returns = "Tensor"
    arrow = re.match(r"\s*->\s*", task_input[end:])
    if arrow:
        rest = task_input[end + arrow.end() :]
        if rest.startswith("("):
            close = _balanced(rest, 1)
            returns = rest[:close] if close > 0 else "Tensor"
        else:
            named = re.match(r"[A-Za-z_]\w*(?:\.\w+)*(?:\[[^\]]*\])?", rest)
            returns = named.group(0) if named else "Tensor"
    return WrapperSignature(match.group(1), params, returns.strip())


# --------------------------------------------------------------------------- static check


@lru_cache(maxsize=1)
def known_triton_api() -> tuple[frozenset[str], frozenset[str]]:
    """(names under triton.language, names under triton), from the snapshot file,
    plus whatever an installed triton package offers."""
    data = json.loads(_API_FILE.read_text(encoding="utf-8"))
    language = set(data["language"]) | set(data.get("legacy", []))
    top = set(data["top_level"])
    try:  # a newer installed version may know more names than the snapshot
        import triton
        import triton.language as tl

        language |= {n for n in dir(tl) if not n.startswith("_")}
        top |= {n for n in dir(triton) if not n.startswith("_")}
    except Exception:  # triton is optional; the snapshot is enough
        pass
    return frozenset(language), frozenset(top)


@dataclass
class StaticReport:
    problems: list[str] = field(default_factory=list)  # must be fixed
    warnings: list[str] = field(default_factory=list)  # suspicious, not fatal
    facts: dict[str, object] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.problems

    @property
    def score(self) -> float:
        """1.0 = clean.  Used to rank candidates when nothing could be executed."""
        if self.facts.get("syntax_error"):
            return 0.0
        return max(0.0, 1.0 - 0.2 * len(self.problems) - 0.05 * len(self.warnings))


def _is_jit(decorator: ast.expr) -> bool:
    target = decorator.func if isinstance(decorator, ast.Call) else decorator
    if isinstance(target, ast.Attribute):
        return target.attr == "jit"
    return isinstance(target, ast.Name) and target.id == "jit"


def _stub_signature(args: ast.arguments) -> inspect.Signature:
    """Build a callable signature from a parsed argument list without running any
    of its default expressions (they are replaced by None)."""
    clean = ast.arguments(
        posonlyargs=[ast.arg(a.arg) for a in args.posonlyargs],
        args=[ast.arg(a.arg) for a in args.args],
        vararg=ast.arg(args.vararg.arg) if args.vararg else None,
        kwonlyargs=[ast.arg(a.arg) for a in args.kwonlyargs],
        kw_defaults=[ast.Constant(None) if d is not None else None for d in args.kw_defaults],
        kwarg=ast.arg(args.kwarg.arg) if args.kwarg else None,
        defaults=[ast.Constant(None) for _ in args.defaults],
    )
    namespace: dict[str, object] = {}
    exec(f"def _f({ast.unparse(clean)}): pass", namespace)
    return inspect.signature(namespace["_f"])  # type: ignore[arg-type]


def _call_compatible(args: ast.arguments, signature: WrapperSignature) -> str | None:
    """Would the calls a user of the requested signature makes work on this wrapper?"""
    wrapper = _stub_signature(args)
    positional = [p.name for p in signature.params if p.kind == "positional"]
    required = [p.name for p in signature.params if p.kind == "positional" and p.default is None]
    keyword = {p.name: None for p in signature.params if p.kind == "keyword_only"}
    try:
        wrapper.bind(*([None] * len(required)))  # shortest call: required arguments only
        wrapper.bind(*([None] * len(positional)), **keyword)  # everything positional + keywords
        wrapper.bind(**{name: None for name in [*positional, *keyword]})  # everything by name
    except TypeError as err:
        return str(err)
    return None


def static_check(code: str, signature: WrapperSignature | None = None) -> StaticReport:
    report = StaticReport()
    try:
        tree = ast.parse(code)
    except SyntaxError as err:
        report.facts["syntax_error"] = True
        report.problems.append(f"Syntax error on line {err.lineno}: {err.msg}.")
        return report

    language_names, top_names = known_triton_api()
    tl_aliases: set[str] = set()
    triton_aliases: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "triton.language" and alias.asname:
                    tl_aliases.add(alias.asname)
                elif alias.name == "triton":
                    triton_aliases.add(alias.asname or "triton")
        elif isinstance(node, ast.ImportFrom) and node.module == "triton":
            for alias in node.names:
                if alias.name == "language":
                    tl_aliases.add(alias.asname or "language")
    if not triton_aliases and not tl_aliases:
        report.problems.append("The code never imports triton (need `import triton` and `import triton.language as tl`).")

    functions = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
    jit_functions = [f for f in functions if any(_is_jit(d) for d in f.decorator_list)]
    report.facts["n_kernels"] = len(jit_functions)
    if not jit_functions:
        report.problems.append("No function is decorated with @triton.jit, so there is no Triton kernel.")

    unknown = sorted(
        {
            node.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id in tl_aliases
            and node.attr not in language_names
        }
    )
    if unknown:
        report.problems.append(
            "These names do not exist in triton.language: " + ", ".join(f"tl.{n}" for n in unknown) + "."
        )
    unknown_top = sorted(
        {
            node.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id in triton_aliases
            and node.attr not in top_names
            and not node.attr.startswith("__")
        }
    )
    if unknown_top:
        report.problems.append(
            "These names do not exist in triton: " + ", ".join(f"triton.{n}" for n in unknown_top) + "."
        )

    for kernel in jit_functions:
        # Only the body: decorators such as @triton.heuristics(lambda: ... torch.float16)
        # run in normal Python and may use torch freely.
        body_nodes = [node for statement in kernel.body for node in ast.walk(statement)]
        foreign = sorted(
            {
                node.value.id
                for node in body_nodes
                if isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Name)
                and node.value.id in {"torch", "np", "numpy", "F", "math"}
            }
        )
        if foreign:
            report.problems.append(
                f"Kernel `{kernel.name}` uses {', '.join(foreign)} inside @triton.jit; "
                "only triton.language operations work inside a kernel."
            )
        calls = {
            node.func.attr
            for node in body_nodes
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id in tl_aliases
        }
        if "store" not in calls and not any(isinstance(n, ast.Return) and n.value for n in body_nodes):
            report.warnings.append(f"Kernel `{kernel.name}` never calls tl.store, so it writes no result.")

    launches = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and (
            isinstance(n.func, ast.Subscript)
            or (isinstance(n.func, ast.Attribute) and n.func.attr == "run")
        )
    ]
    if jit_functions and not launches:
        report.problems.append("No kernel is ever launched (expected a call like `kernel[grid](...)`).")

    if signature is not None:
        accepted = {signature.py_name, signature.full_name.replace(".", "_")}
        top_level = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
        wrapper = next((top_level[name] for name in accepted if name in top_level), None)
        report.facts["wrapper_found"] = wrapper is not None
        if wrapper is None:
            report.problems.append(
                f"There is no top-level function named `{signature.py_name}`. "
                f"It must be defined exactly as `{signature.render()}`."
            )
        else:
            if any(_is_jit(d) for d in wrapper.decorator_list):
                report.problems.append(
                    f"`{signature.py_name}` must be a normal Python wrapper, not the @triton.jit kernel."
                )
            mismatch = _call_compatible(wrapper.args, signature)
            if mismatch:
                report.problems.append(
                    f"The wrapper's parameters do not match the requested ones ({mismatch}). "
                    f"It must be defined exactly as `{signature.render()}`."
                )
    return report
