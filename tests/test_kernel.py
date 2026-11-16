import pytest

from cairn.tasks.kernel import parse_wrapper_signature, static_check
from cairn.tasks.kernel_exec import ExecReport, consistency_score, execute_kernel

DIV_SPEC = "Wrapper Entry Information: div(input, other, *, rounding_mode=None, out=None) -> Tensor; input (Tensor): x"

KERNEL = """
import torch
import triton
import triton.language as tl


@triton.jit
def _k(x_ptr, y_ptr, o_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.load(y_ptr + offs, mask=mask)
    tl.store(o_ptr + offs, x OP y, mask=mask)


def div(input, other, *, rounding_mode=None, out=None):
    input, other = torch.broadcast_tensors(input, other)
    input, other = input.contiguous(), other.contiguous()
    result = torch.empty_like(input)
    n = input.numel()
    _k[(triton.cdiv(n, 64),)](input, other, result, n, BLOCK=64)
    return result
"""


def test_signature_parsing_variants():
    sig = parse_wrapper_signature(DIV_SPEC)
    assert sig.py_name == "div" and sig.returns == "Tensor"
    assert [p.kind for p in sig.params] == [
        "positional",
        "positional",
        "keyword_only",
        "keyword_only",
    ]
    assert sig.render() == "def div(input, other, *, rounding_mode=None, out=None):"
    dotted = parse_wrapper_signature(
        "Wrapper Entry Information: def linalg.svd(A, full_matrices=True) -> (Tensor, Tensor, Tensor)"
    )
    assert dotted.py_name == "svd" and dotted.returns == "(Tensor, Tensor, Tensor)"
    typed = parse_wrapper_signature(
        "Wrapper Entry Information: f(x: torch.Tensor, n: int = 2) -> torch.Tensor. Args"
    )
    assert [p.name for p in typed.params] == ["x", "n"] and typed.returns == "torch.Tensor"
    assert parse_wrapper_signature("no wrapper here") is None


def test_static_check_accepts_a_good_kernel():
    report = static_check(KERNEL.replace("OP", "/"), parse_wrapper_signature(DIV_SPEC))
    assert report.ok, report.problems
    assert report.score == 1.0


@pytest.mark.parametrize(
    ("change", "needle"),
    [
        (("tl.arange", "tl.linspace"), "tl.linspace"),
        (
            ("def div(input, other, *, rounding_mode=None, out=None)", "def divide(a, b)"),
            "no top-level function",
        ),
        (
            ("def div(input, other, *, rounding_mode=None, out=None)", "def div(input)"),
            "parameters do not match",
        ),
        (("@triton.jit", ""), "no Triton kernel"),
        (
            ("    _k[(triton.cdiv(n, 64),)](input, other, result, n, BLOCK=64)", "    pass"),
            "ever launched",
        ),
        (
            ("    x = tl.load(x_ptr + offs, mask=mask)", "    x = torch.zeros(4)"),
            "uses torch inside",
        ),
        (("tl.store(", "tl.store(("), "Syntax error"),
    ],
)
def test_static_check_finds_typical_mistakes(change, needle):
    code = KERNEL.replace("OP", "/").replace(*change)
    report = static_check(code, parse_wrapper_signature(DIV_SPEC))
    assert not report.ok
    assert any(needle in problem for problem in report.problems), report.problems


def test_extra_optional_parameters_are_allowed():
    code = KERNEL.replace("OP", "/").replace("out=None):", "out=None, debug=False):")
    assert static_check(code, parse_wrapper_signature(DIV_SPEC)).ok


def test_consistency_metric_punishes_silent_wrong_answers():
    assert consistency_score(10, 10, 10) == pytest.approx(1.0)
    assert consistency_score(10, 5, 10) == pytest.approx(0.25)  # 0.5 * (1 - 0.5/1.0)
    assert consistency_score(5, 5, 10) == pytest.approx(
        0.5
    )  # same correct count, fewer silent failures
    assert consistency_score(0, 0, 10) == 0.0


def test_exec_report_feedback():
    assert "wrong result" in ExecReport("wrong_result", "off by 2").feedback()
    assert ExecReport("exec_ok").feedback() == ""
    assert ExecReport("exec_ok").conclusive and not ExecReport("timeout").conclusive


@pytest.mark.triton
@pytest.mark.parametrize(("op", "status"), [("/", "exec_ok"), ("*", "wrong_result")])
def test_kernel_runs_on_cpu_and_is_compared_with_torch(triton_python, op, status):
    report = execute_kernel(
        KERNEL.replace("OP", op), parse_wrapper_signature(DIV_SPEC), python_exe=triton_python
    )
    assert report.status == status, report.detail


@pytest.mark.triton
def test_kernel_failures_are_reported_not_raised(triton_python):
    sig = parse_wrapper_signature(DIV_SPEC)
    broken = KERNEL.replace("OP", "/").replace("import torch", "import torch\nimport not_a_module")
    assert execute_kernel(broken, sig, python_exe=triton_python).status == "import_error"
    hang = KERNEL.replace("OP", "/").replace(
        "    n = input.numel()", "    while True:\n        pass"
    )
    assert execute_kernel(hang, sig, python_exe=triton_python, timeout_s=6).status == "timeout"


def test_missing_triton_is_reported_as_unavailable(tmp_path):
    report = execute_kernel(
        KERNEL, parse_wrapper_signature(DIV_SPEC), python_exe=str(tmp_path / "no-python")
    )
    assert report.status == "unavailable"


@pytest.mark.data
def test_signatures_parse_for_every_test_input(task_loader):
    task = task_loader(8)
    assert all(parse_wrapper_signature(q.input) is not None for q in task.test)


@pytest.mark.data
def test_reference_kernels_pass_the_static_check(task_loader):
    """The dataset's own solutions are the ground truth for the checker: it may only
    flag what is really wrong.  (One example calls `tl.cidv`, a typo.)"""
    task = task_loader(8)
    flagged = [e for e in task.examples if not static_check(e.output).ok]
    assert len(flagged) == 1 and "tl.cidv" in static_check(flagged[0].output).problems[0]
