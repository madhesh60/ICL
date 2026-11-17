"""Runs ONE generated Triton module in a throw-away process and reports what happened.

This file is executed as a script by `kernel_exec.py`; it must not import cairn.
Input (JSON on stdin):  {"code", "name", "full_name", "params", "interpret"}
Output (JSON on stdout): {"status", "detail", ...}

Statuses
    exec_ok              ran and matches the PyTorch function of the same name
    wrong_result         ran but the numbers (or shapes) differ from PyTorch
    called_no_reference  ran and returned finite tensors; nothing to compare with
    call_error           raised an exception when called with valid arguments
    unconfirmed_error    raised on every guess, but we cannot prove the guesses were valid
    import_error         the module itself failed to load
    no_valid_inputs      we could not guess arguments that the function accepts
    unavailable          torch / triton are not installed here

The key trick: when PyTorch has a function with the requested name (`div`,
`tanh`, `argmax` ...), we first try argument guesses on *PyTorch's* function.
The first guess PyTorch accepts is a valid input by definition, and PyTorch's
result is the expected output.  The generated wrapper is then called with
exactly those arguments.
"""

import io
import itertools
import json
import os
import sys
import traceback

job = json.loads(sys.stdin.read())
if job.get("interpret"):
    # Triton's interpreter mode: kernels run on the CPU through numpy.  Slow, but it
    # needs no GPU, which is all a correctness check asks for.
    os.environ["TRITON_INTERPRET"] = "1"

_real_stdout = sys.stdout
sys.stdout = io.StringIO()


def finish(status, detail="", **extra):
    _real_stdout.write(json.dumps({"status": status, "detail": str(detail)[:1500], **extra}))
    _real_stdout.flush()
    os._exit(0)


try:
    import torch
    import triton  # noqa: F401
except Exception as err:
    finish("unavailable", f"{type(err).__name__}: {err}")

torch.manual_seed(0)
DEVICE = "cuda" if (torch.cuda.is_available() and not job.get("interpret")) else "cpu"

namespace = {"__name__": "candidate"}
try:
    exec(compile(job["code"], "<candidate>", "exec"), namespace)
except BaseException as err:
    finish("import_error", f"{type(err).__name__}: {err}\n{traceback.format_exc(limit=3)}")

wrapper = namespace.get(job["name"]) or namespace.get(job["full_name"].replace(".", "_"))
if not callable(wrapper):
    finish("call_error", f"no callable named {job['name']} is defined")


def find_reference(full_name):
    last = full_name.split(".")[-1]
    for root, dotted in (
        (torch, full_name),
        (torch.nn.functional, last),
        (torch.linalg, last),
        (torch.special, last),
    ):
        target = root
        try:
            for part in dotted.split("."):
                target = getattr(target, part)
        except AttributeError:
            continue
        if callable(target):
            return target
    return None


# Parameters that are plainly not tensors, with values worth trying.
SCALARS = {
    "dim": [-1, 0, 1], "axis": [-1, 0], "dim0": [0], "dim1": [1], "dims": [(1, 0), (0,)],
    "k": [2], "n": [2], "p": [2.0, 0.5], "exponent": [2.0], "alpha": [1.0], "beta": [1.0],
    "value": [0.5], "min": [-0.5], "max": [0.5], "eps": [1e-5], "diagonal": [0],
    "shifts": [1], "repeats": [2], "chunks": [2], "sections": [2], "split_size_or_sections": [2],
    "num_classes": [16], "groups": [1], "kernel_size": [2], "stride": [1], "padding": [0],
    "threshold": [0.1], "negative_slope": [0.01], "lambd": [0.5], "scale_factor": [2],
    "output_size": [2], "start_dim": [0], "end_dim": [-1], "keepdim": [False], "num_groups": [1],
    "size": [(32, 64)], "shape": [(2048,), (64, 32)], "sorted": [True], "upper": [False],
    "source": [0], "destination": [1], "offset": [0], "steps": [5], "decimals": [1],
    "training": [False], "inplace": [False], "normalized_shape": [64], "num_features": [64],
}  # fmt: skip
INDEX_NAMES = {"index", "indices", "idx", "target", "labels"}

# Tensor shapes to try, by number of tensor arguments.  Ordered from "same shape"
# (elementwise maths) to matrix products, linear layers and convolutions.
#
# Sizes are multiples of 16 and powers of two on purpose.  Triton kernels have hard
# rules that small odd shapes break even when the kernel is right: `tl.dot` needs every
# side to be at least 16, `tl.arange` needs a power-of-two length, and many generated
# kernels pick a block size from the input size.  A correct kernel that is handed a
# (6, 10) tensor raises, and we would wrongly count it as broken.  The odd shapes at the
# end stay as a last resort for functions that PyTorch only accepts in such shapes.
SHAPES = {
    0: [()],
    1: [((32, 64),), ((64,),), ((16, 16),), ((2, 4, 16, 16),), ((6, 10),)],
    2: [
        ((32, 64), (32, 64)), ((32, 64), (64, 16)), ((32, 64), (64,)), ((2, 32, 64), (2, 64, 16)),
        ((16, 16), (16, 16)), ((16, 16), (16, 32)), ((16, 16), (16,)), ((32, 64), (16, 64)),
        ((2, 4, 16, 16), (8, 4, 3, 3)), ((64,), (64,)), ((6, 10), (6, 10)), ((6, 10), (10, 4)),
    ],
    3: [
        ((32, 64), (32, 64), (32, 64)), ((2, 32, 64), (2, 64, 16), (2, 32, 16)),
        ((32, 16), (32, 64), (64, 16)), ((32, 64), (16, 64), (16,)),
        ((2, 4, 16, 16), (8, 4, 3, 3), (8,)), ((32,), (32, 64), (64,)),
        ((2, 32, 16), (2, 32, 64), (2, 64, 16)), ((6, 10), (6, 10), (6, 10)),
    ],
}  # fmt: skip


def make_tensor(shape, positive):
    values = (
        torch.rand(shape, device=DEVICE) + 0.5 if positive else torch.randn(shape, device=DEVICE)
    )
    return values.to(torch.float32)


def argument_guesses(params):
    """Yield (args, kwargs, description) guesses for the required parameters."""
    required = [p for p in params if p["kind"] == "positional" and p["default"] is None]
    star = next((p for p in params if p["kind"] == "var_positional"), None)
    tensor_slots = [
        p["name"] for p in required if p["name"] not in SCALARS and p["name"] not in INDEX_NAMES
    ]
    n_tensors = len(tensor_slots) + (2 if star else 0)
    shape_sets = SHAPES.get(n_tensors) or [tuple((32, 64) for _ in range(n_tensors))]
    scalar_slots = [p["name"] for p in required if p["name"] in SCALARS]
    scalar_sets = list(itertools.product(*(SCALARS[name] for name in scalar_slots)))[:6] or [()]
    for shapes in shape_sets:
        for positive in (False, True):
            for scalars in scalar_sets:
                tensors = iter(make_tensor(s, positive) for s in shapes)
                chosen = dict(zip(scalar_slots, scalars, strict=True))
                args = []
                for p in required:
                    if p["name"] in chosen:
                        args.append(chosen[p["name"]])
                    elif p["name"] in INDEX_NAMES:
                        args.append(torch.tensor([0, 2], device=DEVICE))
                    else:
                        args.append(next(tensors))
                if star:
                    args.extend(tensors)
                note = f"tensor shapes {list(shapes)}" + (f", {chosen}" if chosen else "")
                yield args, {}, note


def clone_all(args):
    return [a.clone() if isinstance(a, torch.Tensor) else a for a in args]


def as_list(value):
    return list(value) if isinstance(value, (tuple, list)) else [value]


def compare(got, expected):
    got, expected = as_list(got), as_list(expected)
    if len(got) != len(expected):
        return f"returned {len(got)} values, expected {len(expected)}"
    for i, (g, e) in enumerate(zip(got, expected, strict=True)):
        if not isinstance(g, torch.Tensor):
            return f"output {i} is {type(g).__name__}, expected a Tensor"
        if tuple(g.shape) != tuple(e.shape):
            return f"output {i} has shape {tuple(g.shape)}, expected {tuple(e.shape)}"
        g64, e64 = g.detach().cpu().to(torch.float64), e.detach().cpu().to(torch.float64)
        if not torch.allclose(g64, e64, rtol=1e-4, atol=1e-5, equal_nan=True):
            worst = (g64 - e64).abs().nan_to_num(posinf=1e30).max().item()
            return f"output {i} differs from the PyTorch result (largest error {worst:.4g})"
    return ""


reference = find_reference(job["full_name"])
tried = 0
last_error = ""
for args, kwargs, note in argument_guesses(job["params"]):
    tried += 1
    expected = None
    if reference is not None:
        try:
            expected = reference(*clone_all(args), **kwargs)
        except Exception:
            continue  # PyTorch rejects this guess, so it is not a valid input
    try:
        got = wrapper(*clone_all(args), **kwargs)
    except BaseException as err:
        last_error = f"calling with {note} raised {type(err).__name__}: {err}"
        if reference is not None:
            finish("call_error", last_error, inputs=note)
        continue  # without a reference we cannot tell a bad guess from a bad wrapper
    if reference is not None:
        problem = compare(got, expected)
        if problem:
            finish("wrong_result", f"with {note}: {problem}", inputs=note)
        finish("exec_ok", f"matches torch for {note}", inputs=note)
    outputs = as_list(got)
    if not outputs or not all(isinstance(o, torch.Tensor) for o in outputs):
        finish("call_error", f"with {note}: the wrapper did not return a Tensor", inputs=note)
    if any(o.is_floating_point() and not torch.isfinite(o).all() for o in outputs):
        finish("wrong_result", f"with {note}: the output contains NaN or infinity", inputs=note)
    finish("called_no_reference", f"ran with {note}", inputs=note)

if last_error:
    # Every guess raised and there is no PyTorch twin to tell us whether the guesses
    # or the wrapper are at fault.  Report it as a hint, not as a verdict.
    finish("unconfirmed_error", last_error)
finish("no_valid_inputs", f"none of {tried} argument guesses was accepted")
