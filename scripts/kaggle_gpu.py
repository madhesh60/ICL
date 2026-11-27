"""Check the two E003 GPUs through the CUDA driver before installing any packages."""

import ctypes
import sys


def gpu_names(driver):
    def check(status, operation):
        if status:
            raise RuntimeError(f"{operation} returned CUDA error {status}")

    check(driver.cuInit(0), "cuInit")
    count = ctypes.c_int()
    check(driver.cuDeviceGetCount(ctypes.byref(count)), "cuDeviceGetCount")
    names = []
    for ordinal in range(count.value):
        device = ctypes.c_int()
        check(driver.cuDeviceGet(ctypes.byref(device), ordinal), "cuDeviceGet")
        name = ctypes.create_string_buffer(256)
        check(driver.cuDeviceGetName(name, len(name), device), "cuDeviceGetName")
        names.append(name.value.decode("utf-8"))
    return names


def main():
    try:
        names = gpu_names(ctypes.CDLL("libcuda.so.1"))
    except (OSError, RuntimeError) as err:
        raise SystemExit(
            f"GPU preflight failed: {err}. Check GPU T4 x2 in the saved version's "
            "accelerator settings and check your available GPU quota."
        ) from err
    if len(names) != 2 or any("T4" not in name.split() for name in names):
        raise SystemExit(f"Expected two T4 GPUs for this comparison. CUDA sees {names!r}.")
    print(f"GPU preflight passed: {names[0]} + {names[1]}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
