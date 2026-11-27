import ctypes
from unittest.mock import Mock

import pytest

from scripts import kaggle_gpu


@pytest.mark.parametrize(
    ("names", "expected"),
    [([], "CUDA sees"), ([b"Tesla T4"], "CUDA sees"), ([b"Tesla T4", b"Tesla T4"], None)],
)
def test_checks_cuda_devices_without_smi(names, expected, monkeypatch, capsys):
    driver = Mock()
    driver.cuInit.return_value = 0

    def set_int(ptr, value):
        ctypes.cast(ptr, ctypes.POINTER(ctypes.c_int))[0] = value
        return 0

    def name(buf, size, device):
        buf.value = names[device.value]
        return 0

    driver.cuDeviceGetCount.side_effect = lambda ptr: set_int(ptr, len(names))
    driver.cuDeviceGet.side_effect = set_int
    driver.cuDeviceGetName.side_effect = name
    monkeypatch.setattr(ctypes, "CDLL", lambda path: driver)
    if expected:
        with pytest.raises(SystemExit, match=expected):
            kaggle_gpu.main()
    else:
        kaggle_gpu.main()
        assert "Tesla T4 + Tesla T4" in capsys.readouterr().out


def test_missing_driver_reports_saved_version_setting(monkeypatch):
    monkeypatch.setattr(ctypes, "CDLL", Mock(side_effect=OSError("libcuda.so.1 unavailable")))
    with pytest.raises(SystemExit, match="saved version's accelerator"):
        kaggle_gpu.main()
