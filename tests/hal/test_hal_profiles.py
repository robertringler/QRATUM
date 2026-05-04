from qsk.hal.cpu import CPU
from qsk.hal.fpga import FPGA
from qsk.hal.gpu import GPU
from qsk.hal.telemetry import collect


def test_hal_profiles_collect():
    data = collect(CPU(2, 3.0), GPU(4, 2.0), FPGA(1000, 100.0))
    assert data["cpu"]["cores"] == 2
    assert data["gpu"]["memory_gb"] == 2.0
    assert data["fpga"]["clock_mhz"] == 100.0
