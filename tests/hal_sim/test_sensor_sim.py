from math import isclose

from qsk.hal.sensors import Sensor, sensor_bundle


def test_sensor_bundle():
    sensors = [Sensor("sine", lambda t: t * 0.1), Sensor("step", lambda t: 1 if t > 0 else 0)]
    reading = sensor_bundle(sensors, 3)
    assert isclose(reading["sine"], 0.3)
    assert reading["step"] == 1
