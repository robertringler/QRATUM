from qnx_agi.formal.abstract_domains.interval import Interval
from qnx_agi.formal.static_analyzer import StaticAnalyzer


def test_static_analyzer_accepts_valid_intervals():
    analyzer = StaticAnalyzer()
    assert analyzer.analyze([Interval(0, 1), Interval(-1, 2)])
    assert not analyzer.analyze([Interval(5, 4)])
