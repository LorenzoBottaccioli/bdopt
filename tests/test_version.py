import pytest

from bdopt.eplus import Version


@pytest.mark.parametrize(
    "text,expected",
    [("9.6", (9, 6, None)), ("9.6.0", (9, 6, 0)), ("V22-1-0", (22, 1, 0)), ("EnergyPlus-8-7-0", (8, 7, 0)),
     ("9.6.0-f420c06a69", (9, 6, 0)), ("23.1", (23, 1, None))],
)
def test_parse(text, expected):
    v = Version.parse(text)
    assert (v.major, v.minor, v.patch) == expected


def test_parse_error():
    with pytest.raises(ValueError):
        Version.parse("nonsense")


def test_ordering_and_matching():
    assert Version.parse("8.7") < Version.parse("8.9") < Version.parse("9.0.1") < Version.parse("22.1")
    assert Version.parse("9.0.1").matches("9.0")
    assert not Version.parse("9.0.1").matches("9.0.0")
    assert Version.parse("9.6.0").matches("9.6.0")
    assert Version.parse("25.2.0").dashed == "25-2-0"
    assert Version.parse("9.6") == Version.parse("9.6.0")
