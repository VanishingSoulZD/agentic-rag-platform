from calc import calculator

def test_calculator() -> None:
    assert calculator("1+1") == "2"
    assert calculator("1*1") == "1"
    assert calculator("2/1") == "2.0"

from calc import agent
def test_agent() -> None:
    result = agent.run("帮我算一下 1 乘以 7 再加 5")
    assert "12" in result