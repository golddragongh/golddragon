from golddragon._core import hello_from_bin
from golddragon.model import LinearRegression

_all_ = ["hello", "LinearRegression"]

def hello() -> str:
    return hello_from_bin()
