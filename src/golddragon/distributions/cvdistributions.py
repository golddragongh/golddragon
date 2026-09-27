import secrets
import math

def uniform(a: float = 0.0, b: float = 1.0) -> float:
    """Cryptographically secure uniform sample."""
    u = secrets.randbits(53) / (1 << 53)
    return a + (b - a) * u

def exponentialdist(lam: float) -> float:
    """Inverse transform sample from Exponential(lam)."""
    y = uniform(0.0, 1.0)
    return -math.log(y) / lam

def poissiondist(lam: float) -> int:
    """Inverse transform sample from Poisson(lam)."""
    y = uniform(0.0, 1.0)
    k = 0
    pmf = math.exp(-lam)
    cdf = pmf
    while y > cdf:
        k += 1
        pmf *= lam / k
        cdf += pmf
    return k