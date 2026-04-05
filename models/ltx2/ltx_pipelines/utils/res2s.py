import math


def phi(j: int, neg_h: float) -> float:
    if abs(neg_h) < 1e-10:
        return 1.0 / math.factorial(j)
    remainder = sum(neg_h**k / math.factorial(k) for k in range(j))
    return (math.exp(neg_h) - remainder) / (neg_h**j)


def get_res2s_coefficients(h: float, phi_cache: dict, c2: float = 0.5) -> tuple[float, float, float]:
    def get_phi(j: int, neg_h: float) -> float:
        cache_key = (j, neg_h)
        if cache_key in phi_cache:
            return phi_cache[cache_key]
        value = phi(j, neg_h)
        phi_cache[cache_key] = value
        return value

    neg_h_c2 = -h * c2
    a21 = c2 * get_phi(1, neg_h_c2)

    neg_h_full = -h
    b2 = get_phi(2, neg_h_full) / c2
    b1 = get_phi(1, neg_h_full) - b2
    return a21, b1, b2
