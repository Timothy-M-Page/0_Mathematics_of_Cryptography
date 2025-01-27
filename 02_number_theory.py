import math


def extended_euclid_algorithm(a: int, b: int) -> list[int]:
    if a == 0:
        return [b, [0, 1]]
    if b == 0:
        return [a, [1, 0]]

    larger, smaller = max(a, b), min(a, b)
    remainder = larger % smaller
    quotient = larger//smaller
    remainders = [larger, smaller, remainder]
    quotients = [quotient]
    # These s and t sets determine the integer coefficients in ax + by = gcd.
    s_coefficients = [1, 0]
    t_coefficients = [0, 1]

    while remainders[-1] != 0:
        quotient = remainders[-2] // remainders[-1]
        remainder = remainders[-2] % remainders[-1]
        quotients.append(quotient)
        remainders.append(remainder)

        s = s_coefficients[-2] - quotients[-2] * s_coefficients[-1]
        t = t_coefficients[-2] - quotients[-2] * t_coefficients[-1]
        s_coefficients.append(s)
        t_coefficients.append(t)

    if a == larger:
        return [remainders[-2], s_coefficients[-1], t_coefficients[-1]]
    else:
        return [remainders[-2], t_coefficients[-1], s_coefficients[-1]]


def gcd(a: int, b: int) -> int:
    # May call the Euclidean algorithm using the math package.
    return math.gcd(a, b)


def modular_inv(g: int, mod: int) -> int:
    if math.gcd(g, mod) != 1:
        raise ValueError(f'{g} has no inverse mod {mod}.')
    return pow(g, -1, mod)


def phi(n: int) -> int:
    result = n
    # For each prime that divides n, multiply result by (1 - 1/prime).
    prime = 2
    while prime**2 <= n:
        if n % prime == 0:
            while n % prime == 0:
                n //= prime
            result -= result // prime
        prime += 1
    # A condition for the case that n itself is prime.
    if n > 1:
        result -= result // n
    return result


def pollard_rho(n: int) -> list[int]:
    """
    Pollard's rho algorithm provides an O(n**1/4) algorithm for
    factoring numbers. The algorithm iterates values of a polynomial
    mod n to create two sequences, whose difference is more likely
    to share a factor with n, due to the birthday paradox.
    """
    # Choose a polynomial
    def polynomial(z: int) -> int:
        return z**2 + 1

    original_input = n
    factors = []
    x = 2
    y = 2
    # Iterate the values of x,y and d, until a factor is shared with n.
    iteration = 0
    while n > 1:
        iteration += 1
        x = polynomial(x) % n
        y = polynomial(polynomial(y)) % n
        d = gcd(abs(x-y), n)
        if d == 1:
            continue
        # Remove each factor when it is found and continue the loop.
        if d in range(2, n):
            n = n//d
            factors.append(d)
            if n == 1:
                break
        # The case the remaining factor n//d is itself prime.
        if d == n:
            if (math.prod(factors) == original_input
                    or math.prod(factors)*d == original_input):
                factors.append(d)
                break
            else:
                raise ValueError(f'Polynomial failure as gcd(|x-y|, n) = n.')
    return factors


def crt(moduli: list[int], remainders: list[int]) -> int:
    """
    The Chinese Remainder theorem.

    For a set of pairwise co-prime moduli and a set of remainders,
    there exists one x mod(moduli_product), such that
    x = remainder[i] % moduli[i] for all i.

    Form the set of products of all moduli but one, for each modulus.
    Calculate the modular inverses of these products.
    Sum remainders[i]*products[i]*inverses[i] to form the result.
    This expression necessarily returns remainder[i] % moduli[i].
    """
    for i in moduli:
        for j in moduli:
            if math.gcd(i, j) != 1:
                raise ValueError('Moduli must be pairwise coprime.')
    moduli_product = math.prod(moduli)
    products = [moduli_product // m for m in moduli]
    inverses = []
    for i in range(len(moduli)):
        inverses.append(modular_inv(products[i], moduli[i]))
    x = 0
    for i in range(len(moduli)):
        x += remainders[i]*products[i]*inverses[i]
    x = x % moduli_product
    return x
