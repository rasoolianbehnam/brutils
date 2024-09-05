import numpy as np
import numba as nu

from functools import partial

njit = nu.njit
pnjit = partial(njit, parallel=True)
ϕ = (1 + np.sqrt(5)) / 2


@njit
def diff_complex(f, x, h=np.finfo(np.float64).resolution):
    h = h**2
    y = x + h * 1j
    out = f(y)
    return out.real, out.imag / h


@njit
def bracket_minimum(f, x=0, s=1e-2, k=2.0, max_iterations=int(1e5)):
    a, ya = x, f(x)
    b, yb = x + s, f(x + s)
    if yb > ya:
        a, ya, b, yb, s = b, yb, a, ya, -s
    for i in range(max_iterations):
        c, yc = b + s, f(b + s)
        if yc > yb:
            return (a, c) if a < c else (c, a)
        a, ya, b, yb, s = b, yb, c, yc, s * k


@njit
def fibonacci_search(f, a, b, n, ϵ=0.01):
    s = (1 - np.sqrt(5)) / (1 + np.sqrt(5))
    ρ = 1 / (ϕ * (1 - s ** (n + 1)) / (1 - s**n))
    d = ρ * b + (1 - ρ) * a
    yd = f(d)
    for i in range(1, n):
        if i == n - 1:
            c = ϵ * a + (1 - ϵ) * d
        else:
            c = ρ * a + (1 - ρ) * b
        yc = f(c)
        if yc < yd:
            b, d, yd = d, c, yc
        else:
            a, b = b, c
        ρ = 1 / (ϕ * (1 - s ** (n - i + 1)) / (1 - s ** (n - i)))
    return (a, b) if a < b else (b, a)


@nu.njit
def golden_section_search(f, a, b, n):
    ρ = ϕ - 1
    d = ρ * b + (1 - ρ) * a
    yd = f(d)
    for i in range(1, n):
        c = ρ * a + (1 - ρ) * b
        yc = f(c)
        if yc < yd:
            b, d, yd = d, c, yc
        else:
            a, b = b, c
    return (a, b) if a < b else (b, a)


def line_search(f, x, d):
    @nu.njit
    def objective(a):
        return f(x + a * d)

    a, b = bracket_minimum(objective)
    a, b = golden_section_search(objective, a, b, 100)
    α = (a + b) / 2
    return x + α * d


@nu.njit
def backgracking_line_search(f, Δf, x, d, α, p=0.5, β=1e-4):
    y, g = f(x), Δf(x)
    while f(x + α * d) > y + β * α * ((g * d).sum()):
        α *= p
    return α


@pnjit
def dot(a, b):
    return (a * b).sum()


@njit
def strong_backtracking(f, Δ, x, d, α=1, β=1e-4, σ=0.1):
    y0, g0, y_prev, α_prev = f(x), dot(Δ(x), d), np.nan, 0
    αlo, αhi = np.nan, np.nan
    while True:
        y = f(x + α * d)
        if y > y0 + β * α * g0 or (~np.isnan(y_prev) and y >= y_prev):
            αlo, αhi = α_prev, α
            break
        g = dot(Δ(x + α * d), d)
        if np.abs(g) <= -σ * g0:
            return α
        elif g >= 0:
            αlo, αhi = α, α_prev
        y_prev, α_prev, α = y, α, 2 * α
    ylo = f(x + αlo * d)
    while True:
        α = (αlo + αhi) / 2
        y = f(x + α * d)
        if y > y0 + β * α * g0 or y >= ylo:
            αhi = α
        else:
            g = dot(Δ(x + α * d), d)
            if abs(g) <= -σ * g0:
                return α
            elif g * (αhi - αlo) >= 0:
                αhi = αlo
            αlo = α
