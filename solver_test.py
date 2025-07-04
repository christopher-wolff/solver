import numpy as np
from scipy.optimize import linprog
import mlx.core as mx

from solver import solve, compute_ev

try:
    mx.set_default_device(mx.Device("cpu"))
except TypeError:
    mx.set_default_device(mx.Device(mx.cpu))


def analytic_solution_value() -> float:
    """Return the Nash value for the simple two-card game."""
    cards = [0, 1]
    bets = [0, 1]
    p1_strats = [(a0, a1) for a0 in [0, 1] for a1 in [0, 1]]
    p2_strats = [(c0, c1) for c0 in [0, 1] for c1 in [0, 1]]
    payoff_matrix = np.zeros((len(p1_strats), len(p2_strats)))
    for i, (a0, a1) in enumerate(p1_strats):
        for j, (c0, c1) in enumerate(p2_strats):
            ev = 0.0
            for x in [0, 1]:
                for y in [0, 1]:
                    prob = 0.25
                    bet = [a0, a1][x]
                    if bet == 0:
                        payoff = 1 if x > y else 0
                    else:
                        call = [c0, c1][y]
                        payoff_if_call = 2 if x > y else -1
                        payoff = (1 - call) + call * payoff_if_call
                    ev += prob * payoff
            payoff_matrix[i, j] = ev
    m, n = payoff_matrix.shape
    c = np.zeros(m + 1)
    c[-1] = -1
    A_ub = np.hstack([-payoff_matrix.T, np.ones((n, 1))])
    A_eq = np.ones((1, m + 1))
    A_eq[0, -1] = 0
    res = linprog(c, A_ub=A_ub, b_ub=np.zeros(n), A_eq=A_eq, b_eq=np.array([1.0]))
    return -res.fun


def test_convergence_small_game():
    target = analytic_solution_value()
    p1, p2 = solve(num_cards=2, num_bets=2, bet_max=1.0, iterations=2000)
    cards = mx.linspace(0, 1, 2)
    bets = mx.linspace(0, 1, 2)
    win_matrix = (cards[:, None] > cards[None, :]).astype(mx.float32)
    value = compute_ev(
        p1_strategy=p1,
        p2_strategy=p2,
        cards=cards,
        bets=bets,
        win_matrix=win_matrix,
    ).item()
    assert abs(value - target) <= 0.05
