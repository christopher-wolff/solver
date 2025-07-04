import unittest
import numpy as np
from scipy.optimize import linprog
import mlx.core as mx
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from solver import solve, compute_ev

class TestCFRConvergence(unittest.TestCase):
    def _analytic_solution_value(self):
        # compute Nash value for two-card single bet game
        cards = [0, 1]
        bets = [0, 1]
        # enumerate pure strategies
        p1_strats = [(a0, a1) for a0 in [0,1] for a1 in [0,1]]
        p2_strats = [(c0, c1) for c0 in [0,1] for c1 in [0,1]]
        A = np.zeros((len(p1_strats), len(p2_strats)))
        for i, (a0, a1) in enumerate(p1_strats):
            for j, (c0, c1) in enumerate(p2_strats):
                ev = 0.0
                for x in [0,1]:
                    for y in [0,1]:
                        prob = 0.25
                        bet = [a0, a1][x]
                        if bet == 0:
                            p1_payoff = 1 if x > y else 0
                        else:
                            call = [c0, c1][y]
                            payoff_if_call = (2 if x>y else -1)
                            p1_payoff = (1-call)*1 + call*payoff_if_call
                        ev += prob * p1_payoff
                A[i,j] = ev
        m, n = A.shape
        c = np.zeros(m+1)
        c[-1] = -1
        A_ub = np.hstack([-A.T, np.ones((n,1))])
        A_eq = np.ones((1,m+1))
        A_eq[0,-1] = 0
        b_eq = np.array([1.0])
        res = linprog(c, A_ub=A_ub, b_ub=np.zeros(n), A_eq=A_eq, b_eq=b_eq)
        return -res.fun

    def test_convergence_small_game(self):
        target_value = self._analytic_solution_value()
        p1, p2 = solve(num_cards=2, num_bets=2, bet_max=1.0, iterations=2000)
        cards = mx.linspace(0, 1, 2)
        bets = mx.linspace(0, 1, 2)
        card_win = (cards[:, None] > cards[None, :]).astype(mx.float32)
        value = compute_ev(p1, p2, cards, bets, card_win).item()
        self.assertAlmostEqual(value, target_value, delta=0.05)

if __name__ == '__main__':
    unittest.main()
