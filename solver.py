"""Counterfactual Regret Minimization solver for a simple poker game."""

from typing import Tuple
from tqdm import tqdm
import json
import mlx.core as mx


def compute_ev(
    *,
    p1_strategy: mx.array,
    p2_strategy: mx.array,
    cards: mx.array,
    bets: mx.array,
    win_matrix: mx.array,
) -> mx.array:
    """Return the expected value for player 1 given the strategies."""

    n_cards = len(cards)
    total = mx.sum(p1_strategy[:, 0][:, None] * win_matrix)

    for bet_idx in range(1, len(bets)):
        bet_size = bets[bet_idx]
        call_prob = p2_strategy[:, bet_idx - 1, 1]
        payoff_if_call = win_matrix * (1 + bet_size) - (1 - win_matrix) * bet_size
        payoff = (1 - call_prob)[None, :] + call_prob[None, :] * payoff_if_call
        total += mx.sum(p1_strategy[:, bet_idx][:, None] * payoff)

    return total / (n_cards * n_cards)


def regret_matching(regrets: mx.array) -> mx.array:
    """Return a probability distribution proportional to positive regrets."""
    positive = mx.maximum(regrets, 0)
    total = positive.sum(axis=-1, keepdims=True)
    num_actions = regrets.shape[-1]
    uniform = mx.ones_like(positive) / num_actions
    return mx.where(total > 0, positive / total, uniform)


def solve(
    num_cards: int,
    num_bets: int,
    bet_max: float,
    iterations: int,
) -> Tuple[mx.array, mx.array]:
    """Run CFR for the discretised game and return average strategies."""
    cards = mx.linspace(0, 1, num_cards)
    bets = bet_max * mx.linspace(0, 1, num_bets)
    win_matrix = (cards[:, None] > cards[None, :]).astype(mx.float32)

    p1_regrets = mx.zeros((num_cards, num_bets))
    p1_strategy_total = mx.zeros_like(p1_regrets)

    p2_regrets = mx.zeros((num_cards, num_bets - 1, 2))
    p2_strategy_total = mx.zeros_like(p2_regrets)

    progress = tqdm(range(iterations))
    for i in progress:
        p1_strategy = regret_matching(p1_regrets)
        p2_strategy = regret_matching(p2_regrets)

        p1_strategy_total += p1_strategy
        p2_strategy_total += p2_strategy

        p1_action_utilities = mx.zeros_like(p1_regrets)
        p1_action_utilities[:, 0] = win_matrix.mean(axis=1)
        for bet_idx in range(1, num_bets):
            bet_size = bets[bet_idx]
            call_prob = p2_strategy[:, bet_idx - 1, 1]
            payoff_if_call = win_matrix * (1 + bet_size) - (1 - win_matrix) * bet_size
            payoff = (1 - call_prob)[None, :] + call_prob[None, :] * payoff_if_call
            p1_action_utilities[:, bet_idx] = payoff.mean(axis=1)
        p1_expected_utility = (p1_strategy * p1_action_utilities).sum(axis=1, keepdims=True)
        p1_regrets += p1_action_utilities - p1_expected_utility

        # utilities for P2 actions
        p2_call_utilities = mx.zeros((num_cards, num_bets - 1))
        p2_fold_utilities = mx.zeros((num_cards, num_bets - 1))
        for bet_idx in range(1, num_bets):
            bet_size = bets[bet_idx]
            p1_prob = p1_strategy[:, bet_idx]
            payoff_if_call = win_matrix * (1 + bet_size) - (1 - win_matrix) * bet_size
            payoff_p2_call = -payoff_if_call
            p2_call_utilities[:, bet_idx - 1] = (p1_prob[:, None] * payoff_p2_call).sum(axis=0) / num_cards
            p2_fold_utilities[:, bet_idx - 1] = -p1_prob.sum() / num_cards
        p2_expected_utility = (
            p2_strategy[:, :, 1] * p2_call_utilities
            + p2_strategy[:, :, 0] * p2_fold_utilities
        )
        p2_regrets[:, :, 1] += p2_call_utilities - p2_expected_utility
        p2_regrets[:, :, 0] += p2_fold_utilities - p2_expected_utility

        if (i + 1) % max(1, iterations // 10) == 0:
            ev_now = compute_ev(
                p1_strategy=p1_strategy,
                p2_strategy=p2_strategy,
                cards=cards,
                bets=bets,
                win_matrix=win_matrix,
            ).item()
            nash_distance = float(
                mx.maximum(p1_regrets, 0).sum() + mx.maximum(p2_regrets, 0).sum()
            ) / float(i + 1)
            progress.set_description(f"ΔN {nash_distance:.4f}")
            progress.set_postfix(ev=f"{ev_now:.4f}")

    p1_avg = p1_strategy_total / p1_strategy_total.sum(axis=1, keepdims=True)
    p2_avg = p2_strategy_total / p2_strategy_total.sum(axis=2, keepdims=True)
    ev_final = compute_ev(
        p1_strategy=p1_avg,
        p2_strategy=p2_avg,
        cards=cards,
        bets=bets,
        win_matrix=win_matrix,
    ).item()

    with open("p1_strategy.json", "w") as f:
        json.dump(
            {
                "cards": cards.tolist(),
                "bets": bets.tolist(),
                "strategy": p1_avg.tolist(),
            },
            f,
            indent=2,
        )
    with open("p2_strategy.json", "w") as f:
        json.dump(
            {
                "cards": cards.tolist(),
                "bets": bets[1:].tolist(),
                "strategy": p2_avg.tolist(),
            },
            f,
            indent=2,
        )

    print("Final EV for P1:", ev_final)
    print("Final EV for P2:", -ev_final)
    avg_bet = (p1_avg * bets[None, :]).sum(axis=1)
    print("Average bet per card:")
    for val, b in zip(cards.tolist(), avg_bet.tolist()):
        print(f"  card {val:.3f}: {b:.4f}")

    return p1_avg, p2_avg


if __name__ == "__main__":
    solve(num_cards=21, num_bets=11, bet_max=1.0, iterations=5000)
