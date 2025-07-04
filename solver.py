import json
import mlx.core as mx
from typing import Tuple


def compute_ev(
    p1_strategy: mx.array,
    p2_strategy: mx.array,
    cards: mx.array,
    bets: mx.array,
    card_win: mx.array,
) -> mx.array:
    """Return P1 EV for given strategies."""

    n_cards = len(cards)
    total = mx.array(0.0)

    # check action (bet 0) has no response from P2
    payoff_check = card_win
    total += mx.sum(p1_strategy[:, 0][:, None] * payoff_check) / (n_cards * n_cards)

    # positive bets have a P2 response
    for j in range(1, len(bets)):
        bet = bets[j]
        call_prob = p2_strategy[:, j - 1, 1]
        payoff_if_call = card_win * (1 + bet) - (1 - card_win) * bet
        payoff_j = (1 - call_prob)[None, :] * 1 + call_prob[None, :] * payoff_if_call
        total += mx.sum(p1_strategy[:, j][:, None] * payoff_j) / (n_cards * n_cards)
    return total


def solve(
    num_cards: int,
    num_bets: int,
    bet_max: float,
    iterations: int,
) -> Tuple[mx.array, mx.array]:
    cards = mx.linspace(0, 1, num_cards)
    bets = bet_max * mx.linspace(0, 1, num_bets)
    card_win = (cards[:, None] > cards[None, :]).astype(mx.float32)

    # strategies and regret buffers
    p1_regret = mx.zeros((num_cards, num_bets))
    p1_strategy_sum = mx.zeros_like(p1_regret)

    p2_regret = mx.zeros((num_cards, num_bets - 1, 2))
    p2_strategy_sum = mx.zeros_like(p2_regret)

    def regret_matching(regrets: mx.array) -> mx.array:
        pos = mx.maximum(regrets, 0)
        total = pos.sum(axis=-1, keepdims=True)
        num_actions = regrets.shape[-1]
        uniform = mx.ones_like(pos) * (1.0 / num_actions)
        return mx.where(total > 0, pos / total, uniform)

    for i in range(iterations):
        p1_strategy = regret_matching(p1_regret)
        p2_strategy = regret_matching(p2_regret)

        p1_strategy_sum += p1_strategy
        p2_strategy_sum += p2_strategy

        # utilities for P1 actions
        util1_actions = mx.zeros_like(p1_regret)
        util1_actions[:, 0] = card_win.mean(axis=1)
        for j in range(1, num_bets):
            bet = bets[j]
            call_prob = p2_strategy[:, j - 1, 1]
            payoff_if_call = card_win * (1 + bet) - (1 - card_win) * bet
            payoff = (1 - call_prob)[None, :] * 1 + call_prob[None, :] * payoff_if_call
            util1_actions[:, j] = payoff.mean(axis=1)
        util1 = (p1_strategy * util1_actions).sum(axis=1, keepdims=True)
        p1_regret += util1_actions - util1

        # utilities for P2 actions
        util2_call = mx.zeros((num_cards, num_bets - 1))
        util2_fold = mx.zeros((num_cards, num_bets - 1))
        for j in range(1, num_bets):
            bet = bets[j]
            p1_prob_j = p1_strategy[:, j]
            payoff_if_call = card_win * (1 + bet) - (1 - card_win) * bet
            payoff_p2_call = -payoff_if_call
            util2_call[:, j - 1] = (p1_prob_j[:, None] * payoff_p2_call).sum(axis=0) / num_cards
            util2_fold[:, j - 1] = -p1_prob_j.sum() / num_cards
        util2 = p2_strategy[:, :, 1] * util2_call + p2_strategy[:, :, 0] * util2_fold
        p2_regret[:, :, 1] += util2_call - util2
        p2_regret[:, :, 0] += util2_fold - util2

        if (i + 1) % max(1, iterations // 10) == 0:
            ev_now = compute_ev(p1_strategy, p2_strategy, cards, bets, card_win).item()
            print(f"Iter {i + 1}/{iterations}, EV={ev_now:.4f}")

    p1_avg = p1_strategy_sum / p1_strategy_sum.sum(axis=1, keepdims=True)
    p2_avg = p2_strategy_sum / p2_strategy_sum.sum(axis=2, keepdims=True)
    ev_final = compute_ev(p1_avg, p2_avg, cards, bets, card_win).item()

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
    solve(21, 11, 1.0, 5000)
