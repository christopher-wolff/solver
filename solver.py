import json
import mlx.core as mx


def compute_ev(
    p1_logits: mx.array,
    p2_logits: mx.array,
    cards: mx.array,
    bets: mx.array,
    card_win: mx.array,
) -> mx.array:
    """Return P1 EV for given logits."""

    p1_prob = mx.softmax(p1_logits, axis=1)
    p2_prob = mx.sigmoid(p2_logits)
    n_cards = len(cards)

    total = mx.array(0.0)
    for j in range(len(bets)):
        bet = bets[j]
        call_prob = p2_prob[:, j]
        payoff_if_call = card_win * (1 + bet) - (1 - card_win) * bet
        payoff_j = (1 - call_prob)[None, :] * 1 + call_prob[None, :] * payoff_if_call
        total += mx.sum(p1_prob[:, j][:, None] * payoff_j) / (n_cards * n_cards)
    return total


def solve(
    num_cards: int,
    num_bets: int,
    bet_max: float,
    iterations: int,
    lr: float,
) -> None:
    cards = mx.linspace(0, 1, num_cards)
    bets = bet_max * mx.linspace(0, 1, num_bets)
    p1_logits = mx.zeros((num_cards, num_bets))
    p2_logits = mx.zeros((num_cards, num_bets))
    card_win = (cards[:, None] > cards[None, :]).astype(mx.float32)

    grad_ev = mx.grad(compute_ev, argnums=[0, 1])
    for i in range(iterations):
        g1, g2 = grad_ev(p1_logits, p2_logits, cards, bets, card_win)
        p1_logits = p1_logits + lr * g1
        p2_logits = p2_logits - lr * g2
        if (i + 1) % max(1, iterations // 10) == 0:
            ev_now = compute_ev(p1_logits, p2_logits, cards, bets, card_win).item()
            print(f"Iter {i + 1}/{iterations}, EV={ev_now:.4f}")

    p1_probs = mx.softmax(p1_logits, axis=1)
    p2_probs = mx.sigmoid(p2_logits)
    ev_final = compute_ev(p1_logits, p2_logits, cards, bets, card_win).item()

    with open("p1_strategy.json", "w") as f:
        json.dump(
            {
                "cards": cards.tolist(),
                "bets": bets.tolist(),
                "strategy": p1_probs.tolist(),
            },
            f,
            indent=2,
        )
    with open("p2_strategy.json", "w") as f:
        json.dump(
            {
                "cards": cards.tolist(),
                "bets": bets.tolist(),
                "strategy": p2_probs.tolist(),
            },
            f,
            indent=2,
        )

    print("Final EV for P1:", ev_final)
    print("Final EV for P2:", -ev_final)
    avg_bet = (p1_probs * bets[None, :]).sum(axis=1)
    print("Average bet per card:")
    for val, b in zip(cards.tolist(), avg_bet.tolist()):
        print(f"  card {val:.3f}: {b:.4f}")


if __name__ == "__main__":
    solve(21, 11, 1.0, 5000, 0.05)
