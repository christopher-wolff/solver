"""Counterfactual Regret Minimization solver for a simple poker game."""

import json
import math
from dataclasses import dataclass

from tqdm import tqdm
import mlx.core as mx


@dataclass
class GameTables:
    """Hold the fixed tables describing the discretised game."""

    cards: mx.array
    bets: mx.array
    win_matrix: mx.array
    win_means: mx.array
    payoff_if_call: mx.array


def _build_game(num_cards: int, num_bets: int, bet_max: float) -> GameTables:
    """Return lookup tables for a discretised game."""

    cards = mx.linspace(0, 1, num_cards)
    bets = bet_max * mx.linspace(0, 1, num_bets)
    win_matrix = (cards[:, None] > cards[None, :]).astype(mx.float32)
    win_means = win_matrix.mean(axis=1)
    payoff_if_call = (
        win_matrix[None, :, :] * (1 + bets[:, None, None])
        - (1 - win_matrix[None, :, :]) * bets[:, None, None]
    )

    return GameTables(cards, bets, win_matrix, win_means, payoff_if_call)


def compute_ev(
    *,
    p1_strategy: mx.array,
    p2_strategy: mx.array,
    cards: mx.array,
    bets: mx.array,
    win_matrix: mx.array,
    payoff_if_call: mx.array | None = None,
) -> mx.array:
    """Return the expected value for player 1 given the strategies."""

    n_cards = len(cards)
    total = mx.sum(p1_strategy[:, 0][:, None] * win_matrix)

    if payoff_if_call is None:
        payoff_if_call = (
            win_matrix[None, :, :] * (1 + bets[:, None, None])
            - (1 - win_matrix[None, :, :]) * bets[:, None, None]
        )

    for bet_idx in range(1, len(bets)):
        call_prob = p2_strategy[:, bet_idx - 1, 1]
        payoff = (1 - call_prob)[None, :] + call_prob[None, :] * payoff_if_call[bet_idx]
        total += mx.sum(p1_strategy[:, bet_idx][:, None] * payoff)

    return total / (n_cards * n_cards)


def _regret_matching(regrets: mx.array) -> mx.array:
    """Return a probability distribution proportional to positive regrets."""
    positive = mx.maximum(regrets, 0)
    total = positive.sum(axis=-1, keepdims=True)
    num_actions = regrets.shape[-1]
    uniform = mx.ones_like(positive) / num_actions
    return mx.where(total > 0, positive / total, uniform)


def _player_one_action_utilities(
    p2_strategy: mx.array, payoff_if_call: mx.array, win_means: mx.array
) -> mx.array:
    """Return the utilities for each P1 action."""

    num_cards = len(win_means)
    num_bets = payoff_if_call.shape[0]
    utilities = mx.zeros((num_cards, num_bets))
    utilities[:, 0] = win_means
    for bet_idx in range(1, num_bets):
        call_prob = p2_strategy[:, bet_idx - 1, 1]
        payoff = (1 - call_prob)[None, :] + call_prob[None, :] * payoff_if_call[bet_idx]
        utilities[:, bet_idx] = payoff.mean(axis=1)
    return utilities


def _player_two_action_utilities(
    p1_strategy: mx.array, payoff_if_call: mx.array
) -> tuple[mx.array, mx.array]:
    """Return utilities for P2's call and fold actions."""

    num_cards, num_bets = p1_strategy.shape
    call_utilities = mx.zeros((num_cards, num_bets - 1))
    fold_utilities = mx.zeros((num_cards, num_bets - 1))
    for bet_idx in range(1, num_bets):
        p1_prob = p1_strategy[:, bet_idx]
        payoff_p2_call = -payoff_if_call[bet_idx]
        call_utilities[:, bet_idx - 1] = (
            (p1_prob[:, None] * payoff_p2_call).sum(axis=0) / num_cards
        )
        fold_utilities[:, bet_idx - 1] = -p1_prob.sum() / num_cards
    return call_utilities, fold_utilities


def _update_player_one_regrets(
    regrets: mx.array, strategy: mx.array, utilities: mx.array
) -> None:
    expected = (strategy * utilities).sum(axis=1, keepdims=True)
    regrets += utilities - expected


def _update_player_two_regrets(
    regrets: mx.array,
    strategy: mx.array,
    call_utilities: mx.array,
    fold_utilities: mx.array,
) -> None:
    expected = strategy[:, :, 1] * call_utilities + strategy[:, :, 0] * fold_utilities
    regrets[:, :, 1] += call_utilities - expected
    regrets[:, :, 0] += fold_utilities - expected


def _log_progress(
    iteration: int,
    iterations: int,
    p1_strategy: mx.array,
    p2_strategy: mx.array,
    game: GameTables,
    p1_regrets: mx.array,
    p2_regrets: mx.array,
    progress: tqdm,
) -> None:
    if (iteration + 1) % max(1, iterations // 10) != 0:
        return

    ev_now = compute_ev(
        p1_strategy=p1_strategy,
        p2_strategy=p2_strategy,
        cards=game.cards,
        bets=game.bets,
        win_matrix=game.win_matrix,
        payoff_if_call=game.payoff_if_call,
    ).item()
    nash_distance = float(
        mx.maximum(p1_regrets, 0).sum() + mx.maximum(p2_regrets, 0).sum()
    ) / float(iteration + 1)
    progress.set_description(f"ΔN {nash_distance:.4f}")
    progress.set_postfix(ev=f"{ev_now:.4f}")


def _save_strategies(p1: mx.array, p2: mx.array, game: GameTables) -> None:
    with open("p1_strategy.json", "w") as f:
        json.dump(
            {
                "cards": game.cards.tolist(),
                "bets": game.bets.tolist(),
                "strategy": p1.tolist(),
            },
            f,
            indent=2,
        )
    with open("p2_strategy.json", "w") as f:
        json.dump(
            {
                "cards": game.cards.tolist(),
                "bets": game.bets[1:].tolist(),
                "strategy": p2.tolist(),
            },
            f,
            indent=2,
        )


def _print_summary(ev: float, p1: mx.array, game: GameTables) -> None:
    print("Final EV for P1:", ev)
    print("Final EV for P2:", -ev)
    avg_bet = (p1 * game.bets[None, :]).sum(axis=1)
    print("Average bet per card:")
    for val, b in zip(game.cards.tolist(), avg_bet.tolist()):
        print(f"  card {val:.3f}: {b:.4f}")


def solve(
    num_cards: int,
    num_bets: int,
    bet_max: float,
    iterations: int,
) -> tuple[mx.array, mx.array]:
    """Run CFR for the discretised game and return average strategies."""

    game = _build_game(num_cards, num_bets, bet_max)

    n_cards = len(game.cards)
    n_bets = len(game.bets)

    p1_regrets = mx.zeros((n_cards, n_bets))
    p1_strategy_total = mx.zeros_like(p1_regrets)

    p2_regrets = mx.zeros((n_cards, n_bets - 1, 2))
    p2_strategy_total = mx.zeros_like(p2_regrets)

    initial_distance = math.inf
    progress = tqdm(range(iterations), desc=f"\u0394N {initial_distance:.4f}")
    for i in progress:
        p1_strategy = _regret_matching(p1_regrets)
        p2_strategy = _regret_matching(p2_regrets)

        p1_strategy_total += p1_strategy
        p2_strategy_total += p2_strategy

        p1_utilities = _player_one_action_utilities(
            p2_strategy, game.payoff_if_call, game.win_means
        )
        _update_player_one_regrets(p1_regrets, p1_strategy, p1_utilities)

        call_utils, fold_utils = _player_two_action_utilities(
            p1_strategy, game.payoff_if_call
        )
        _update_player_two_regrets(p2_regrets, p2_strategy, call_utils, fold_utils)

        mx.eval(p1_regrets, p2_regrets, p1_strategy_total, p2_strategy_total)
        mx.clear_cache()

        _log_progress(
            iteration=i,
            iterations=iterations,
            p1_strategy=p1_strategy,
            p2_strategy=p2_strategy,
            game=game,
            p1_regrets=p1_regrets,
            p2_regrets=p2_regrets,
            progress=progress,
        )

    p1_avg = p1_strategy_total / p1_strategy_total.sum(axis=1, keepdims=True)
    p2_avg = p2_strategy_total / p2_strategy_total.sum(axis=2, keepdims=True)
    ev_final = compute_ev(
        p1_strategy=p1_avg,
        p2_strategy=p2_avg,
        cards=game.cards,
        bets=game.bets,
        win_matrix=game.win_matrix,
        payoff_if_call=game.payoff_if_call,
    ).item()

    _save_strategies(p1_avg, p2_avg, game)
    _print_summary(ev_final, p1_avg, game)

    return p1_avg, p2_avg


if __name__ == "__main__":
    solve(num_cards=21, num_bets=11, bet_max=1.0, iterations=10_000)
