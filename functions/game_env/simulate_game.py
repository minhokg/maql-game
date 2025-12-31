from typing import Callable, List, Tuple


# game simulation
def simulate_game(
    player1_strategy: Callable[[List[str]], str],
    player2_strategy: Callable[[List[str]], str],
    payoff_matrix: dict,
    rounds: int,
    gamma: float = 0.9,
) -> Tuple[List[int], List[int], int, int]:
    """
    Simulate a game of player 1 strategy against player 2 strategy.

    :param player1_strategy: A function that takes a list of opponent moves and returns a string.
    :param player2_strategy: A function that takes a list of opponent moves and returns a string.
    :param payoff_matrix: The payoff matrix to use.
    :param rounds: The number of rounds to simulate.
    :param gamma: The discount factor.
    :return: A list of player 1 moves, a list of player 2 moves, player 1 payoff, and player 2 payoff.
    """
    # initialize cumulative payoffs for both players
    history1, history2 = [], []
    player1_moves, player2_moves = [], []
    player1_payoff, player2_payoff = (
        0,
        0,
    )

    for _round_num in range(rounds):
        # determine the moves of both players
        move1 = player1_strategy(history2)
        move2 = player2_strategy(history1)

        # append the moves to history
        history1.append(move1)
        history2.append(move2)

        # record the moves for plotting
        player1_moves.append(1 if move1 == "C" else 0)  # 1 for cooperate, 0 for defect
        player2_moves.append(1 if move2 == "C" else 0)

        # calculate payoffs
        round_payoff1, round_payoff2 = payoff_matrix[(move1, move2)]

        # discount future payoffs
        discounted_payoff1 = round_payoff1 * (gamma**_round_num)
        discounted_payoff2 = round_payoff2 * (gamma**_round_num)

        # update cumulative payoffs
        player1_payoff += discounted_payoff1
        player2_payoff += discounted_payoff2

    return player1_moves, player2_moves, player1_payoff, player2_payoff
