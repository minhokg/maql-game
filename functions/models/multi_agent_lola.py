from typing import Dict, List

import numpy as np


class MultiAgentLOLA:
    r"""
    A class that implements the Q-learning algorithm for multi-agent with LOLA \.

    Attributes:
        actions (List[str]): The list of possible actions.
        alpha (float): The learning rate for the Q-learning update rule.
        gamma (float): The discount factor for future rewards.
        epsilon (float): The exploration rate for the epsilon-greedy policy.
        q_table (dict): A dictionary storing the Q-values for each action.
        history (List[int]): A list storing the history of actions taken, encoded as 1 (C) or 0 (D).

    """

    def __init__(
        self,
        actions: List[str],
        alpha: float = 0.1,
        gamma: float = 0.9,
        epsilon: float = 0.1,
        beta: float = 0.2,
        random_seed: int = 42,
    ) -> None:
        """
        Initialize the MultiAgentLOLA agent.

        Args:
            actions (List[str]): A list of possible actions (e.g., ['C', 'D']).
            alpha (float, optional): The learning rate (default is 0.1).
            gamma (float, optional): The discount factor (default is 0.9).
            epsilon (float, optional): The exploration rate (default is 0.1).
            beta (float, optional): Shaping parameter (default is 0.1).
            random_seed (int, optional): The random seed (default is 42).

        """
        self.actions = actions
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.beta = beta
        self.q_table: Dict[str, float] = {action: 0.0 for action in actions}  # Initialize Q-table with float values
        self.history: List[int] = []  # Track history of actions
        np.random.seed(random_seed)

    def choose_action(self) -> str:
        """
        Chooses an action based on the epsilon-greedy policy.

        Returns:
            str: The selected action.

        """
        if np.random.randint(0, 100) < self.epsilon * 100:  # Random number in [0, 100)
            return np.random.choice(self.actions)  # Randomly choose an action
        return max(self.q_table, key=self.q_table.get)  # Choose the action with the highest Q-value

    def update(self, action: str, reward: float, opponent_action: str, opponent_q_table: dict) -> None:
        """
        Update the LOLA Q-learning based on the chosen action, received reward, and opponent q table.

        Args:
            action (str): The action taken by the agent.
            reward (float): The reward received by the agent.
            opponent_action (str): The action taken by the opponent agent.
            opponent_q_table (dict): A dictionary storing the Q-values for each action.

        """
        current_q = self.q_table[action]
        next_q = self.q_table[opponent_action]
        self.q_table[action] = (1 - self.alpha) * current_q + self.alpha * (reward + self.gamma * next_q - current_q)
        shaping_term = self.beta * max(opponent_q_table.values())
        self.q_table[action] += shaping_term

    def record_history(self, action: str) -> None:
        """
        Record the history of the agent's actions, converting 'C' to 1 and 'D' to 0.

        Args:
            action (str): The action taken by the agent ('C' for cooperate, 'D' for defect).

        """
        if action == "C":
            action = 1
        else:
            action = 0
        self.history.append(action)

    def get_history(self) -> List[int]:
        """
        Return the history of actions taken by the agent.

        Returns:
            List[int]: A list of the actions taken, encoded as 1 (C) or 0 (D).

        """
        return self.history
