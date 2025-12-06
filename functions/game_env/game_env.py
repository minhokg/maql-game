import logging
from typing import Union

from functions.helper.setup_logging import setup_logging
from functions.models.multi_agent_lola import MultiAgentLOLA
from functions.models.multi_agent_q_learning import MultiAgentQLearning


class GameEnv:
    """
    A class to represent a game environment where two agents (using Q-learning or LOLA) interact.

    This environment simulates a repeated game where two agents choose actions (Cooperate or Defect) in
    each episode. The agents are rewarded based on their actions according to a predefined payoff matrix.
    The agents' policies (either Q-learning or LOLA) are updated after each episode.

    Attributes:
        actions (list): List of possible actions for each agent (default is ['C', 'D'] for Cooperate and Defect).
        payoff_matrix (dict): A dictionary defining the rewards for each pair of actions taken by the agents.

    Methods:
        simulate_game(agent1, agent2, n_episodes):
            Simulates a game between two agents for a given number of episodes, updating their policies after each episode.

    """

    def __init__(self, actions: list = None) -> None:
        """
        Initialize the game environment with the specified actions and sets up the payoff matrix.

        Args:
            actions (list, optional): A list of actions available to the agents. Defaults to ['C', 'D'].

        """
        # If no actions are provided, set default actions to 'C' and 'D'
        if actions is None:
            actions = ["C", "D"]
        self.actions = actions  # Each agent has two possible actions: C (Cooperate), D (Defect)

        # Define the payoff matrix where the keys are action pairs (action1, action2) and values are rewards for each agent
        self.payoff_matrix = {
            ("C", "C"): (6, 6),  # Both cooperate: both get 6
            ("C", "D"): (2, 10),  # Agent 1 cooperates, Agent 2 defects: Agent 1 gets 2, Agent 2 gets 10
            ("D", "C"): (10, 2),  # Agent 1 defects, Agent 2 cooperates: Agent 1 gets 10, Agent 2 gets 2
            ("D", "D"): (4, 4),  # Both defect: both get 4
        }

    def simulate_game(self, agent1: Union[MultiAgentQLearning, MultiAgentLOLA], agent2: Union[MultiAgentQLearning, MultiAgentLOLA], n_episodes: int) -> None:
        """
        Simulate a game between two agents for a given number of episodes.
        During each episode, the agents choose actions based on their current policies,
        update their Q-values or LOLA parameters, and receive rewards based on the actions taken.

        Args:
            agent1 (Union[MultiAgentQLearning, MultiAgentLOLA]): The first agent in the simulation, either using Q-learning or LOLA.
            agent2 (Union[MultiAgentQLearning, MultiAgentLOLA]): The second agent in the simulation, either using Q-learning or LOLA.
            n_episodes (int): The number of episodes to run the simulation for.

        Returns:
            None: The method updates the agents' policies and records their actions but does not return any values.

        """
        # Set up logging for this simulation
        setup_logging()

        # Simulate the game for a number of episodes
        for _ in range(n_episodes):
            # Agents choose their actions based on their policy
            action1 = agent1.choose_action()
            action2 = agent2.choose_action()

            # Record the actions taken in this episode for both agents
            agent1.record_history(action1)
            agent2.record_history(action2)

            # Retrieve the rewards from the payoff matrix based on the chosen actions
            reward1, reward2 = self.payoff_matrix[(action1, action2)]

            # Update the Q-values for Agent 1 based on their action and reward
            if isinstance(agent1, MultiAgentQLearning):
                # If agent1 is using Q-learning, update based on action1 and reward1
                agent1.update(action1, reward1, action2)
            else:
                # If agent1 is using LOLA, update using agent2's Q-table (opponent's Q-values)
                opponent_q_table = agent2.q_table
                agent1.update(action1, reward1, action2, opponent_q_table)

            # Update the Q-values for Agent 2 based on their action and reward
            if isinstance(agent2, MultiAgentQLearning):
                # If agent2 is using Q-learning, update based on action2 and reward2
                agent2.update(action2, reward2, action1)
            else:
                # If agent2 is using LOLA, update using agent1's Q-table (opponent's Q-values)
                opponent_q_table = agent1.q_table
                agent2.update(action2, reward2, action1, opponent_q_table)

        # After training is done, both agents choose a final action
        final_action1 = agent1.choose_action()
        final_action2 = agent2.choose_action()

        # Record the final actions for both agents
        agent1.record_history(final_action1)
        agent2.record_history(final_action2)

        # Log the final actions taken after all episodes
        logging.info(f"Final actions after training: Agent 1 chooses {final_action1}, Agent 2 chooses {final_action2}")
