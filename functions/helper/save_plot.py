import os
from typing import List

import matplotlib.pyplot as plt


def save_plot(
    agent1_history: List[int],
    agent2_history: List[int],
    filename: str,
    save_dir: str,
) -> None:
    """
    Save a plot of the moves made by two agents over time, where each move is represented as either
    cooperate (1) or defect (0) action. The plot compares the actions of both agents across rounds.

    Args:
        agent1_history (List[int]): A list of integers representing the moves (Cooperate = 1, Defect = 0) of Agent 1.
        agent2_history (List[int]): A list of integers representing the moves (Cooperate = 1, Defect = 0) of Agent 2.
        filename (str): The name of the file to save the plot as (e.g., "agent_moves.png").
        save_dir (str): The directory where the plot will be saved.

    Returns:
        None: The function saves the plot as a file to the specified directory and does not return any value.

    """
    # Ensure that the save directory exists; if not, create it
    os.makedirs(save_dir, exist_ok=True)

    # Create a range for the rounds based on the length of agent histories
    rounds = range(1, len(agent1_history) + 1)

    # Set up the plot with specific figure size
    plt.figure(figsize=(10, 5))

    # Plot the moves for Agent 1 as green circles connected by lines
    plt.plot(rounds, agent1_history, "go-", label="Agent 1")

    # Plot the moves for Agent 2 as blue circles connected by lines
    plt.plot(rounds, agent2_history, "bo-", label="Agent 2")

    # Set the title and labels for the plot
    plt.title("Moves over Time: Agent 1 vs. Agent 2")
    plt.xlabel("Rounds")
    plt.ylabel("Move (Cooperate = 1, Defect = 0)")

    # Display the legend to differentiate the two agents
    plt.legend()

    # Enable the grid for easier visualization of the plot
    plt.grid(True)

    # Combine the directory path and filename to create the full file path
    plot_path = os.path.join(save_dir, filename)

    # Save the plot as a PNG file to the specified directory with the given filename
    plt.savefig(plot_path)

    # Close the plot to free up memory
    plt.close()

    # Return None (no explicit return is needed)
    return None
