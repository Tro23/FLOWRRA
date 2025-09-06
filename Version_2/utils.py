"""
Utility functions for visualization.
"""
import matplotlib.pyplot as plt
import numpy as np
import os
from typing import List, Optional
from NodePosition import Node_Position
from EnvironmentB import EnvironmentB

def plot_system_state(density_field: np.ndarray,
                      nodes: List[Node_Position],
                      env_b: Optional[EnvironmentB],
                      out_path: str,
                      title: str):
    """
    Generates and saves a plot of the current system state.

    Args:
        density_field (np.ndarray): The repulsion field grid.
        nodes (List[Node_Position]): List of node objects.
        env_b (Optional[EnvironmentB]): The external obstacle environment.
        out_path (str): The file path to save the image.
        title (str): The title for the plot.
    """
    fig, ax = plt.subplots(figsize=(8, 8), facecolor='black')
    ax.set_facecolor('black')

    # Plot density field
    if np.max(density_field) > 0:
        im = ax.imshow(density_field.T, origin='lower', extent=(0, 1, 0, 1),
                       cmap='inferno', interpolation='bicubic', alpha=0.8)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # Plot obstacles from EnvironmentB
    if env_b:
        obstacle_states = env_b.get_obstacle_states()
        fixed_obs = np.array([s['pos'] for s in obstacle_states if s['type'] == 'fixed'])
        moving_obs = np.array([s['pos'] for s in obstacle_states if s['type'] == 'moving'])

        if len(fixed_obs) > 0:
            ax.scatter(fixed_obs[:, 0], fixed_obs[:, 1], c='red', marker='s', s=100, ec='white', label='Fixed Obstacles', zorder=4, alpha=0.9)
        if len(moving_obs) > 0:
            ax.scatter(moving_obs[:, 0], moving_obs[:, 1], c='orange', marker='o', s=100, ec='white', label='Moving Obstacles', zorder=4, alpha=0.9)


    # Plot nodes and their velocity vectors
    positions = np.array([n.pos for n in nodes])
    velocities = np.array([n.velocity() for n in nodes])

    ax.scatter(positions[:, 0], positions[:, 1], c='cyan', s=50, ec='white', lw=1, zorder=3)
    ax.quiver(positions[:, 0], positions[:, 1],
              velocities[:, 0], velocities[:, 1],
              color='lime', scale=0.2, width=0.005, headwidth=4, zorder=2)

    # Title and labels
    ax.set_title(title, color='white', fontsize=14)
    ax.set_xlabel("X", color='white')
    ax.set_ylabel("Y", color='white')
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')

    for spine in ax.spines.values():
        spine.set_edgecolor('white')

    plt.tight_layout()
    plt.savefig(out_path, dpi=100, facecolor='black')
    plt.close(fig)

