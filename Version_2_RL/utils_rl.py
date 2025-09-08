"""
Utility functions for visualization.
"""
import matplotlib.pyplot as plt
import numpy as np
import os
from typing import List, Optional
from NodePosition_RL import Node_Position
from EnvironmentB_RL import EnvironmentB
from PIL import Image, ImageDraw, ImageFont
import glob
import re

def plot_system_state_rl(density_field: np.ndarray,
                      nodes: List[Node_Position],
                      env_b: Optional[EnvironmentB],
                      out_path: str,
                      title: str):
    """
    Generates and saves a plot of the current system state, adapted for RL.

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
        ax.imshow(density_field.T, origin='lower', extent=(0, 1, 0, 1),
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

    # Plot the loop line connecting the nodes
    positions = np.array([n.pos for n in nodes])
    if len(positions) > 0:
        # Connect last node to first to close the loop
        loop_positions = np.vstack([positions, positions[0]])
        ax.plot(loop_positions[:, 0], loop_positions[:, 1], 'w--', lw=1, alpha=0.5, zorder=2)

    # Plot nodes and their velocity vectors
    velocities = np.array([n.velocity() for n in nodes])

    ax.scatter(positions[:, 0], positions[:, 1], c='cyan', s=50, ec='white', lw=1, zorder=3)
    
    # Plot velocity arrows
    ax.quiver(positions[:, 0], positions[:, 1], velocities[:, 0], velocities[:, 1],
              color='lime', angles='xy', scale_units='xy', scale=1.0, width=0.005, headwidth=4, headlength=6, zorder=3)

    # Plot text for node ID
    for node in nodes:
        ax.text(node.pos[0] + 0.01, node.pos[1] + 0.01, str(node.id), color='white', fontsize=10, zorder=5)

    ax.set_title(title, color='white')
    ax.set_aspect('equal', adjustable='box')
    ax.axis('off')

    plt.savefig(out_path, facecolor=fig.get_facecolor(), bbox_inches='tight', dpi=100)
    plt.close(fig)

def create_gif(image_folder: str, output_gif: str, pattern: str = "*.png", duration: int = 200):
    """
    Creates an animated GIF from a sequence of PNG images in a folder.
    
    Args:
        image_folder (str): Path to the folder containing PNG images.
        output_gif (str): The name of the output GIF file.
        pattern (str): The file pattern to search for (e.g., "t_*.png").
        duration (int): Duration of each frame in milliseconds.
    """
    img_files = glob.glob(os.path.join(image_folder, pattern))

    # Sort files numerically based on the timestamp in the filename
    img_files.sort(key=lambda f: int(re.search(r"(\d+)\.png$", os.path.basename(f)).group(1)))
    
    if not img_files:
        print(f"No images found matching pattern '{pattern}' in '{image_folder}'.")
        return

    images = [Image.open(f) for f in img_files]
    
    if images:
        images[0].save(
            output_gif,
            save_all=True,
            append_images=images[1:],
            optimize=True,
            duration=duration,
            loop=0
        )
        print(f"GIF successfully created at {output_gif}")
