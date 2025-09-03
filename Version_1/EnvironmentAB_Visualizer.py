import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

class EnvironmentABVisualizer:
    """
    Visualizes the states of Environment A and B.
    """
    def __init__(self, env_b):
        # We now only bind to the static env_b
        self.env_b = env_b

    def render(self, nodes_to_render, grid_size, angle_steps, visited_cells_b, 
           title="Environment State", save=False, filename=None, ax=None):
        """
        Renders the given nodes on top of Environment B.
        If ax is provided, draw into it (for animations).
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 8))

        ax.clear()
        ax.set_xlim(0, grid_size)
        ax.set_ylim(0, grid_size)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])

        # --- Everything else (drawing obstacles, agents, arrows, loop) ---
        for (x, y) in self.env_b.fixed_blocks:
            rect = patches.Rectangle((x, y), 1, 1, fc='black', alpha=0.8)
            ax.add_patch(rect)
        for (x, y) in self.env_b.moving_blocks:
            rect = patches.Rectangle((x, y), 1, 1, fc='darkred', alpha=0.8)
            ax.add_patch(rect)

        for (x, y) in visited_cells_b:
            rect = patches.Rectangle((x, y), 1, 1, fc='lightgreen', alpha=0.2)
            ax.add_patch(rect)

        for i, node in enumerate(nodes_to_render):
            x_center, y_center = node.pos[0] + 0.5, node.pos[1] + 0.5
            disk_patch = patches.Circle((x_center, y_center), 0.4, fc='skyblue', ec='blue', lw=0.8, alpha=0.9)
            ax.add_patch(disk_patch)

            angle_idx = node.eye_angle_idx
            angle_deg = angle_idx * (360 / angle_steps)
            rad = np.deg2rad(angle_deg)
            dx, dy = 0.3 * np.cos(rad), 0.3 * np.sin(rad)
            arrow_patch = patches.FancyArrow(x_center, y_center, dx, dy, width=0.05,
                                             head_width=0.3, head_length=0.3,
                                             fc='darkorange', ec='red', lw=1.0, zorder=2)
            ax.add_patch(arrow_patch)

        num_nodes = len(nodes_to_render)
        for i in range(num_nodes):
            curr = nodes_to_render[i].pos
            next_ = nodes_to_render[(i + 1) % num_nodes].pos
            x1, y1 = curr[0] + 0.5, curr[1] + 0.5
            x2, y2 = next_[0] + 0.5, next_[1] + 0.5
            ax.plot([x1, x2], [y1, y2], 'k:', alpha=0.4, lw=1.0)

        if save and filename:
            plt.savefig(filename)
            plt.close()
        elif ax is None:  # interactive only if not in animation
            plt.show()

