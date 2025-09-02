import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

class EnvironmentAVisualizer:
    def __init__(self, env_a):
        self.env_a = env_a

    def render(self):
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_xlim(0, self.env_a.grid_size)
        ax.set_ylim(0, self.env_a.grid_size)
        ax.set_title("Environment-A: Moving Disks with Dynamic Eye Direction & Loop")
        ax.set_xticks([])
        ax.set_yticks([])

        nodes = self.env_a.nodes
        num_nodes = len(nodes)

        for i in range(num_nodes):
            node = nodes[i]
            x_center, y_center = node['pos'][0] + 0.5, node['pos'][1] + 0.5

            # Draw Disk
            disk_patch = patches.Circle((x_center, y_center), 0.4, fc='skyblue', ec='blue', lw=0.8, alpha=0.9)
            ax.add_patch(disk_patch)

            # Draw Arrow
            angle_idx = node['eye_angle_idx']
            angle_deg = angle_idx * (360 / self.env_a.angle_steps)
            rad = np.deg2rad(angle_deg)
            dx, dy = 0.3 * np.cos(rad), 0.3 * np.sin(rad)

            arrow_patch = patches.FancyArrow(x_center, y_center, dx, dy,
                                             width=0.05, head_width=0.3, head_length=0.3,
                                             fc='darkorange', ec='red', lw=1.0, zorder=2)
            ax.add_patch(arrow_patch)

        for i in range(num_nodes):
            current_node = nodes[i]
            next_node = nodes[(i + 1) % num_nodes]
            x1_center, y1_center = current_node['pos'][0] + 0.5, current_node['pos'][1] + 0.5
            x2_center, y2_center = next_node['pos'][0] + 0.5, next_node['pos'][1] + 0.5

            ax.plot([x1_center, x2_center], [y1_center, y2_center], 'k:', alpha=0.4, lw=1.0, zorder=1)

        plt.gca().set_aspect('equal', adjustable='box')
        plt.tight_layout()
        plt.show()
