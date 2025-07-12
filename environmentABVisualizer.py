import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

class EnvironmentABVisualizer:
    def __init__(self, env_a, env_b):
        self.env_a = env_a
        self.env_b = env_b

    def render(self,save=False, filename=None):
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_xlim(0, self.env_a.grid_size)
        ax.set_ylim(0, self.env_a.grid_size)
        ax.set_title("Environment-A + Environment-B Overlay")
        ax.set_xticks([])
        ax.set_yticks([])

        # 🌐 Environment A — agents
        nodes = self.env_a.nodes
        for i, node in enumerate(nodes):
            x_center, y_center = node['pos'][0] + 0.5, node['pos'][1] + 0.5

            # Disk = agent
            disk_patch = patches.Circle((x_center, y_center), 0.4,
                                        fc='skyblue', ec='blue', lw=0.8, alpha=0.9)
            ax.add_patch(disk_patch)

            # Eye arrow
            angle_idx = node['eye_angle_idx']
            angle_deg = angle_idx * (360 / self.env_a.angle_steps)
            rad = np.deg2rad(angle_deg)
            dx, dy = 0.3 * np.cos(rad), 0.3 * np.sin(rad)

            arrow_patch = patches.FancyArrow(x_center, y_center, dx, dy,
                                             width=0.05, head_width=0.3, head_length=0.3,
                                             fc='darkorange', ec='red', lw=1.0, zorder=2)
            ax.add_patch(arrow_patch)

        # Connect loop lines
        num_nodes = len(nodes)
        for i in range(num_nodes):
            curr = nodes[i]['pos']
            next_ = nodes[(i + 1) % num_nodes]['pos']
            x1, y1 = curr[0] + 0.5, curr[1] + 0.5
            x2, y2 = next_[0] + 0.5, next_[1] + 0.5
            ax.plot([x1, x2], [y1, y2], 'k:', alpha=0.4, lw=1.0)

        # 🧱 Environment B — fixed obstacles
        for (x, y) in self.env_b.fixed_blocks:
            rect = patches.Rectangle((x, y), 1, 1, fc='black', alpha=0.8)
            ax.add_patch(rect)

        # 🧊 Environment B — moving blocks
        for (x, y) in self.env_b.moving_blocks:
            rect = patches.Rectangle((x + 0.1, y + 0.1), 0.8, 0.8, fc='crimson', alpha=0.7, lw=1.2, ec='darkred')
            ax.add_patch(rect)

        plt.gca().set_aspect('equal', adjustable='box')
        plt.tight_layout()
        if save and filename:
            plt.savefig(filename)
            plt.close()
        else:
            plt.show()
