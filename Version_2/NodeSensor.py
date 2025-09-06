"""
NodeSensor.py

Defines the sensing capabilities of a node.
This class simulates a 360-degree sensor that can detect other nodes within a certain
range. It incorporates noise and probabilities of false negatives/positives to
model real-world sensor imperfections.
"""
import numpy as np
from typing import TYPE_CHECKING, List, Dict, Any

# Prevents circular import issues while allowing type hinting
if TYPE_CHECKING:
    from NodePosition import Node_Position

class Node_Sensor:
    """
    Simulates a sensor for a FLOWRRA node.

    Attributes:
        max_range (float): The maximum distance the sensor can detect other nodes.
        std_noise (float): Standard deviation of Gaussian noise added to signal strength.
        fn_prob (float): Probability of a false negative (failing to detect a node in range).
        fp_prob (float): Probability of a false positive (detecting a phantom node).
    """
    def __init__(self,
                 max_range: float = 0.2,
                 std_noise: float = 0.01,
                 false_negative_prob: float = 0.01,
                 false_positive_prob: float = 0.005):
        self.max_range = max_range
        self.std_noise = std_noise
        self.fn_prob = false_negative_prob
        self.fp_prob = false_positive_prob

    def sense(self, self_node: 'Node_Position', all_nodes: List['Node_Position'], dt: float = 1.0) -> List[Dict[str, Any]]:
        """
        Performs a 360-degree sense action, returning a list of detections.

        Each detection includes the relative position, velocity, and signal strength
        of a detected node.

        Args:
            self_node (Node_Position): The node that is performing the sensing.
            all_nodes (List[Node_Position]): A list of all nodes in the environment.
            dt (float): Timestep delta for velocity calculations.

        Returns:
            A list of detection dictionaries.
        """
        detections = []
        for other_node in all_nodes:
            if other_node.id == self_node.id:
                continue  # Don't sense yourself

            delta_pos = other_node.pos - self_node.pos
            distance = np.linalg.norm(delta_pos)

            # Check if the other node is within range
            if distance > self.max_range:
                continue

            # Simulate false negatives
            if np.random.rand() < self.fn_prob:
                continue

            # Calculate relative kinematics
            bearing_rad = np.arctan2(delta_pos[1], delta_pos[0])
            relative_velocity = other_node.velocity(dt=dt) - self_node.velocity(dt=dt)

            # Signal strength is inversely proportional to distance, with noise
            signal_strength = (1.0 / (distance + 1e-6)) + np.random.normal(0, self.std_noise)

            detections.append({
                'id': other_node.id,
                'pos': other_node.pos.copy(),
                'distance': float(distance),
                'bearing_rad': float(bearing_rad),
                'signal': float(signal_strength),
                'velocity': relative_velocity,
                'angle_idx': self_node.angle_idx
            })

        # Simulate false positives
        if np.random.rand() < self.fp_prob:
            phantom_distance = np.random.rand() * self.max_range
            phantom_bearing = np.random.rand() * 2 * np.pi
            phantom_pos = self_node.pos + np.array([
                phantom_distance * np.cos(phantom_bearing),
                phantom_distance * np.sin(phantom_bearing)
            ])
            detections.append({
                'id': -1,  # Phantom node ID
                'pos': phantom_pos,
                'distance': float(phantom_distance),
                'bearing_rad': float(phantom_bearing),
                'signal': 0.1 + np.random.rand() * 0.2, # Weak signal
                'velocity': np.random.randn(2) * 0.01, # Random slow drift
                'angle_idx': self_node.angle_idx
            })

        return detections
