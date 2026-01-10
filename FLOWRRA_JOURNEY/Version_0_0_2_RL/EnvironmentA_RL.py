"""
EnvironmentA.py

Manages the overall simulation state. It contains the grid, the list of nodes,
and the main step function to advance the simulation time. This class is responsible
for initializing the nodes and applying actions to them.
"""
import numpy as np
import random
from typing import List, Dict, Any
from NodePosition_RL import Node_Position

class Environment_A:
    """
    The main environment class for the FLOWRRA simulation.

    Attributes:
        num_nodes (int): The number of nodes to simulate.
        angle_steps (int): The resolution of angular space for nodes.
        nodes (List[Node_Position]): The list of active node objects.
        loopdata (list): A history of system snapshots for analysis and collapse rollback.
        t (int): The current simulation timestep.
    """
    def __init__(self,
                 num_nodes: int = 10,
                 angle_steps: int = 360,
                 seed: int | None = None):
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        self.num_nodes = num_nodes
        self.angle_steps = angle_steps
        self.nodes: List[Node_Position] = []
        self.loopdata: List[List[Dict[str, Any]]] = []
        self.t: int = 0
        self.seed = seed
        self.reset()
    
    def reset(self):
        """
        Resets the environment, generating new, random positions for all nodes.
        This is called at the start of each new training episode.
        """
        
        self.nodes = []
        for i in range(self.num_nodes):
            node = Node_Position(
                id=i,
                pos=np.random.rand(2), # Assign a random position
                angle_idx=np.random.randint(0, self.angle_steps),
                angle_steps=self.angle_steps
            )
            self.nodes.append(node)
        
        self.loopdata = [self.snapshot_nodes()]
        self.t = 0


    def snapshot_nodes(self) -> List[Dict[str, Any]]:
        """
        Creates a serializable snapshot of the current state of all nodes.
        """
        return [{
            'id': n.id,
            'pos': n.pos.copy(),
            'angle_idx': int(n.angle_idx),
            'velocity': n.velocity()
        } for n in self.nodes]

    def get_node_positions_flat(self) -> List[float]:
        """
        Returns the x,y positions of all nodes in a single flattened list.
        """
        positions = []
        for node in self.nodes:
            positions.extend(node.pos.tolist())
        return positions
    
    def step(self, actions: List[Dict[str, Any]], dt: float = 1.0):
        """
        Advances the simulation by one timestep.
        """
        if len(actions) != len(self.nodes):
            raise ValueError("Number of actions must match number of nodes.")

        for node, act in zip(self.nodes, actions):
            target_angle = act.get('target_angle_idx')
            if target_angle is not None:
                node.target_angle_idx = int(target_angle)
                node.step_rotate_and_move(dt)

        self.t += 1
        self.loopdata.append(self.snapshot_nodes())
