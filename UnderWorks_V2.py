# FLOWRRA v2 - Repository scaffold
# Files contained below are separated by markers: "### FILE: <filename>"
# Copy each block into its own .py file (filename shown) to run.

### FILE: NodePosition.py
"""
NodePosition.py
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass

@dataclass
class Node_Position:
    id: int
    pos: np.ndarray           # shape (2,) -> x,y in [0,1)
    angle_idx: int            # discrete angle index
    rotation_speed: float     # angle indices per step
    move_speed: np.ndarray    # shape (2,) maximum per-step displacement
    target_angle_idx: int = None
    last_pos: np.ndarray = None
    angle_steps: int = 360

    def __post_init__(self):
        if self.last_pos is None:
            self.last_pos = self.pos.copy()

    def step_rotate_towards_target(self, dt=1.0):
        if self.target_angle_idx is None:
            return
        # shortest difference on circular index domain
        a = self.angle_idx
        b = self.target_angle_idx
        diff = (b - a + self.angle_steps//2) % self.angle_steps - self.angle_steps//2
        move = np.clip(diff, -self.rotation_speed*dt, self.rotation_speed*dt)
        self.angle_idx = int((self.angle_idx + move) % self.angle_steps)

    def move(self, action_vector: np.ndarray, dt=1.0, bounds='toroidal'):
        displacement = np.clip(action_vector, -self.move_speed*dt, self.move_speed*dt)
        self.last_pos = self.pos.copy()
        self.pos = self.pos + displacement
        if bounds == 'toroidal':
            self.pos = np.mod(self.pos, 1.0)
        elif bounds is not None:
            (xmin,xmax),(ymin,ymax) = bounds
            self.pos[0] = np.clip(self.pos[0], xmin, xmax)
            self.pos[1] = np.clip(self.pos[1], ymin, ymax)

    def velocity(self, dt=1.0):
        return (self.pos - self.last_pos)/dt


### FILE: NodeSensor.py
"""
NodeSensor.py
"""
import numpy as np

class Node_Sensor:
    def __init__(self,
                 horizontal_range: float = 0.25,
                 std_noise: float = 0.01,
                 false_negative_prob: float = 0.01,
                 false_positive_prob: float = 0.01,
                 angle_steps: int = 360):
        self.horizontal_range = horizontal_range
        self.std_noise = std_noise
        self.fn_prob = false_negative_prob
        self.fp_prob = false_positive_prob
        self.angle_steps = angle_steps

    def sense_360(self, node: 'Node_Position', nodes, dt=1.0):
        detections = []
        for other in nodes:
            if other.id == node.id:
                continue
            delta = other.pos - node.pos
            r = np.linalg.norm(delta)
            if r > self.horizontal_range:
                continue
            if np.random.rand() < self.fn_prob:
                continue
            bearing = np.arctan2(delta[1], delta[0])
            v_rel = other.velocity(dt=dt) - node.velocity(dt=dt)
            signal = 1.0 / (r + 1e-6) + np.random.normal(0, self.std_noise)
            detections.append({'id': other.id, 'r': float(r), 'bearing': float(bearing), 'signal': float(signal), 'v_rel': v_rel})
        # false positive
        if np.random.rand() < self.fp_prob:
            detections.append({'id': None, 'r': float(np.random.rand()*self.horizontal_range), 'bearing': float(np.random.rand()*2*np.pi), 'signal': 0.2, 'v_rel': np.zeros(2)})
        return detections


### FILE: EnvironmentA.py
"""
EnvironmentA.py
"""
import numpy as np
from typing import List
from NodePosition import Node_Position
from NodeSensor import Node_Sensor

class Environment_A:
    def __init__(self,
                 grid_size=(64,64),
                 angle_steps=360,
                 num_nodes=8,
                 world_bounds=((0.0,1.0),(0.0,1.0)),
                 node_params=None,
                 seed=None):
        self.grid_size = grid_size
        self.angle_steps = angle_steps
        self.num_nodes = num_nodes
        self.world_bounds = world_bounds
        self.nodes: List[Node_Position] = []
        self.node_sensor = Node_Sensor(angle_steps=angle_steps)
        self.loopdata = []   # history of node snapshots
        self.t = 0
        if seed is not None:
            np.random.seed(seed)
        self.init_nodes(node_params)

    def init_nodes(self, node_params=None):
        self.nodes = []
        for i in range(self.num_nodes):
            pos = np.random.rand(2)
            node = Node_Position(id=i,
                                 pos=pos,
                                 angle_idx=int(np.random.randint(0, self.angle_steps)),
                                 rotation_speed=1.0,
                                 move_speed=np.array([0.02,0.02]),
                                 angle_steps=self.angle_steps)
            self.nodes.append(node)
        self.loopdata = [self.snapshot_nodes()]

    def snapshot_nodes(self):
        # return compact serializable snapshot
        return [{'id': n.id, 'pos': n.pos.copy(), 'angle_idx': int(n.angle_idx)} for n in self.nodes]

    def calculate_angle_idx(self, angle_rad):
        idx = int(round((angle_rad % (2*np.pi)) / (2*np.pi) * self.angle_steps)) % self.angle_steps
        return idx

    def step(self, actions, dt=1.0):
        for node, act in zip(self.nodes, actions):
            target = act.get('target_angle_idx')
            if target is not None:
                node.target_angle_idx = int(target)
                node.step_rotate_towards_target(dt)
            move_vec = act.get('move', np.zeros(2))
            node.move(move_vec, dt, bounds='toroidal')
        self.t += 1
        self.loopdata.append(self.snapshot_nodes())


### FILE: DensityFunctionEstimator.py
"""
Repulsive Density Estimator - vectorized
"""
import numpy as np
from typing import Sequence

class Density_Function_Estimator:
    def __init__(self,
                 grid_shape=(64,64),
                 world_bounds=((0.,1.),(0.,1.)),
                 bandwidth=0.05,
                 repulsion_sigma=0.02,
                 repulsion_strength=1.0):
        self.grid_shape = grid_shape
        self.world_bounds = world_bounds
        self.bandwidth = bandwidth
        self.repulsion_sigma = repulsion_sigma
        self.repulsion_strength = repulsion_strength
        self.grid_x, self.grid_y = self._make_grid()
        self.last_density = np.zeros(self.grid_shape)
        self.accumulated_density = np.zeros(self.grid_shape)

    def _make_grid(self):
        xs = np.linspace(self.world_bounds[0][0], self.world_bounds[0][1], self.grid_shape[0])
        ys = np.linspace(self.world_bounds[1][0], self.world_bounds[1][1], self.grid_shape[1])
        gx, gy = np.meshgrid(xs, ys, indexing='xy')
        return gx, gy

    def _gaussian_kernel_vec(self, positions: np.ndarray, h: float):
        # positions: (N,2)
        gx = self.grid_x[None, :, :]
        gy = self.grid_y[None, :, :]
        px = positions[:,0][:,None,None]
        py = positions[:,1][:,None,None]
        dx = gx - px
        dy = gy - py
        r2 = dx*dx + dy*dy
        coeff = 1.0 / (2*np.pi*h*h)
        return coeff * np.exp(-0.5 * r2 / (h*h))

    def evaluate_density(self, node_positions: Sequence[np.ndarray], node_speeds=None):
        if len(node_positions) == 0:
            self.last_density = np.zeros(self.grid_shape)
            return self.last_density
        pos_arr = np.array(node_positions)
        h = self.bandwidth
        K = self._gaussian_kernel_vec(pos_arr, h)  # shape (N, gx, gy)
        weights = np.ones((pos_arr.shape[0],1,1))
        if node_speeds is not None:
            sp = np.array(node_speeds)
            sp = sp.reshape(-1,1,1)
            weights += 0.5 * np.tanh(sp)
        D = np.sum(weights * K, axis=0)
        # repulsive field (instantaneous negative bumps)
        Rk = self._gaussian_kernel_vec(pos_arr, self.repulsion_sigma)
        R = np.sum(Rk, axis=0)
        total = D - self.repulsion_strength * R
        total = np.clip(total, a_min=0.0, a_max=None)
        # normalize to probability mass for metrics
        s = total.sum()
        if s > 0:
            total = total / s
        self.last_density = total
        self.accumulated_density += total
        return total

    def density_to_features(self, density=None):
        if density is None:
            density = self.last_density
        flat = density.flatten()
        s = flat.sum()
        if s <= 0:
            return {'mean':0.0,'var':0.0,'entropy':0.0,'peaks':[]}
        mean = float(flat.mean())
        var = float(flat.var())
        probs = flat/flat.sum()
        entropy = -float(np.sum(probs * np.log(probs + 1e-12)))
        topk_idx = np.argsort(flat)[-5:][::-1]
        peaks = [np.unravel_index(int(i), density.shape) for i in topk_idx]
        return {'mean': mean, 'var': var, 'entropy': entropy, 'peaks': peaks}


### FILE: WaveFunctionCollapse.py
"""
WaveFunctionCollapse.py
"""
import numpy as np
from typing import List, Dict

class Wave_Function_Collapse:
    def __init__(self,
                 history_length=500,
                 tail_length=20,
                 collapse_threshold=0.15,
                 tau=5):
        self.history_length = history_length
        self.tail_length = tail_length
        self.history = []  # list of {'t':int,'snapshot':..., 'coherence':float, 'collisions':int, 'block':bool}
        self.collapse_threshold = collapse_threshold
        self.tau = tau  # number of consecutive steps below threshold required for soft collapse

    def append_history(self, snapshot: Dict, coherence: float, collisions: int, block: bool, t: int):
        """
        Append a step to history. `block` is a boolean indicating an immediate hard collision
        with an obstacle (wall/block). If a block event occurs, collapse should be immediate.
        """
        self.history.append({'t': t, 'snapshot': snapshot, 'coherence': float(coherence), 'collisions': int(collisions), 'block': bool(block)})
        if len(self.history) > self.history_length:
            self.history.pop(0)

    def should_collapse(self):
        """
        Collapse logic:
        - If the most recent step has `block==True` -> immediate (hard) collapse.
        - Else if the last `tau` steps all have coherence < collapse_threshold -> soft collapse.
        """
        if len(self.history) == 0:
            return False
        # Hard collapse: immediate block collision at last step
        if bool(self.history[-1].get('block', False)):
            return True
        # Soft collapse: require tau consecutive low-coherence steps
        if len(self.history) < self.tau:
            return False
        window = self.history[-self.tau:]
        coherences = [w['coherence'] for w in window]
        if all(c < self.collapse_threshold for c in coherences):
            return True
        return False

    def select_candidate_tail(self):
        # scan history for best contiguous window of length tail_length by mean coherence
        L = len(self.history)
        if L < self.tail_length:
            return None
        best_score = -np.inf
        best_idx = None
        for i in range(0, L - self.tail_length + 1):
            window = self.history[i:i+self.tail_length]
            score = float(np.mean([w['coherence'] for w in window]))
            # small recency bonus
            score += 0.001 * (i / L)
            if score > best_score:
                best_score = score
                best_idx = i
        if best_idx is None:
            return None
        tail = [self.history[i]['snapshot'] for i in range(best_idx, best_idx + self.tail_length)]
        return tail

    def smooth_tail(self, tail_snapshots):
        if not tail_snapshots:
            return None
        T = len(tail_snapshots)
        N = len(tail_snapshots[0])
        arr = np.zeros((T, N, 2))
        angles = np.zeros((T, N), dtype=int)
        for ti, s in enumerate(tail_snapshots):
            for ni, node in enumerate(s):
                arr[ti, ni, :] = node['pos']
                angles[ti, ni] = int(node.get('angle_idx', 0))
        # gaussian temporal kernel
        sig = max(1.0, T/6.0)
        weights = np.exp(-0.5 * ((np.arange(T) - (T-1)/2.0)**2) / (sig**2))
        weights = weights / (weights.sum() + 1e-12)
        smoothed = (weights[:,None,None] * arr).sum(axis=0)  # (N,2)
        last_angles = angles[-1]
        new_frame = [{'id': ni, 'pos': smoothed[ni], 'angle_idx': int(last_angles[ni])} for ni in range(N)]
        return new_frame

    def collapse_and_reinitialize(self, env: 'Environment_A'):
        candidate_tail = self.select_candidate_tail()
        if candidate_tail is None:
            # fallback: small random jitter
            for n in env.nodes:
                n.pos = np.mod(n.pos + 0.02*(np.random.rand(2)-0.5), 1.0)
            return {'reinit_from': 'random_jitter', 'timestamp': env.t}
        new_loop = self.smooth_tail(candidate_tail)
        if new_loop is None:
            return None
        for ni, nd in enumerate(new_loop):
            env.nodes[ni].pos = np.array(nd['pos'])
            env.nodes[ni].angle_idx = int(nd.get('angle_idx', env.nodes[ni].angle_idx))
        return {'reinit_from': 'tail', 'timestamp': env.t}


### FILE: utils.py
"""
Utility functions: visualization + logging helpers
"""
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_density_and_nodes(density, nodes, out_path=None, title=None):
    fig, ax = plt.subplots(figsize=(6,6))
    im = ax.imshow(density.T, origin='lower', extent=(0,1,0,1), aspect='equal')
    xs = [n.pos[0] for n in nodes]
    ys = [n.pos[1] for n in nodes]
    ax.scatter(xs, ys, c='w', edgecolors='k')
    for n in nodes:
        ax.annotate(str(n.id), xy=(n.pos[0], n.pos[1]), color='w', fontsize=8)
    if title:
        ax.set_title(title)
    fig.colorbar(im, ax=ax)
    plt.tight_layout()
    if out_path is not None:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        plt.savefig(out_path)
        plt.close(fig)
    else:
        plt.show()


### FILE: FLOWRRA.py
"""
Top-level Flowrra class wiring together modules
"""
import logging
import csv
import numpy as np
from NodePosition import Node_Position
from NodeSensor import Node_Sensor
from EnvironmentA import Environment_A
from DensityFunctionEstimator import Density_Function_Estimator
from WaveFunctionCollapse import Wave_Function_Collapse
from utils import plot_density_and_nodes

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('FLOWRRA')

class Flowrra:
    def __init__(self, config=None):
        if config is None:
            config = {}
        self.env = Environment_A(grid_size=config.get('grid_size',(64,64)),
                                 angle_steps=config.get('angle_steps',360),
                                 num_nodes=config.get('num_nodes',8),
                                 seed=config.get('seed',None))
        self.density = Density_Function_Estimator(grid_shape=config.get('grid_size',(64,64)),
                                                  world_bounds=config.get('world_bounds',((0,1),(0,1))),
                                                  bandwidth=config.get('bandwidth',0.05),
                                                  repulsion_sigma=config.get('repulsion_sigma',0.02),
                                                  repulsion_strength=config.get('repulsion_strength',1.0))
        self.wfc = Wave_Function_Collapse(history_length=config.get('history_length',500),
                                          tail_length=config.get('tail_length',20),
                                          collapse_threshold=config.get('collapse_threshold',0.15),
                                          tau=config.get('tau',5),
                                          collision_allowance=config.get('collision_allowance',1))
        self.t = 0
        self.collapse_log = []
        self.coherence_history = []
        self.logfile = config.get('logfile','flowrra_log.csv')
        self.visual_dir = config.get('visual_dir','flowrra_visuals')
        # write csv header
        with open(self.logfile, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['t','coherence','collisions','stability','reward','collapse'])
            writer.writeheader()

    def Generate_Actions(self):
        actions = []
        for n in self.env.nodes:
            move = (np.random.randn(2) * 0.005)
            target_angle = int((n.angle_idx + np.random.randint(-1,2)) % self.env.angle_steps)
            actions.append({'move': move, 'target_angle_idx': target_angle})
        return actions

    def Get_Q_States(self):
        dens = self.density.last_density
        dens_feat = dens.flatten()
        node_feats = []
        for n in self.env.nodes:
            node_feats.extend([n.pos[0], n.pos[1], n.angle_idx])
        state = np.concatenate([dens_feat, np.array(node_feats)])
        return state

    def Apply_Action(self, actions):
        self.env.step(actions)

    def Coherence_Metric(self):
        dens = self.density.last_density
        s = dens.sum()
        if s <= 0:
            return 0.0
        peak_frac = float(dens.max())
        flat = dens.flatten()
        probs = flat / (flat.sum() + 1e-12)
        entropy = -float(np.sum(probs * np.log(probs + 1e-12)))
        coherence = peak_frac * (1.0 / (1.0 + entropy))
        return float(coherence)

    def Score_Environment(self):
        coherence = self.Coherence_Metric()
        collisions = self._count_collisions()
        stability = self._compute_loop_stability()
        return {'coherence': coherence, 'collisions': collisions, 'stability': stability}

    def _count_collisions(self, min_dist=0.01):
        pts = np.array([n.pos for n in self.env.nodes])
        if len(pts) < 2:
            return 0
        dists = np.sqrt(((pts[:,None,:] - pts[None,:,:])**2).sum(-1))
        near = np.sum((dists < min_dist)) - len(pts)
        return int(max(near//2, 0))

    def _compute_loop_stability(self):
        if len(self.env.loopdata) < 2:
            return 1.0
        last = self.env.loopdata[-1]
        prev = self.env.loopdata[-2]
        diffs = [np.linalg.norm(last[i]['pos'] - prev[i]['pos']) for i in range(len(last))]
        return float(1.0 / (1.0 + np.mean(diffs)))

    def Compute_Reward(self, scores):
        reward = scores['coherence'] - 0.5*float(scores['collisions']) + 0.1*float(scores['stability'])
        return float(reward)

    def Record_Collapse_State(self, metadata):
        self.collapse_log.append({'t': self.t, **metadata})

    def After_Collapsed_Initialized_Loop(self):
        for i in range(5):
            acts = self.Generate_Actions()
            for a in acts:
                a['move'] *= 0.1
            self.Apply_Action(acts)
            self.update_density_and_history()

    def update_density_and_history(self):
        pos_list = [n.pos for n in self.env.nodes]
        speeds = [float(np.linalg.norm(n.velocity())) for n in self.env.nodes]
        dens = self.density.evaluate_density(pos_list, node_speeds=speeds)
        coherence = self.Coherence_Metric()
        collisions = self._count_collisions()
        self.wfc.append_history(self.env.snapshot_nodes(), coherence, collisions, self.env.t)
        self.coherence_history.append(coherence)

    def STEP(self, visualize_every=100):
        actions = self.Generate_Actions()
        self.Apply_Action(actions)
        self.update_density_and_history()
        scores = self.Score_Environment()
        reward = self.Compute_Reward(scores)
        collapsed = False
        if self.wfc.should_collapse():
            meta = self.wfc.collapse_and_reinitialize(self.env)
            self.Record_Collapse_State({'meta': meta, 'scores': scores, 't': self.env.t})
            self.After_Collapsed_Initialized_Loop()
            collapsed = True
        last_hist = self.wfc.history[-1]
        collapse_reason = 'block' if last_hist.get('block', False) else 'soft'
        self.Record_Collapse_State({'meta': meta, 'scores': scores, 't': self.env.t, 'reason': collapse_reason})

        # logging to CSV
        with open(self.logfile, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['t','coherence','collisions','stability','reward','collapse'])
            writer.writerow({'t': self.env.t, 'coherence': scores['coherence'], 'collisions': scores['collisions'], 'stability': scores['stability'], 'reward': reward, 'collapse': int(collapsed)})
        # visualization
        if self.env.t % visualize_every == 0:
            plot_density_and_nodes(self.density.last_density, self.env.nodes, out_path=f"{self.visual_dir}/dens_t{self.env.t:05d}.png", title=f"t={self.env.t} coh={scores['coherence']:.3f}")
        self.t += 1
        return {'scores': scores, 'reward': reward, 'collapse': collapsed}


### FILE: main_rl.py
"""
Main runner: simple random policy baseline + logging + visualization.
Run: python main_rl.py
"""
from FLOWRRA import Flowrra
import argparse
import time

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=2000)
    parser.add_argument('--visual_every', type=int, default=100)
    args = parser.parse_args()

    config = {
        'grid_size': (64,64),
        'angle_steps': 360,
        'num_nodes': 10,
        'bandwidth': 0.04,
        'repulsion_sigma': 0.02,
        'repulsion_strength': 1.2,
        'history_length': 500,
        'tail_length': 20,
        'collapse_threshold': 0.15,
        'tau': 5,
        'collision_allowance': 1,
        'seed': 42,
        'logfile': 'flowrra_log.csv',
        'visual_dir': 'flowrra_visuals'
    }

    model = Flowrra(config)
    start = time.time()
    for i in range(args.steps):
        out = model.STEP(visualize_every=args.visual_every)
        if i % 50 == 0:
            print(f"step {i} coherence {out['scores']['coherence']:.4f} reward {out['reward']:.4f} collapsed={out['collapse']}")
    elapsed = time.time() - start
    print('Done. Time:', elapsed)


### FILE: requirements.txt
numpy
scipy
matplotlib

# End of repository scaffold
# Instructions:
# 1) Create files exactly named as above and paste each block into its file.
# 2) Install requirements (pip install -r requirements.txt).
# 3) Run `python main_rl.py`.
# 4) Outputs: flowrra_log.csv and images in flowrra_visuals/.
