# FLOWRRA_RL.py
import logging
import csv
import os
import numpy as np
from typing import List, Dict, Any, Tuple

from NodePosition_RL import Node_Position
from EnvironmentA_RL import Environment_A
from EnvironmentB_RL import EnvironmentB
from DensityFunctionEstimator_RL import Density_Function_Estimator
from WaveFunctionCollapse_RL import Wave_Function_Collapse
from RLAgent import SharedRLAgent  # shared single agent

# Setup logger for this module
logger = logging.getLogger("FLOWRRA_RL")
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(h)
logger.setLevel(logging.INFO)


class Flowrra_RL:
    """
    Main orchestrator integrating EnvironmentA, EnvironmentB, Density Estimator,
    Wave Function Collapse (WFC), and a shared RL agent interface.
    """
    def __init__(self, config: Dict[str, Any]):
        self.config = config

        # Environments
        self.env = Environment_A(
            num_nodes=config.get('num_nodes', 10),
            angle_steps=config.get('angle_steps', 24),
            seed=config.get('seed', None)
        )
        self.env_b = EnvironmentB(
            grid_size=config.get('env_b_grid_size', 60),
            num_fixed=config.get('env_b_num_fixed', 10),
            num_moving=config.get('env_b_num_moving', 4),
            seed=config.get('seed', None)
        )

        # Density estimator & WFC
        grid_shape = config.get('grid_size', (60, 60))
        if isinstance(grid_shape, int):
            grid_shape = (grid_shape, grid_shape)
        self.density_estimator = Density_Function_Estimator(
            grid_shape=grid_shape,
            eta=config.get('eta', 0.01),
            gamma_f=config.get('gamma_f', 0.4),
            k_f=config.get('k_f', 4),
            sigma_f=config.get('sigma_f', 2.0),
            decay_lambda=config.get('decay_lambda', 0.003),
            blur_delta=config.get('blur_delta', 0.1)
        )
        self.wfc = Wave_Function_Collapse(
            history_length=config.get('wfc_history_length', 200),
            tail_length=config.get('wfc_tail_length', 15),
            collapse_threshold=config.get('wfc_collapse_threshold', 0.25),
            tau=config.get('wfc_tau', 5)
        )

        # Shared RL Agent will be injected / created outside (see main_runner)
        self.agent = None

        # Visual/logging
        self.visual_dir = config.get('visual_dir', 'flowrra_rl_visuals')
        os.makedirs(self.visual_dir, exist_ok=True)
        self.logfile = config.get('logfile', 'flowrra_rl_log.csv')
        self._setup_log_file()

        # collapse params
        self.repulsion_collapse_threshold = config.get('repulsion_collapse_threshold', 0.4)
        self.t = 0

    def attach_agent(self, agent: SharedRLAgent):
        """Attach the shared agent after it's constructed (so agent can be created with dynamic state size)."""
        self.agent = agent

    def _setup_log_file(self):
        with open(self.logfile, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['timestep', 'episode', 'coherence', 'reward', 'epsilon', 'collapse_event'])

    def _log_step(self, episode: int, coherence: float, reward: float, epsilon: float, collapse_event: bool):
        with open(self.logfile, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([self.env.t, episode, coherence, reward, epsilon, int(collapse_event)])

    def get_state(self) -> np.ndarray:
        """
        Return a global state vector:
        [positions_flat (num_nodes*2), coherence (1), repulsion_potentials (num_nodes)]
        This is concise and deterministic for the shared agent.
        """
        # positions
        positions = np.array([n.pos[:2] for n in self.env.nodes]).flatten()  # (num_nodes*2,)

        # coherence
        coherence = self.wfc.compute_coherence(self.env.nodes)  # scalar

        # potentials at node positions
        node_positions = np.array([n.pos[:2] for n in self.env.nodes])
        potentials = self.density_estimator.get_potential_at_positions(node_positions)  # (num_nodes,)

        state = np.concatenate([positions, np.array([coherence]), potentials])
        return state.astype(np.float32)

    def step(self, actions: List[int]) -> Tuple[Dict[int, float], bool, Dict[str, Any]]:
        """
        Perform one step of the simulated world using the provided per-node actions
        (actions: length == num_nodes, ints from 0..action_size-1).
        Returns: rewards dict keyed by node index, done flag, info dict.
        """
        if len(actions) != len(self.env.nodes):
            raise ValueError("Actions length must equal number of nodes")

        # 1) Apply node-level actions -> convert to env-specific action dicts
        flowrra_actions = []
        for i, a in enumerate(actions):
            target_angle = self.env.nodes[i].angle_idx
            # Mapping actions: 0 -> turn left; 1 -> turn right; 2 -> noop; 3 -> noop (extend as needed)
            if a == 0:
                target_angle -= 1
            elif a == 1:
                target_angle += 1
            # keep angle_idx in valid range using env's angle_steps if available
            if hasattr(self.env, 'angle_steps'):
                target_angle = int(target_angle) % getattr(self.env, 'angle_steps', 24)
            flowrra_actions.append({'target_angle_idx': int(target_angle)})

        # 2) Step environments (EnvA then EnvB) and density dynamics
        self.env.step(flowrra_actions)   # advances node positions based on angle_idx or similar
        self.env_b.step()
        self.density_estimator.step_dynamics()

        # 3) Build rewards for each node
        rewards = {}
        done = False
        total_reward = 0.0

        # Build new_positions array for potential checks
        node_positions = np.array([n.pos[:2] for n in self.env.nodes])
        repulsion_vals = self.density_estimator.get_potential_at_positions(node_positions)

        # Compute pairwise distances robustly (slice to 2D to avoid shape mismatch)
        for i, node in enumerate(self.env.nodes):
            # assume env.update already updated node.pos; use node.pos[:2]
            new_pos = node.pos[:2]
            # survival shaping
            reward = 0.01

            # average distance to others
            other_dists = []
            for n in self.env.nodes:
                if n.id == node.id:
                    continue
                # both are 2D vectors
                other_dists.append(np.linalg.norm(new_pos - np.asarray(n.pos[:2])))
            if other_dists:
                avg_dist = float(np.mean(other_dists))
                # reward shaping: prefer moderate distance (0.2 - 0.6); penalties outside
                if avg_dist < 0.2:
                    reward -= 0.5
                elif avg_dist > 0.6:
                    reward -= 0.2
                else:
                    reward += 0.2

            # repulsion penalty
            rep_val = float(repulsion_vals[i]) if repulsion_vals.size > i else 0.0
            reward += -rep_val

            # collision check (grid-based)
            node_pos_grid = np.floor(new_pos * self.env_b.grid_size).astype(int)
            # clamp indices
            gx = int(np.clip(node_pos_grid[0], 0, self.env_b.grid_size - 1))
            gy = int(np.clip(node_pos_grid[1], 0, self.env_b.grid_size - 1))
            if (gx, gy) in self.env_b.all_blocks:
                # collision penalty
                reward -= 5.0
                done = True
                logger.info(f"[FLOWRRA_RL] Collision detected node={node.id} grid=({gx},{gy}) at t={self.env.t}")

            rewards[i] = float(reward)
            total_reward += reward

        # 4) Compute coherence and record step in WFC
        coherence = self.wfc.compute_coherence(self.env.nodes)
        # record nodes snapshot with minimal info to history
        snapshot = [{'pos': n.pos.copy(), 'angle_idx': n.angle_idx} for n in self.env.nodes]
        self.wfc.record_step(snapshot, coherence, self.env.t)

        # 5) Collapse checks (both repulsion-based and coherence-based)
        collapse_event = False
        info: Dict[str, Any] = {}

        # repulsion-based
        if repulsion_vals.size and np.any(repulsion_vals > self.repulsion_collapse_threshold):
            collapse_event = True
            info['collapse_reason'] = 'repulsion'
        # coherence-based (uses WFC's tau/history)
        elif self.wfc.check_for_collapse():
            collapse_event = True
            info['collapse_reason'] = 'coherence'

        if collapse_event:
            meta = self.wfc.collapse_and_reinitialize(self.env)
            info.update(meta)
            logger.warning(f"[FLOWRRA_RL] Collapse triggered at t={self.env.t} reason={info.get('collapse_reason')} -> {meta.get('reinit_from')}")

        # 6) increment time
        self.env.t += 1

        # 7) Return
        info['coherence'] = float(coherence)
        info['total_reward'] = float(total_reward)
        info['repulsion_max'] = float(np.max(repulsion_vals) if repulsion_vals.size else 0.0)
        info['collapse_event'] = int(collapse_event)

        return rewards, done, info

    def train(self, total_steps: int, episode_steps: int, visualize_every_n_steps: int, agent: SharedRLAgent):
        """
        Training loop wrapper kept for backward compatibility. Expects a shared agent to be attached.
        """
        if agent is None:
            raise ValueError("A shared RL agent must be provided to Flowrra_RL.train()")
        self.attach_agent(agent)

        num_episodes = total_steps // episode_steps
        bell_curve_epsilon = lambda t, total_t: 0.1 + 0.9 * np.exp(-0.5 * ((t - total_t / 2) / (total_t / 6))**2)

        global_step = 0
        for episode in range(num_episodes):
            logger.info(f"--- Starting Episode {episode+1}/{num_episodes} ---")
            self.env.reset()
            self.env_b.reset()
            self.density_estimator.reset()
            self.wfc.reset()
            episode_reward = 0.0

            for step in range(episode_steps):
                epsilon = bell_curve_epsilon(global_step, total_steps)
                state = self.get_state()
                # choose actions for all nodes from shared agent
                actions = self.agent.choose_actions(state, epsilon)  # np array (num_nodes,)

                # perform step
                rewards_dict, done, info = self.step(list(actions))

                # prepare next_state and push to shared buffer
                next_state = self.get_state()
                rewards_arr = np.array([rewards_dict[i] for i in range(len(self.env.nodes))], dtype=np.float32)
                self.agent.push_experience(state, actions, rewards_arr, next_state, done)

                # train shared agent
                self.agent.train_step(self.config.get('batch_size', 32))

                # logging & visualize periodically
                coherence = info.get('coherence', 0.0)
                self._log_step(episode, coherence, info.get('total_reward', 0.0), epsilon, bool(info.get('collapse_event', 0)))
                if global_step % visualize_every_n_steps == 0:
                    try:
                        from utils_rl import plot_system_state_rl
                        plot_system_state_rl(
                            density_field=self.density_estimator.repulsion_field,
                            nodes=self.env.nodes,
                            env_b=self.env_b,
                            out_path=os.path.join(self.visual_dir, f"t_{global_step:05d}.png"),
                            title=f"FLOWRRA RL: Step {global_step} | Eps: {epsilon:.2f}"
                        )
                    except Exception as e:
                        logger.debug(f"Visualization failed at step {global_step}: {e}")

                # If collapse occurred, bias/prune buffer (simple retrocausal effect)
                if info.get('collapse_event'):
                    # prune transitions with low mean reward (tune threshold)
                    try:
                        self.agent.retrocausal_prune(0.0)
                    except Exception:
                        # fallback to available prune method
                        if hasattr(self.agent, 'replay_buffer') and hasattr(self.agent.replay_buffer, 'prune_low_reward'):
                            self.agent.replay_buffer.prune_low_reward(0.0)

                global_step += 1
                episode_reward += info.get('total_reward', 0.0)

                if done:
                    logger.info(f"Episode ended early at step {step} due to done signal.")
                    break

            # periodic target network update
            if hasattr(self.agent, 'update_target_network') and (episode % max(1, self.config.get('target_update_freq', 100)) == 0):
                self.agent.update_target_network()

            logger.info(f"Episode {episode+1} complete | Episode reward: {episode_reward:.2f} | Replay size: {len(self.agent.replay_buffer)}")

        # Save agent model if supported
        if hasattr(self.agent, 'save'):
            self.agent.save(self.config.get('model_save_path', 'flowrra_shared_agent.pth'))

    def deploy(self, total_steps: int, visualize_every_n_steps: int):
        """
        Greedy deployment using the attached shared agent (epsilon=0).
        """
        if self.agent is None:
            raise ValueError("Agent not attached. Call attach_agent() before deploy().")

        logger.info("--- Starting deployment ---")
        self.env.reset()
        self.env_b.reset()
        self.density_estimator.reset()
        self.wfc.reset()

        epsilon = 0.0
        for step in range(total_steps):
            state = self.get_state()
            actions = self.agent.choose_actions(state, epsilon)
            rewards, done, info = self.step(list(actions))

            if step % visualize_every_n_steps == 0:
                try:
                    from utils_rl import plot_system_state_rl
                    plot_system_state_rl(
                        density_field=self.density_estimator.repulsion_field,
                        nodes=self.env.nodes,
                        env_b=self.env_b,
                        out_path=os.path.join(self.visual_dir, f"deploy_t_{step:05d}.png"),
                        title=f"FLOWRRA RL: Deployment Step {step}"
                    )
                except Exception as e:
                    logger.debug(f"Deployment visualization failed at step {step}: {e}")

            if done:
                logger.info(f"Deployment ended early at step {step}.")
                break

        logger.info("--- Deployment finished ---")
