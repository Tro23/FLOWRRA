"""
animated_flowrra.py

Animated deployment pipeline for the FLOWRRA orchestrator.
Combines the trained GNN weights with the interactive Plotly animation framework
(Play/Pause, sliders, moving AGV markers) originally built for the VDA 5050 environment.

FIXED (2026-08-29). Three things made the previous version report 1/25 fleets
arriving while the same weights were scoring 25/25 in training:

  1. THE STALE FROZEN SET.  GNNAgent.save() persists `frozen_nodes` into the
     checkpoint and load() restores it. The checkpoint is written at the END of
     the last training episode, when ~23 of 25 fleets were parked -- so
     agent.load() came back with 23 fleet IDs already marked frozen. In
     choose_actions(), `active_mask = [nid not in self.frozen_nodes ...]`, so
     those 23 fleets were never assigned an action and sat at idle for the whole
     rollout. That is the "others didn't even move" symptom exactly, and the
     giveaway was the log printing "Total frozen nodes: 24" immediately after the
     FIRST fleet claimed a goal. main_runner_warehouse.py calls
     reset_episode_state() every episode, which is why training never hit this.

  2. MODE MISMATCH.  Training ran with shared_pool_mode=True and the Hungarian
     1:1 assignment. This script used load_warehouse_data() (fixed missions) and
     passed neither shared_pool_mode nor goal_pool, so the deployment task was not
     the task the policy was trained on. USE_SHARED_POOL_MODE below mirrors
     main_runner's switch; keep the two in sync.

  3. NO OUTCOME REPORTING.  It rendered an animation without ever saying how many
     fleets actually arrived, so a broken rollout looked like a successful run
     that happened to produce a boring video.
"""

import numpy as np
import plotly.graph_objects as go

from core_warehouse import FLOWRRA
from agent_warehouse import GNNAgent
from main_runner_warehouse import load_warehouse_data, load_warehouse_data_pool_mode
from node_warehouse import precompute_goal_distances
from config_warehouse import CONFIG

# Must match main_runner_warehouse.py's switch -- the policy was trained under
# whichever mode was active there.
USE_SHARED_POOL_MODE = True

NODES_FILE = 'custom/50_5_5_10_5_10_Nodes.csv'
EDGES_FILE = 'custom/50_5_5_10_5_10_Edges.csv'
MISSIONS_FILE = 'custom/50_5_5_10_5_10_StartGoalLocations_STRESS_TEST.csv'
CHECKPOINT = 'checkpoints/flowrra_warehouse_gnn.pth'


def deploy_and_animate():

    # ==========================================================================
    # 1. ENVIRONMENT & MODEL SETUP
    # ==========================================================================
    print("[Animate] Loading warehouse topology and mission data...")

    goal_pool = None
    if USE_SHARED_POOL_MODE:
        G, pos_dict, fleet_missions, goal_pool = load_warehouse_data_pool_mode(
            NODES_FILE, EDGES_FILE, MISSIONS_FILE
        )
        goal_distance_maps = precompute_goal_distances(
            G, [{"goal_node": gid} for gid in goal_pool]
        )
    else:
        G, pos_dict, fleet_missions = load_warehouse_data(
            NODES_FILE, EDGES_FILE, MISSIONS_FILE
        )
        goal_distance_maps = precompute_goal_distances(G, fleet_missions)

    env = FLOWRRA(
        G, pos_dict, fleet_missions, mode="eval",
        goal_distance_maps=goal_distance_maps,
        shared_pool_mode=USE_SHARED_POOL_MODE,
        goal_pool=goal_pool,
    )

    # GNN input dimension
    test_node = env.nodes[0]
    dummy_state = test_node.get_state_vector(env.nodes)
    dummy_affordance = env.density.get_local_affordance(
        test_node.current_pos, env.nodes, set()
    )
    input_dim = len(dummy_state) + len(dummy_affordance)

    agent = GNNAgent(
        node_feature_dim=input_dim,
        edge_feature_dim=0,
        action_size=CONFIG["gnn"]["action_size"],
        hidden_dim=CONFIG["gnn"]["hidden_dim"],
        num_layers=CONFIG["gnn"]["num_layers"],
        n_heads=CONFIG["gnn"]["num_heads"],
        dropout=CONFIG["gnn"]["dropout"],
        stability_coef=CONFIG["gnn"]["stability_coef"],
    )

    print("[Animate] Loading trained FLOWRRA weights...")
    agent.load(CHECKPOINT)

    # THE FIX. load() restores the frozen set that was current when the
    # checkpoint was written -- i.e. every fleet that had already parked by the
    # end of the final training episode. Without this line those fleets are
    # treated as retired before the rollout even starts and never receive an
    # action. See this module's docstring.
    agent.reset_episode_state()
    print(f"[Animate] Frozen set cleared -- {len(env.nodes)} fleets active at t=0.")

    agent.epsilon_gaussian = lambda *args, **kwargs: 0.02  # pure exploitation
    env.gnn = agent

    # ==========================================================================
    # 2. RUN SIMULATION & CAPTURE TIMESTEPS
    # ==========================================================================
    trajectory_frames = []
    finished_at = None

    print("[Animate] Simulating fleet trajectories...")
    for step in range(CONFIG["training"]["max_steps_per_episode"]):
        trajectory_frames.append({
            node.id: {
                'x': float(node.current_pos[0]),
                'y': float(node.current_pos[1]),
                'z': float(node.current_pos[2]),
            }
            for node in env.nodes
        })

        env.step(episode_step=1, total_episodes=1)

        if len(env.frozen_nodes) == len(env.nodes):
            trajectory_frames.append({
                node.id: {
                    'x': float(node.current_pos[0]),
                    'y': float(node.current_pos[1]),
                    'z': float(node.current_pos[2]),
                }
                for node in env.nodes
            })
            finished_at = step
            print(f"[Animate] All fleets crystallized at step {step}!")
            break

    # Outcome report -- without this a broken rollout renders as a perfectly
    # valid-looking animation of fleets doing nothing.
    completed = len(env.frozen_nodes)
    unfinished = [n for n in env.nodes if n.id not in env.frozen_nodes]
    print(f"\n[Animate] ===== DEPLOYMENT RESULT =====")
    print(f"[Animate] Completed : {completed}/{len(env.nodes)}"
          + (f" at step {finished_at}" if finished_at is not None else ""))
    print(f"[Animate] Collisions: {env.loop.get_statistics()['total_collisions_occurred']}")
    print(f"[Animate] Recovery  : {env.recovery.get_statistics()}")
    if unfinished:
        remaining = [
            (n.id, round(float(n.get_graph_distance_to_goal()), 1)) for n in unfinished
        ]
        print(f"[Animate] Unfinished (id, hops left): {remaining}")
    print(f"[Animate] GradAgree : {env.get_gradient_agreement():.3f}")
    print(f"[Animate] ==============================\n")

    # ==========================================================================
    # 3. STATIC BACKGROUND TRACES
    # ==========================================================================
    print("[Animate] Rendering 3D animation...")

    edge_x, edge_y, edge_z = [], [], []
    for u, v in G.edges():
        if u in pos_dict and v in pos_dict:
            edge_x.extend([pos_dict[u]['X'], pos_dict[v]['X'], None])
            edge_y.extend([pos_dict[u]['Y'], pos_dict[v]['Y'], None])
            edge_z.extend([pos_dict[u]['Z'], pos_dict[v]['Z'], None])

    edge_trace = go.Scatter3d(
        x=edge_x, y=edge_y, z=edge_z,
        line=dict(width=1, color='rgba(158, 250, 150, 0.4)'),
        hoverinfo='none', mode='lines', name='Warehouse Edges'
    )

    node_trace = go.Scatter3d(
        x=[p['X'] for p in pos_dict.values()],
        y=[p['Y'] for p in pos_dict.values()],
        z=[p['Z'] for p in pos_dict.values()],
        mode='markers', hoverinfo='none',
        marker=dict(size=1.5, color='rgba(200, 120, 207, 0.5)'),
        name='Grid Nodes'
    )

    agent_start_trace = go.Scatter3d(
        x=[m['start_pos'][0] for m in fleet_missions],
        y=[m['start_pos'][1] for m in fleet_missions],
        z=[m['start_pos'][2] for m in fleet_missions],
        mode='markers', name='Starts',
        marker=dict(size=6, color='lime', symbol='diamond',
                    line=dict(width=1, color='black'))
    )

    # In pool mode the missions carry no goal_pos (goals aren't pre-assigned in
    # the CSV), so the goal markers come from the pool itself.
    if USE_SHARED_POOL_MODE:
        goal_positions = list(goal_pool.values())
    else:
        goal_positions = [m['goal_pos'] for m in fleet_missions]

    agent_goal_trace = go.Scatter3d(
        x=[g[0] for g in goal_positions],
        y=[g[1] for g in goal_positions],
        z=[g[2] for g in goal_positions],
        mode='markers', name='Goals',
        marker=dict(size=6, color='crimson', symbol='circle',
                    line=dict(width=1, color='black'))
    )

    # ==========================================================================
    # 4. ANIMATED AGV TRACES
    # ==========================================================================
    agent_ids = [node.id for node in env.nodes]
    init_text = [f"Fleet {aid}" for aid in agent_ids]

    def frame_xyz(tick):
        f = trajectory_frames[tick]
        return ([f[a]['x'] for a in agent_ids],
                [f[a]['y'] for a in agent_ids],
                [f[a]['z'] for a in agent_ids])

    x0, y0, z0 = frame_xyz(0)
    marker_style = dict(
        size=10, color=list(range(len(agent_ids))), colorscale='Rainbow',
        symbol='circle', line=dict(width=1, color='white')
    )

    agent_trace = go.Scatter3d(
        x=x0, y=y0, z=z0,
        mode='markers+text', name='FLOWRRA Fleets',
        text=init_text, textposition="top center", hoverinfo='text',
        marker=marker_style
    )

    frame_indices = list(range(len(trajectory_frames)))
    frames = []
    for tick in frame_indices:
        fx, fy, fz = frame_xyz(tick)
        frames.append(go.Frame(
            data=[go.Scatter3d(x=fx, y=fy, z=fz, mode='markers+text',
                               text=init_text, marker=marker_style)],
            traces=[4],  # index of agent_trace in the figure's data list
            name=str(tick)
        ))

    # ==========================================================================
    # 5. LAYOUT
    # ==========================================================================
    fig = go.Figure(
        data=[edge_trace, node_trace, agent_start_trace, agent_goal_trace, agent_trace],
        frames=frames
    )

    fig.update_layout(
        title=f'FLOWRRA Deployment -- {completed}/{len(env.nodes)} fleets delivered',
        template="plotly_dark",
        showlegend=True,
        scene=dict(
            xaxis=dict(showbackground=True, showgrid=True, zeroline=True, visible=True),
            yaxis=dict(showbackground=True, showgrid=True, zeroline=True, visible=True),
            zaxis=dict(showbackground=True, showgrid=True, zeroline=True, visible=True)
        ),
        updatemenus=[dict(
            type="buttons",
            buttons=[
                dict(label="Play", method="animate",
                     args=[None, dict(frame=dict(duration=100, redraw=True),
                                      transition=dict(duration=0),
                                      fromcurrent=True, mode='immediate')]),
                dict(label="Pause", method="animate",
                     args=[[None], dict(frame=dict(duration=0, redraw=False),
                                        mode='immediate', transition=dict(duration=0))])
            ],
            direction="left", pad={"r": 10, "t": 87}, showactive=True,
            x=0.1, xanchor="right", y=0, yanchor="top",
            bgcolor="#333", font=dict(color="white")
        )],
        sliders=[dict(
            steps=[dict(
                method='animate',
                args=[[str(k)], dict(mode='immediate',
                                     frame=dict(duration=0, redraw=True),
                                     transition=dict(duration=0))],
                label=str(k)
            ) for k in frame_indices],
            transition=dict(duration=0), x=0.1, len=0.9,
            xanchor="left", y=0, yanchor="top", font=dict(color="white")
        )]
    )

    fig.write_html("flowrra_animated_deployment.html")
    print("[Animate] Interactive animation saved to: flowrra_animated_deployment.html")


if __name__ == '__main__':
    deploy_and_animate()