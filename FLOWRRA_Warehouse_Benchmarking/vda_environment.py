import time
import uuid
import pandas as pd
import networkx as nx
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timezone
from protocol.vda_2_0_0.vda5050_2_0_0_order import Order, Node as VDANode, Edge as VDAEdge
from protocol.vda_2_0_0.vda5050_2_0_0_action import Action, ActionParameter, ActionParameterValue, BlockingType
from protocol.vda5050_common import NodePosition

class VDAFleetManager:
    """
    Acts as the master control / Commander to manage MQTT connections, 
    dispatch orders, and capture state/telemetry logs.
    """
    def __init__(self, broker_host='localhost', broker_port=1883, vda_interface='uagv', vda_version='v2'):
        self.broker = broker_host
        self.port = broker_port
        self.base_topic_sub = f"{vda_interface}/{vda_version}/+/+"
        
        # Use a dynamic UUID to avoid connection collisions
        import paho.mqtt.client as mqtt
        self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=f"NotebookCommander_{uuid.uuid4().hex[:8]}", protocol=mqtt.MQTTv5)
        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message
        
        self.logs = [] # Data array to store incoming telemetry
        self.latest_state = {} # NEW: Tracks the exact latest state for lightning-fast checks
        
    def on_connect(self, client, userdata, flags, reason_code, properties=None):
        print(f"Connected to MQTT Broker at {self.broker}:{self.port} (Code {reason_code})")
        client.subscribe(f"{self.base_topic_sub}/state", qos=1)
        client.subscribe(f"{self.base_topic_sub}/visualization", qos=1)
        
    def on_message(self, client, userdata, msg):
        topic = msg.topic
        try:
            import json
            payload = json.loads(msg.payload.decode('utf-8'))
            serial_number = payload.get('serialNumber', 'unknown')
            
            # Extract Position & Telemetry
            pos = payload.get("agvPosition", {})
            if pos:
                # FIX: Scrub None values caused by protocol serialization bugs
                raw_order = payload.get('orderId', payload.get('order_id'))
                raw_node = payload.get('lastNodeId', payload.get('last_node_id'))
                
                log_entry = {
                    "Timestamp_Sys": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                    "Timestamp_VDA": payload.get('timestamp'),
                    "Topic": topic.split("/")[-1],
                    "AgentId": serial_number,
                    "X": pos.get("x"),
                    "Y": pos.get("y"),
                    "Z": pos.get("z", 0.0), 
                    "Theta": pos.get("theta"),
                    "Battery": payload.get('batteryState', {}).get('batteryCharge', None),
                    "Driving": payload.get('driving', False),
                    "OrderId": "" if raw_order is None else str(raw_order),
                    "LastNodeId": "" if raw_node is None else str(raw_node)
                }
                
                self.logs.append(log_entry)
                # Keep a separate dictionary updated with only the most recent status
                if log_entry["Topic"] == "state":
                    self.latest_state[serial_number] = log_entry
                    
        except Exception as e:
            print(f"Error decoding message on {topic}: {e}")
            
    def connect(self):
        self.client.connect(self.broker, self.port)
        self.client.loop_start()
        
    def disconnect(self):
        self.client.loop_stop()
        self.client.disconnect()
        print("Disconnected from MQTT Broker.")
        
    def publish_order(self, vda_interface, vda_version, manufacturer, agent_id, order_obj):
        topic = f"{vda_interface}/{vda_version}/{manufacturer}/{agent_id}/order"
        import json
        payload = json.dumps(order_obj.to_dict())
        self.client.publish(topic, payload, qos=1)
        print(f"Published order {order_obj.order_id} to {topic}")
        
    def get_logs_df(self):
        return pd.DataFrame(self.logs)

class VDAEnvironment:
    """
    A wrapper class that manages the VDA 5050 Fleet, handles map data, 
    provides a clean interface for pathfinding algorithms (like FLOWRRA),
    and renders 3D telemetry visualizations.
    """
    def __init__(self, pos_dict, G, agent_data, cfg):
        self.pos_dict = pos_dict
        self.G = G
        self.agent_data = agent_data
        self.cfg = cfg
        
        print("Initializing VDA Environment & MQTT Connection...")
        self.manager = VDAFleetManager(
            broker_host=cfg.mqtt_broker.host,
            broker_port=int(cfg.mqtt_broker.port),
            vda_interface=cfg.mqtt_broker.vda_interface,
            vda_version=cfg.vehicle.vda_version
        )
        self.manager.connect()
        time.sleep(1) # Wait for broker to acknowledge
        
    def create_vda_order_from_path(self, agent_id, path_nodes):
        """Helper function to format VDA 5050 orders with Teleportation Injection"""
        nodes = []
        edges = []
        map_id = self.cfg.settings.map_id
        
        for i, node_id in enumerate(path_nodes):
            node_seq_id = i * 2
            pos = self.pos_dict.get(node_id, {'X': 0, 'Y': 0, 'Z': 0})
            node_pos = NodePosition(x=pos['X'], y=pos['Y'], z=pos['Z'], theta=0.0, map_id=map_id)
            
            actions = []
            if i == 0:
                actions.append(Action(
                    action_type="initPosition",
                    action_id=f"init_{uuid.uuid4().hex[:6]}",
                    blocking_type=BlockingType.HARD,
                    action_parameters=[
                        ActionParameter(key="x", value=ActionParameterValue(pos['X'])),
                        ActionParameter(key="y", value=ActionParameterValue(pos['Y'])),
                        ActionParameter(key="z", value=ActionParameterValue(pos['Z'])),
                        ActionParameter(key="theta", value=ActionParameterValue(0.0)),
                        ActionParameter(key="mapId", value=ActionParameterValue(map_id))
                    ]
                ))

            vda_node = VDANode(
                node_id=str(node_id),
                sequence_id=node_seq_id,
                released=True,
                actions=actions, 
                node_position=node_pos
            )
            nodes.append(vda_node)
            
            if i < len(path_nodes) - 1:
                next_node_id = path_nodes[i+1]
                vda_edge = VDAEdge(
                    edge_id=f"e_{node_id}_{next_node_id}",
                    sequence_id=(i * 2) + 1,
                    released=True,
                    start_node_id=str(node_id),
                    end_node_id=str(next_node_id),
                    actions=[]
                )
                edges.append(vda_edge)
                
        return Order(
            header_id=0,
            timestamp=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            version="2.0.0",
            manufacturer=self.cfg.vehicle.manufacturer,
            serial_number=agent_id,
            order_id=f"order_{agent_id}_{uuid.uuid4().hex[:4]}",
            order_update_id=0,
            nodes=nodes,
            edges=edges
        )

    def dispatch_routes(self, routing_algorithm="networkx"):
        """Calculates paths and dispatches VDA 5050 orders to the fleet."""
        dispatched_agents = {}
        
        for idx, (_, row) in enumerate(self.agent_data.iterrows()):
            agent_id = f"{self.cfg.vehicle.serial_number}{idx}"
            start = str(row['startNodeId']).strip()
            goal = str(row['goalNodeId']).strip()
            
            if start in self.G and goal in self.G:
                # ==========================================
                # ALGORITHM PLUGIN ZONE
                # ==========================================
                if routing_algorithm == "networkx":
                    path = nx.shortest_path(self.G, source=start, target=goal, weight='weight')
                elif routing_algorithm == "flowrra":
                    # path = flowrra.calculate_3d_path(self.G, start, goal) 
                    print("FLOWRRA not yet connected, defaulting to NX.")
                    path = nx.shortest_path(self.G, source=start, target=goal, weight='weight')
                else:
                    path = []
                
                if not path:
                    continue
                    
                vda_order = self.create_vda_order_from_path(agent_id, path)
                
                self.manager.publish_order(
                    vda_interface=self.cfg.mqtt_broker.vda_interface,
                    vda_version=self.cfg.vehicle.vda_version,
                    manufacturer=self.cfg.vehicle.manufacturer,
                    agent_id=agent_id,
                    order_obj=vda_order
                )
                
                # Track both the order ID and the final goal node for strict verification
                dispatched_agents[agent_id] = {
                    'order_id': vda_order.order_id,
                    'goal_node': str(path[-1])
                }
                
        return dispatched_agents

    def wait_for_fleet(self, active_agents, timeout=300):
        """Dynamically waits until all dispatched agents report they have stopped moving at their goals."""
        print(f"Waiting for {len(active_agents)} agents to complete their routes...")
        start_time = time.time()
        last_debug_print = start_time 
        
        # Give the fleet a 2-second grace period to receive orders and start their motors
        time.sleep(2)
        
        while True:
            current_time = time.time()
            if current_time - start_time > timeout:
                print("TIMEOUT: Maximum simulation time reached.")
                break
                
            # Assume all have arrived, then check to see if anyone proves us wrong
            all_arrived = True
            waiting_reasons = [] 
            
            for agent, target_data in active_agents.items():
                latest_status = self.manager.latest_state.get(agent)
                
                if not latest_status:
                    all_arrived = False 
                    waiting_reasons.append(f"{agent}: No telemetry yet")
                    continue
                    
                # BACK TO BASICS 1: Is the motor off?
                if latest_status.get('Driving') == True:
                    all_arrived = False
                    waiting_reasons.append(f"{agent}: Still Driving")
                    continue 
                    
                # BACK TO BASICS 2: Is it physically at the end goal?
                # We bypass the broken protocol strings by checking exact physical coordinates
                goal_node = target_data['goal_node']
                goal_pos = self.pos_dict.get(goal_node)
                
                if goal_pos:
                    dx = latest_status.get('X', 0) - goal_pos['X']
                    dy = latest_status.get('Y', 0) - goal_pos['Y']
                    dz = latest_status.get('Z', 0) - goal_pos.get('Z', 0) # 3D EXTENSION
                    
                    # 3D EXTENSION: Using Manhattan Distance to match utils.py grid logic
                    distance_to_goal = abs(dx) + abs(dy) + abs(dz)
                    
                    # If it stopped but is still far away, it hasn't finished its path
                    if distance_to_goal > 0.5:
                        all_arrived = False
                        waiting_reasons.append(f"{agent}: Stopped but not at goal (Dist: {distance_to_goal:.1f}m)")
                        continue
            
            # If we looped through every agent and NO ONE set all_arrived to False, we are done!
            if all_arrived:
                print(f"\nSUCCESS: All agents have stopped moving at their goals! (Took {current_time - start_time:.1f}s)")
                break
                
            # Print a debug summary every 5 seconds so we aren't flying blind
            if current_time - last_debug_print > 5:
                print(f"Still waiting... Sample reasons: {waiting_reasons[:3]}")
                last_debug_print = current_time
                
            time.sleep(0.5) # Fast loop check without bogging down the CPU 
            
    def close(self):
        """Clean up and export telemetry data."""
        df_logs = self.manager.get_logs_df()
        if not df_logs.empty:
            df_logs.to_csv("vda_flowrra_benchmark_logs.csv", index=False)
        self.manager.disconnect()
        return df_logs

    def render_3d_visualization(self, df_logs, filename="vda_fleet_simulation.html"):
        """
        Takes the raw MQTT telemetry logs and replays the physics over time 
        using an interactive Plotly 3D animation.
        """
        print("Generating Plotly animation from VDA telemetry...")
        if df_logs.empty:
            print("No telemetry data to visualize!")
            return
            
        # 1. Base trace for Edges
        edge_x, edge_y, edge_z = [], [], []
        for u, v in self.G.edges():
            if u in self.pos_dict and v in self.pos_dict:
                edge_x.extend([self.pos_dict[u]['X'], self.pos_dict[v]['X'], None])
                edge_y.extend([self.pos_dict[u]['Y'], self.pos_dict[v]['Y'], None])
                edge_z.extend([self.pos_dict[u]['Z'], self.pos_dict[v]['Z'], None])

        edge_trace = go.Scatter3d(
            x=edge_x, y=edge_y, z=edge_z,
            line=dict(width=1, color='rgba(158, 250, 150, 0.4)'),
            hoverinfo='none', mode='lines', name='Network Edges'
        )

        # 2. Base trace for Nodes
        node_x = [pos['X'] for pos in self.pos_dict.values()]
        node_y = [pos['Y'] for pos in self.pos_dict.values()]
        node_z = [pos['Z'] for pos in self.pos_dict.values()]
        
        node_trace = go.Scatter3d(
            x=node_x, y=node_y, z=node_z,
            mode='markers', hoverinfo='none',
            marker=dict(size=1.5, color='rgba(200, 120, 207, 0.5)'),
            name='Nodes'
        )

        # 3. Base traces for Starts and Goals
        agent_start_x, agent_start_y, agent_start_z, agent_start_text = [], [], [], []
        agent_goal_x, agent_goal_y, agent_goal_z, agent_goal_text = [], [], [], []
        
        for _, row in self.agent_data.iterrows():
            aid = row['agentId']
            s_node = str(row['startNodeId']).strip()
            g_node = str(row['goalNodeId']).strip()
            
            if s_node in self.pos_dict:
                agent_start_x.append(self.pos_dict[s_node]['X'])
                agent_start_y.append(self.pos_dict[s_node]['Y'])
                agent_start_z.append(self.pos_dict[s_node]['Z'])
                agent_start_text.append(f"Agent {aid} Start")
                
            if g_node in self.pos_dict:
                agent_goal_x.append(self.pos_dict[g_node]['X'])
                agent_goal_y.append(self.pos_dict[g_node]['Y'])
                agent_goal_z.append(self.pos_dict[g_node]['Z'])
                agent_goal_text.append(f"Agent {aid} Goal")

        agent_start_trace = go.Scatter3d(
            x=agent_start_x, y=agent_start_y, z=agent_start_z,
            mode='markers', name='Starts', hoverinfo='text', text=agent_start_text,
            marker=dict(size=6, color='lime', symbol='diamond', line=dict(width=1, color='black'))
        )
        agent_goal_trace = go.Scatter3d(
            x=agent_goal_x, y=agent_goal_y, z=agent_goal_z,
            mode='markers', name='Goals', hoverinfo='text', text=agent_goal_text,
            marker=dict(size=6, color='crimson', symbol='circle', line=dict(width=1, color='black'))
        )

        # 4. Transform Telemetry Data for Animation
        df = df_logs[df_logs['Topic'] == 'state'].copy()
        df = df.sort_values('Timestamp_Sys') # Chronological order
        
        trajectories = {}
        agent_ids = df['AgentId'].unique()
        max_frames = 0
        
        # Build raw trajectories
        for aid in agent_ids:
            agent_df = df[df['AgentId'] == aid]
            traj = [{'x': row['X'], 'y': row['Y'], 'z': row['Z']} for _, row in agent_df.iterrows()]
            if traj:
                trajectories[aid] = traj
                max_frames = max(max_frames, len(traj))
                
        if max_frames == 0:
            print("No movement detected in logs.")
            return

        # Pad trajectories so finished robots stay at their goal
        for aid in agent_ids:
            if not trajectories.get(aid):
                trajectories[aid] = [{'x': 0, 'y': 0, 'z': 0}] * max_frames
            last_pos = trajectories[aid][-1]
            while len(trajectories[aid]) < max_frames:
                trajectories[aid].append(last_pos)

        # Initial Agent Trace (Frame 0)
        init_x = [trajectories[aid][0]['x'] for aid in agent_ids]
        init_y = [trajectories[aid][0]['y'] for aid in agent_ids]
        init_z = [trajectories[aid][0]['z'] for aid in agent_ids]
        init_text = [f"Agent {aid}" for aid in agent_ids]

        agent_trace = go.Scatter3d(
            x=init_x, y=init_y, z=init_z,
            mode='markers+text', name='AGVs',
            text=init_text, textposition="top center", hoverinfo='text',
            marker=dict(
                size=6, color=np.arange(len(agent_ids)), colorscale='Viridis', 
                symbol='diamond', line=dict(width=1, color='black')
            )
        )

        # 5. Build Animation Frames
        frames = []
        # Downsample heavily for performance (Plotly struggles with >500 frames)
        step_size = max(1, max_frames // 150) 
        frame_indices = list(range(0, max_frames, step_size))
        if max_frames - 1 not in frame_indices:
            frame_indices.append(max_frames - 1) # Ensure the final destination is shown

        for tick in frame_indices:
            frame_x = [trajectories[aid][tick]['x'] for aid in agent_ids]
            frame_y = [trajectories[aid][tick]['y'] for aid in agent_ids]
            frame_z = [trajectories[aid][tick]['z'] for aid in agent_ids]
            frame_text = [f"Agent {aid}" for aid in agent_ids]
            
            frames.append(go.Frame(
                data=[go.Scatter3d(
                    x=frame_x, y=frame_y, z=frame_z,
                    mode='markers+text', text=frame_text,
                    marker=dict(
                        symbol='diamond', size=10, color=list(range(len(agent_ids))), 
                        colorscale='Rainbow', line=dict(width=1, color='DarkSlateGrey')
                    )
                )],
                traces=[4], # Index 4 corresponds to agent_trace in the data list below
                name=str(tick)
            ))

        # 6. Render Layout
        fig = go.Figure(
            data=[edge_trace, node_trace, agent_start_trace, agent_goal_trace, agent_trace],
            frames=frames
        )

        fig.update_layout(
            title='3D AGV Fleet Simulation (VDA 5050 Telemetry)',
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
                         args=[None, dict(frame=dict(duration=50, redraw=True), transition=dict(duration=0), fromcurrent=True, mode='immediate')]),
                    dict(label="Pause", method="animate",
                         args=[[None], dict(frame=dict(duration=0, redraw=False), mode='immediate', transition=dict(duration=0))])
                ],
                direction="left", pad={"r": 10, "t": 87}, showactive=True, x=0.1, xanchor="right", y=0, yanchor="top"
            )],
            sliders=[dict(
                steps=[dict(
                    method='animate',
                    args=[[str(k)], dict(mode='immediate', frame=dict(duration=0, redraw=True), transition=dict(duration=0))],
                    label=str(k)
                ) for k in frame_indices],
                transition=dict(duration=0), x=0.1, len=0.9, xanchor="left", y=0, yanchor="top"
            )]
        )

        fig.write_html(filename)
        print(f"Interactive animation saved to: {filename}")
        try:
            fig.show() # Attempt to render in notebook
        except:
            pass