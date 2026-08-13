import asyncio
import datetime
import json
import time
import random
import paho.mqtt.client as mqtt
from typing import Dict, List, Optional, Any, Tuple

import config
import mqtt_utils
import utils
from protocol.vda5050_common import AgvPosition
from protocol.vda_2_0_0.vda5050_2_0_0_connection import Connection, ConnectionState
from protocol.vda_2_0_0.vda5050_2_0_0_state import State, NodeState, EdgeState, ActionState, BatteryState, SafetyState, OperatingMode, ActionStatus, EStop
from protocol.vda_2_0_0.vda5050_2_0_0_visualization import Visualization
from protocol.vda_2_0_0.vda5050_2_0_0_action import Action, ActionParameter, ActionParameterValue, BlockingType
from protocol.vda_2_0_0.vda5050_2_0_0_order import Order
from protocol.vda_2_0_0.vda5050_2_0_0_instant_actions import InstantActions

class VehicleSimulator:
    """Simulates a VDA5050 vehicle"""
    def __init__(self, config_data: config.Config):
        self.config = config_data
        
        # Generate base MQTT topic
        base_topic = mqtt_utils.generate_vda_mqtt_base_topic(
            self.config.mqtt_broker.vda_interface,
            self.config.vehicle.vda_version,
            self.config.vehicle.manufacturer,
            self.config.vehicle.serial_number
        )
        
        # Connection
        self.connection_topic = f"{base_topic}/connection"
        self.connection = Connection(
            header_id=0,
            timestamp=utils.get_timestamp(),
            version=self.config.vehicle.vda_full_version,
            manufacturer=self.config.vehicle.manufacturer,
            serial_number=self.config.vehicle.serial_number,
            connection_state=ConnectionState.CONNECTION_BROKEN
        )
        
        # State
        self.state_topic = f"{base_topic}/state"
        random_x = random.random() * 5.0 - 2.5
        random_y = random.random() * 5.0 - 2.5
        random_z = random.random() * 5.0  # 3D EXTENSION: Random starting height
        
        agv_position = AgvPosition(
            x=random_x,
            y=random_y,
            z=random_z, # 3D EXTENSION
            position_initialized=True,
            theta=0.0,
            map_id=self.config.settings.map_id,
            deviation_range=None,
            map_description=None,
            localization_score=None
        )
        
        self.state = State(
            header_id=0,
            timestamp=utils.get_timestamp(),
            version=self.config.vehicle.vda_full_version,
            manufacturer=self.config.vehicle.manufacturer,
            serial_number=self.config.vehicle.serial_number,
            driving=False,
            distance_since_last_node=None,
            operating_mode=OperatingMode.AUTOMATIC,
            node_states=[],
            edge_states=[],
            last_node_id="",
            order_id="",
            order_update_id=0,
            last_node_sequence_id=0,
            action_states=[],
            information=[],
            loads=[],
            errors=[],
            battery_state=BatteryState(
                battery_charge=0.0,
                battery_voltage=None,
                battery_health=None,
                charging=False,
                reach=None
            ),
            safety_state=SafetyState(
                e_stop=EStop.NONE,
                field_violation=False
            ),
            paused=None,
            new_base_request=None,
            agv_position=agv_position,
            velocity=None,
            zone_set_id=None
        )
        
        # Visualization
        self.visualization_topic = f"{base_topic}/visualization"
        self.visualization = Visualization(
            header_id=0,
            timestamp=utils.get_timestamp(),
            version=self.config.vehicle.vda_full_version,
            manufacturer=self.config.vehicle.manufacturer,
            serial_number=self.config.vehicle.serial_number,
            agv_position=agv_position,
            velocity=None
        )
        
        # Order and Instant Actions
        self.order = None
        self.instant_actions = None
        self.action_start_time = None

    def run_action(self, action: Action) -> None:
        """Execute an action"""
        action_state_index = None
        for i, a_state in enumerate(self.state.action_states):
            if a_state.action_id == action.action_id:
                if a_state.action_status == ActionStatus.WAITING:
                     action_state_index = i
                else:
                     return False
                break
            
        if action_state_index is not None:
            action_state = self.state.action_states[action_state_index]
            action_state.action_status = ActionStatus.RUNNING
            print(f"SIM ({self.config.vehicle.serial_number}): Running action ID {action.action_id}, Type: {action.action_type}")
            
            if action.action_type == "initPosition":
                x_param = next((p for p in action.action_parameters if p.key == "x"), None)
                y_param = next((p for p in action.action_parameters if p.key == "y"), None)
                z_param = next((p for p in action.action_parameters if p.key == "z"), None) # 3D EXTENSION
                theta_param = next((p for p in action.action_parameters if p.key == "theta"), None)
                map_id_param = next((p for p in action.action_parameters if p.key == "mapId"), None)
                
                def extract_val(param, default):
                    if not param: return default
                    val = param.value
                    # If it's a wrapper object, extract its inner value, otherwise return it directly
                    return val.value if hasattr(val, 'value') else val
                
                x_float = float(extract_val(x_param, 0.0))
                y_float = float(extract_val(y_param, 0.0))
                z_float = float(extract_val(z_param, 0.0)) # 3D EXTENSION
                theta_float = float(extract_val(theta_param, 0.0))
                map_id_string = str(extract_val(map_id_param, ""))
                
                # Update position
                self.state.agv_position = AgvPosition(
                    x=x_float,
                    y=y_float,
                    z=z_float, # 3D EXTENSION
                    position_initialized=True,
                    theta=theta_float,
                    map_id=map_id_string,
                    deviation_range=None,
                    map_description=None,
                    localization_score=None
                )
                self.visualization.agv_position = self.state.agv_position
                
                action_state.action_status = ActionStatus.FINISHED
                print(f"SIM ({self.config.vehicle.serial_number}): Finished action ID {action.action_id}, Type: {action.action_type}")
                return True
            
            elif action.action_type == "dropOff":
                print(f"SIM ({self.config.vehicle.serial_number}): Starting dropOff action ID {action.action_id}")
                self.action_start_time = datetime.datetime.utcnow()
                return True

            else:
                print(f"SIM ({self.config.vehicle.serial_number}): Unknown action type '{action.action_type}'. Marking as finished.")
                action_state.action_status = ActionStatus.FINISHED
                return True
            
        return False

    
    async def publish_connection(self, mqtt_client: mqtt.Client) -> None:
        """Publish connection state"""
        json_connection_broken = json.dumps(self.connection.to_dict())
        await mqtt_utils.mqtt_publish(mqtt_client, self.connection_topic, json_connection_broken)
        await asyncio.sleep(1)
        
        self.connection.header_id += 1
        self.connection.timestamp = utils.get_timestamp()
        self.connection.connection_state = ConnectionState.ONLINE
        json_connection_online = json.dumps(self.connection.to_dict())
        await mqtt_utils.mqtt_publish(mqtt_client, self.connection_topic, json_connection_online)
    
    async def publish_visualization(self, mqtt_client: mqtt.Client) -> None:
        """Publish visualization data"""
        self.visualization.header_id += 1
        self.visualization.timestamp = utils.get_timestamp()
        json_visualization = json.dumps(self.visualization.to_dict())
        await mqtt_utils.mqtt_publish(mqtt_client, self.visualization_topic, json_visualization)
    
    async def publish_state(self, mqtt_client: mqtt.Client) -> None:
        """Publish state data"""
        self.state.header_id += 1
        self.state.timestamp = utils.get_timestamp()
        json_state = json.dumps(self.state.to_dict())
        await mqtt_utils.mqtt_publish(mqtt_client, self.state_topic, json_state)
    
    def instant_actions_accept_procedure(self, instant_action_request: InstantActions) -> None:
        """Process incoming instant actions"""
        self.instant_actions = instant_action_request
        for instant_action in self.instant_actions.instant_actions:
            action_state = ActionState(
                action_id=instant_action.action_id,
                action_status=ActionStatus.WAITING,
                action_type=instant_action.action_type,
                action_description=None,
                result_description=None
            )
            self.state.action_states.append(action_state)
    
    def order_accept_procedure(self, order_request: Order) -> None:
        """Process incoming order request"""
        if order_request.order_id != self.state.order_id:
            if self.state.order_id == "":
                self.order_accept(order_request)
                return
            
            if len(self.state.node_states) == 0 and len(self.state.edge_states) == 0:
                self.state.action_states = []
                self.order_accept(order_request)
                return
            else:
                self.order_reject("There is order_state or edge_state in state")
                return
        else:
            if order_request.order_update_id > self.state.order_update_id:
                if len(self.state.node_states) > 0 and len(self.state.edge_states) == 0:
                    self.state.action_states = []
                    self.order_accept(order_request)
                    return
                else:
                    self.order_reject("There is order_state or edge_state in state1")
                    return
            else:
                self.order_reject("Order update id is lower")
                return
    
    def order_accept(self, order_request: Order) -> None:
        """Accept an order"""
        self.order = order_request

        first_node = None
        if self.order.nodes:
             sorted_nodes = sorted(self.order.nodes, key=lambda n: n.sequence_id)
             first_node = sorted_nodes[0]

        if first_node:
             self.state.last_node_id = first_node.node_id
             self.state.last_node_sequence_id = first_node.sequence_id
             print(f"SIMULATOR ({self.config.vehicle.serial_number}): Initializing state to first node: ID={self.state.last_node_id}, Seq={self.state.last_node_sequence_id}")
        else:
             self.state.last_node_id = ""
             self.state.last_node_sequence_id = 0
             print(f"SIMULATOR ({self.config.vehicle.serial_number}): Warning - Accepted order has no nodes.")

        self.state.order_id = self.order.order_id
        self.state.order_update_id = self.order.order_update_id

        self.state.action_states = []
        self.state.node_states = []
        self.state.edge_states = []
        
        for node in self.order.nodes:
            node_state = NodeState(
                node_id=node.node_id,
                sequence_id=node.sequence_id,
                released=node.released,
                node_description=node.node_description,
                node_position=node.node_position
            )
            self.state.node_states.append(node_state)
            
            for action in node.actions:
                action_state = ActionState(
                    action_id=action.action_id,
                    action_type=action.action_type,
                    action_description=action.action_description,
                    action_status=ActionStatus.WAITING,
                    result_description=None
                )
                self.state.action_states.append(action_state)
        
        for edge in self.order.edges:
            edge_state = EdgeState(
                edge_id=edge.edge_id,
                sequence_id=edge.sequence_id,
                released=edge.released,
                start_node_id=edge.start_node_id,
                end_node_id=edge.end_node_id,
                edge_description=edge.edge_description,
                trajectory=None
            )
            self.state.edge_states.append(edge_state)
            
            for action in edge.actions:
                action_state = ActionState(
                    action_id=action.action_id,
                    action_type=action.action_type,
                    action_description=action.action_description,
                    action_status=ActionStatus.WAITING,
                    result_description=None
                )
                self.state.action_states.append(action_state)
    
    def order_reject(self, reason: str) -> None:
        """Reject an order"""
        print(f"Order reject: {reason}")
    
    def state_iterate(self) -> None:
        """Update state based on simulation logic"""
        is_action_running = False
        running_action_id = None
        
        if self.action_start_time is not None:
            current_time = datetime.datetime.utcnow()
            action_end_time = self.action_start_time + datetime.timedelta(seconds=self.config.settings.action_time)

            running_action_state = None
            for a_state in self.state.action_states:
                 if a_state.action_status == ActionStatus.RUNNING:
                      running_action_state = a_state
                      break

            if running_action_state:
                 if current_time < action_end_time:
                      is_action_running = True
                      running_action_id = running_action_state.action_id
                 else:
                      print(f"SIM ({self.config.vehicle.serial_number}): Finished timed action ID {running_action_state.action_id}, Type: {running_action_state.action_type}")
                      running_action_state.action_status = ActionStatus.FINISHED
                      self.action_start_time = None

        if self.instant_actions is not None:
            current_instant_actions = self.instant_actions.instant_actions
            self.instant_actions = None
            for action in current_instant_actions:
                action_started = self.run_action(action)
                if action_started and self.action_start_time is not None:
                     is_action_running = True
                     break

        if self.order is None:
            return

        can_move = True
        for a_state in self.state.action_states:
             if a_state.action_status == ActionStatus.RUNNING:
                  can_move = False
                  break

        if not can_move:
            self.state.driving = False
            return

        actions_at_current_node = []
        for node in self.order.nodes:
             if node.sequence_id == self.state.last_node_sequence_id:
                  actions_at_current_node = node.actions
                  break

        action_started_this_tick = False
        if actions_at_current_node:
            for action in actions_at_current_node:
                 for a_state in self.state.action_states:
                      if a_state.action_id == action.action_id and a_state.action_status == ActionStatus.WAITING:
                           if self.run_action(action):
                                action_started_this_tick = True
                                if action.blocking_type == BlockingType.HARD:
                                     return

        if action_started_this_tick:
             return

        if self.state.agv_position is None:
            return

        if len(self.state.node_states) <= 1:
             if len(self.state.node_states) == 1 and self.state.node_states[0].sequence_id == self.state.last_node_sequence_id:
                  return
             elif len(self.state.node_states) == 0:
                  return

        vehicle_position = self.state.agv_position
        last_node_index = None

        for i, node_state in enumerate(self.state.node_states):
            if node_state.sequence_id == self.state.last_node_sequence_id:
                last_node_index = i
                break

        if last_node_index is None or last_node_index >= len(self.state.node_states) - 1:
             if len(self.state.node_states) == 1 and last_node_index == 0:
                  pass
             self.state.driving = False
             return

        next_node_state = self.state.node_states[last_node_index + 1]

        if next_node_state.node_position is None:
            return

        next_node_position = next_node_state.node_position

        # 3D EXTENSION: Calculating physics using Z coordinates
        current_x = vehicle_position.x
        current_y = vehicle_position.y
        current_z = vehicle_position.z
        
        target_x = next_node_position.x
        target_y = next_node_position.y
        target_z = next_node_position.z

        # 3D EXTENSION: Use upgraded distance function
        distance_to_next_node = utils.get_distance(
             current_x, current_y, current_z, target_x, target_y, target_z
        )
        arrival_threshold = self.config.settings.speed * 0.5 + 0.05

        if distance_to_next_node < arrival_threshold:
             print(f"SIM ({self.config.vehicle.serial_number}): Arrived at node {next_node_state.node_id} (Seq: {next_node_state.sequence_id})")

             self.state.driving = False

             # 3D EXTENSION: Snap all 3 coordinates on arrival
             self.state.agv_position.x = target_x 
             self.state.agv_position.y = target_y
             self.state.agv_position.z = target_z

             self.state.last_node_id = next_node_state.node_id
             self.state.last_node_sequence_id = next_node_state.sequence_id
             self.state.driving = False # ADDED: Mark driving as false on arrival

             edge_to_remove_idx = -1
             for idx, edge in enumerate(self.state.edge_states):
                  if edge.end_node_id == next_node_state.node_id:
                       edge_to_remove_idx = idx
                       break
             if edge_to_remove_idx != -1:
                  self.state.edge_states.pop(edge_to_remove_idx)

             self.visualization.agv_position = self.state.agv_position

        else:
            
             self.state.driving = True
             # 3D EXTENSION: Calculate Next Step
             updated_vehicle_position = utils.iterate_position(
                 current_x, current_y, current_z,
                 target_x, target_y, target_z,
                 self.config.settings.speed
             )

             self.state.agv_position.x = updated_vehicle_position[0]
             self.state.agv_position.y = updated_vehicle_position[1]
             self.state.agv_position.z = updated_vehicle_position[2] # 3D EXTENSION
             self.state.agv_position.theta = updated_vehicle_position[3]
             self.state.driving = True # ADDED: Mark driving as true while moving

             self.visualization.agv_position = self.state.agv_position

class MQTTClient:
    """MQTT client for VDA5050 communication"""
    def __init__(self, config_data: config.Config):
        self.config = config_data
        self.client = None
        self.base_topic = mqtt_utils.generate_vda_mqtt_base_topic(
            self.config.mqtt_broker.vda_interface,
            self.config.vehicle.vda_version,
            self.config.vehicle.manufacturer,
            self.config.vehicle.serial_number
        )
        self.message_queue = None
        self.loop = None 
    
    def on_connect(self, client, userdata, flags, rc, properties=None):
        print(f"Connected with result code {rc}")
        topics = [
            f"{self.base_topic}/order",
            f"{self.base_topic}/instantActions"
        ]
        for topic in topics:
            client.subscribe(topic, qos=1)
            print(f"SIM ({self.config.vehicle.serial_number}): Subscribed to {topic}")
    
    def on_message(self, client, userdata, msg):
        topic = msg.topic
        topic_type = utils.get_topic_type(topic)
        payload = msg.payload.decode('utf-8')

        if self.loop is None or self.message_queue is None:
             return

        try:
            if topic_type == "order":
                order_data = json.loads(payload)
                order = Order.from_dict(order_data)
                self.loop.call_soon_threadsafe(self.message_queue.put_nowait, ("order", order))

            elif topic_type == "instantActions":
                instant_actions_data = json.loads(payload)
                instant_actions = InstantActions.from_dict(instant_actions_data)
                self.loop.call_soon_threadsafe(self.message_queue.put_nowait, ("instantActions", instant_actions))

        except Exception as e:
            print(f"SIM ({self.config.vehicle.serial_number}): Error processing message on {topic}: {e}")

    def connect(self):
        opts = mqtt_utils.mqtt_create_opts()
        client_id_suffix = self.config.vehicle.serial_number.replace(" ", "_")
        self.client = mqtt.Client(client_id=f"{opts['client_id']}-{client_id_suffix}", protocol=mqtt.MQTTv5)

        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message
        self.loop = asyncio.get_running_loop()
        self.message_queue = asyncio.Queue()

        self.client.connect(
            host=self.config.mqtt_broker.host,
            port=int(self.config.mqtt_broker.port)
        )

        self.client.loop_start()
        return self.client, self.message_queue

async def subscribe_vda_messages(vehicle_simulator, mqtt_client, message_queue):
    while True:
        try:
            message_type, message_data = await message_queue.get()
            
            if message_type == "order":
                vehicle_simulator.order_accept_procedure(message_data)
            elif message_type == "instantActions":
                vehicle_simulator.instant_actions_accept_procedure(message_data)
            
            message_queue.task_done()
        
        except Exception as e:
            print(f"Error processing message: {e}")
            await asyncio.sleep(1)

async def publish_vda_messages(vehicle_simulator, mqtt_client, state_frequency, visualization_frequency):
    await vehicle_simulator.publish_connection(mqtt_client)
    
    tick_time = 0.05
    counter_state = 0
    counter_visualization = 0
    
    while True:
        vehicle_simulator.state_iterate()
        
        counter_state += 1
        if counter_state * tick_time > 1.0 / state_frequency:
            counter_state = 0
            await vehicle_simulator.publish_state(mqtt_client)
        
        counter_visualization += 1
        if counter_visualization * tick_time > 1.0 / visualization_frequency:
            counter_visualization = 0
            await vehicle_simulator.publish_visualization(mqtt_client)
        
        await asyncio.sleep(tick_time)

async def main():
    import copy

    # Load configuration
    config_data = config.get_config()
    base_serial = config_data.vehicle.serial_number # FIX: Store the original string 's'
    tasks = []
    
    for robot_index in range(config_data.settings.robot_count):
        # FIX: Clone generic config safely so we don't mutate the shared instance
        vehicle_config = copy.deepcopy(config_data)
        
        # Rename robot serial number
        vehicle_config.vehicle.serial_number = f"{base_serial}{robot_index}"
        
        # Create vehicle simulator
        vehicle_simulator = VehicleSimulator(vehicle_config)
        
        mqtt_handler = MQTTClient(vehicle_config)
        mqtt_client, message_queue = mqtt_handler.connect()
        
        subscribe_task = asyncio.create_task(
            subscribe_vda_messages(vehicle_simulator, mqtt_client, message_queue)
        )
        
        publish_task = asyncio.create_task(
            publish_vda_messages(
                vehicle_simulator,
                mqtt_client,
                config_data.settings.state_frequency,
                config_data.settings.visualization_frequency
            )
        )
        
        tasks.extend([subscribe_task, publish_task])
    
    await asyncio.gather(*tasks)

if __name__ == "__main__":
    asyncio.run(main())