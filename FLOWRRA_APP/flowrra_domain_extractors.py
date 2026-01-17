"""
domain_extractors.py

Concrete implementations of FeatureExtractor and RewardCalculator
for different domains.

Domains:
1. Warehouse Robotics
2. Smart Cities (Traffic)
3. Satellite Constellations
"""

from typing import Any, Dict, List

import numpy as np

from holon.node import NodePositionND


# =============================================================================
# WAREHOUSE ROBOTICS
# =============================================================================

class WarehouseFeatureExtractor:
    """
    Feature extractor for warehouse robotics.
    
    Features:
    - Position (2D)
    - Velocity (2D)
    - Battery level
    - Load status (carrying/empty)
    - Distance to current task
    - Local density grid (flattened)
    - Detection counts
    """
    
    def __init__(self):
        self.include_task_features = True
    
    def extract_features(
        self,
        node: NodePositionND,
        local_grid: np.ndarray,
        node_detections: List[Dict],
        constraint_detections: List[Dict],
        **domain_context
    ) -> np.ndarray:
        """Extract warehouse robot features."""
        features = []
        
        # 1. Position (normalized to [0, 1])
        features.extend(node.pos.tolist())
        
        # 2. Velocity
        velocity = node.velocity()
        features.extend(velocity.tolist())
        
        # 3. Battery level (from metadata if available)
        battery = getattr(node, 'battery_level', 100.0) / 100.0
        features.append(battery)
        
        # 4. Load status (0=empty, 1=carrying)
        carrying_load = float(getattr(node, 'carrying_load', False))
        features.append(carrying_load)
        
        # 5. Task-related features
        if self.include_task_features:
            # Distance to task location
            task_pos = getattr(node, 'task_location', node.pos)
            task_dist = np.linalg.norm(node.pos - task_pos)
            features.append(np.clip(task_dist, 0, 1))
            
            # Task urgency
            task_urgency = getattr(node, 'task_urgency', 0.5)
            features.append(task_urgency)
        
        # 6. Local density grid (obstacle avoidance info)
        features.extend(local_grid.flatten().tolist())
        
        # 7. Neighbor density
        num_neighbors = len(node_detections)
        features.append(np.clip(num_neighbors / 10.0, 0, 1))
        
        # 8. Obstacle proximity
        num_obstacles = len(constraint_detections)
        features.append(np.clip(num_obstacles / 5.0, 0, 1))
        
        # 9. Closest obstacle distance
        if constraint_detections:
            min_obstacle_dist = min(det['distance'] for det in constraint_detections)
            features.append(np.clip(min_obstacle_dist, 0, 1))
        else:
            features.append(1.0)  # No obstacles nearby
        
        return np.array(features, dtype=np.float32)


class WarehouseRewardCalculator:
    """
    Reward calculator for warehouse robotics.
    
    Rewards:
    - Task completion: +100
    - Progress toward task: +10 per 0.1 units
    - Collision with shelf: -50
    - Battery depletion: -0.1 per step
    - Idle penalty: -1
    - Formation integrity: +5
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.cfg = config.get('rewards', {})
        
        # Warehouse-specific rewards
        self.r_task_complete = 100.0
        self.r_task_progress = 10.0
        self.r_collision = -50.0
        self.r_battery_drain = -0.1
        self.r_idle = -1.0
        self.r_formation = 5.0
    
    def compute_reward(
        self,
        node: NodePositionND,
        action: int,
        old_pos: np.ndarray,
        new_pos: np.ndarray,
        collided: bool,
        loop_integrity: float,
        **domain_context
    ) -> float:
        """Compute warehouse robot reward."""
        reward = 0.0
        
        # 1. Task progress
        task_pos = getattr(node, 'task_location', node.pos)
        old_dist = np.linalg.norm(old_pos - task_pos)
        new_dist = np.linalg.norm(new_pos - task_pos)
        
        progress = old_dist - new_dist
        reward += self.r_task_progress * progress
        
        # 2. Task completion
        if new_dist < 0.05:  # Reached task location
            carrying = getattr(node, 'carrying_load', False)
            if not carrying:  # Pick up
                reward += self.r_task_complete
                node.carrying_load = True
            else:  # Drop off
                reward += self.r_task_complete * 1.5  # Higher reward for delivery
                node.carrying_load = False
        
        # 3. Collision penalty
        if collided:
            reward += self.r_collision
            # Extra penalty if carrying load
            if getattr(node, 'carrying_load', False):
                reward += self.r_collision * 0.5
        
        # 4. Battery drain
        reward += self.r_battery_drain
        
        # 5. Movement reward (encourage motion)
        move_mag = np.linalg.norm(new_pos - old_pos)
        if move_mag < 0.001:
            reward += self.r_idle
        else:
            reward += move_mag * 2.0
        
        # 6. Formation integrity (stay coordinated)
        reward += self.r_formation * loop_integrity
        
        return reward


# =============================================================================
# SMART CITIES (TRAFFIC MANAGEMENT)
# =============================================================================

class TrafficFeatureExtractor:
    """
    Feature extractor for smart city traffic management.
    
    Features:
    - Position (2D)
    - Velocity (2D)
    - Vehicle type (car/bus/truck)
    - Destination direction
    - Distance to destination
    - Traffic density around vehicle
    - Signal status (if applicable)
    - Local density grid
    """
    
    def __init__(self):
        self.vehicle_types = {'car': 0.33, 'bus': 0.66, 'truck': 1.0}
    
    def extract_features(
        self,
        node: NodePositionND,
        local_grid: np.ndarray,
        node_detections: List[Dict],
        constraint_detections: List[Dict],
        **domain_context
    ) -> np.ndarray:
        """Extract traffic vehicle features."""
        features = []
        
        # 1. Position
        features.extend(node.pos.tolist())
        
        # 2. Velocity
        velocity = node.velocity()
        features.extend(velocity.tolist())
        features.append(np.linalg.norm(velocity))  # Speed magnitude
        
        # 3. Vehicle type (encoded)
        vehicle_type = getattr(node, 'vehicle_type', 'car')
        features.append(self.vehicle_types.get(vehicle_type, 0.33))
        
        # 4. Destination features
        destination = getattr(node, 'destination', node.pos)
        dest_vec = destination - node.pos
        dest_dist = np.linalg.norm(dest_vec)
        
        # Destination direction (normalized)
        if dest_dist > 0.001:
            dest_dir = dest_vec / dest_dist
            features.extend(dest_dir.tolist())
        else:
            features.extend([0.0, 0.0])
        
        # Distance to destination
        features.append(np.clip(dest_dist, 0, 1))
        
        # 5. Traffic density (local congestion)
        local_density = len(node_detections)
        features.append(np.clip(local_density / 8.0, 0, 1))
        
        # Average velocity of nearby vehicles
        if node_detections:
            nearby_speeds = [det.get('speed', 0) for det in node_detections]
            avg_speed = np.mean(nearby_speeds)
            features.append(np.clip(avg_speed, 0, 1))
        else:
            features.append(0.0)
        
        # 6. Road network features (if available)
        current_lane = getattr(node, 'current_lane', 0)
        features.append(current_lane / 4.0)  # Normalize lanes
        
        # 7. Signal status (if at intersection)
        signal_status = getattr(node, 'signal_status', 'green')
        signal_encoding = {'red': 0.0, 'yellow': 0.5, 'green': 1.0}
        features.append(signal_encoding.get(signal_status, 1.0))
        
        # 8. Local grid (obstacles/buildings)
        features.extend(local_grid.flatten().tolist())
        
        # 9. Urgency (emergency vehicle priority)
        urgency = getattr(node, 'urgency', 0.5)
        features.append(urgency)
        
        return np.array(features, dtype=np.float32)


class TrafficRewardCalculator:
    """
    Reward calculator for traffic management.
    
    Rewards:
    - Progress toward destination: +5
    - Reaching destination: +50
    - Maintaining safe distance: +2
    - Collision avoidance: +10
    - Congestion penalty: -5
    - Signal compliance: +3
    - Smooth driving: +1
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.cfg = config.get('rewards', {})
        
        # Traffic-specific rewards
        self.r_progress = 5.0
        self.r_destination = 50.0
        self.r_safe_distance = 2.0
        self.r_collision_avoid = 10.0
        self.r_congestion = -5.0
        self.r_signal_compliance = 3.0
        self.r_smooth = 1.0
    
    def compute_reward(
        self,
        node: NodePositionND,
        action: int,
        old_pos: np.ndarray,
        new_pos: np.ndarray,
        collided: bool,
        loop_integrity: float,
        **domain_context
    ) -> float:
        """Compute traffic vehicle reward."""
        reward = 0.0
        
        # 1. Progress toward destination
        destination = getattr(node, 'destination', node.pos)
        old_dist = np.linalg.norm(old_pos - destination)
        new_dist = np.linalg.norm(new_pos - destination)
        
        progress = old_dist - new_dist
        reward += self.r_progress * progress
        
        # 2. Destination reached
        if new_dist < 0.05:
            reward += self.r_destination
        
        # 3. Collision avoidance
        if collided:
            reward -= 30.0  # Heavy penalty
        else:
            reward += self.r_collision_avoid
        
        # 4. Congestion penalty
        nearby_vehicles = domain_context.get('nearby_vehicles', [])
        if len(nearby_vehicles) > 5:
            reward += self.r_congestion
        
        # 5. Safe distance maintenance
        min_distance = domain_context.get('min_distance_to_vehicle', 1.0)
        if min_distance > 0.08:  # Safe distance
            reward += self.r_safe_distance
        elif min_distance < 0.03:  # Too close
            reward -= self.r_safe_distance
        
        # 6. Signal compliance
        signal_status = getattr(node, 'signal_status', 'green')
        movement = np.linalg.norm(new_pos - old_pos)
        
        if signal_status == 'red' and movement < 0.001:
            reward += self.r_signal_compliance
        elif signal_status == 'red' and movement > 0.01:
            reward -= self.r_signal_compliance * 2
        
        # 7. Smooth driving (penalize erratic motion)
        velocity = node.velocity()
        speed = np.linalg.norm(velocity)
        
        old_speed = domain_context.get('previous_speed', speed)
        speed_change = abs(speed - old_speed)
        
        if speed_change < 0.01:  # Smooth
            reward += self.r_smooth
        else:  # Jerky
            reward -= self.r_smooth
        
        # 8. Emergency vehicle priority
        urgency = getattr(node, 'urgency', 0.5)
        if urgency > 0.8:  # Emergency vehicle
            reward *= 1.5  # Boost all rewards
        
        return reward


# =============================================================================
# SATELLITE CONSTELLATIONS
# =============================================================================

class SatelliteFeatureExtractor:
    """
    Feature extractor for satellite constellation management.
    
    Features:
    - Position (3D orbital coordinates)
    - Velocity (3D)
    - Power level
    - Link quality to ground stations
    - Coverage score
    - Distance to nearest satellite
    - Orbital phase
    - Local density grid
    """
    
    def __init__(self):
        pass
    
    def extract_features(
        self,
        node: NodePositionND,
        local_grid: np.ndarray,
        node_detections: List[Dict],
        constraint_detections: List[Dict],
        **domain_context
    ) -> np.ndarray:
        """Extract satellite features."""
        features = []
        
        # 1. Position (3D orbital)
        features.extend(node.pos.tolist())
        
        # 2. Velocity (orbital velocity)
        velocity = node.velocity()
        features.extend(velocity.tolist())
        
        # 3. Orbital radius
        orbital_radius = np.linalg.norm(node.pos)
        features.append(orbital_radius)
        
        # 4. Power level
        power = getattr(node, 'power_level', 80.0) / 100.0
        features.append(power)
        
        # 5. Link quality to ground stations
        link_quality = getattr(node, 'link_quality', 0.5)
        features.append(link_quality)
        
        # 6. Coverage score
        coverage = getattr(node, 'coverage_score', 0.0)
        features.append(coverage)
        
        # 7. Inter-satellite distances
        if node_detections:
            distances = [det['distance'] for det in node_detections]
            min_dist = min(distances)
            avg_dist = np.mean(distances)
            features.append(np.clip(min_dist, 0, 1))
            features.append(np.clip(avg_dist, 0, 1))
        else:
            features.extend([1.0, 1.0])
        
        # 8. Orbital phase (position in orbit)
        # Simplified: angle in orbital plane
        angle = np.arctan2(node.pos[1], node.pos[0])
        features.append((angle + np.pi) / (2 * np.pi))  # Normalize to [0, 1]
        
        # 9. Debris proximity
        debris_count = len(constraint_detections)
        features.append(np.clip(debris_count / 5.0, 0, 1))
        
        if constraint_detections:
            min_debris_dist = min(det['distance'] for det in constraint_detections)
            features.append(np.clip(min_debris_dist, 0, 1))
        else:
            features.append(1.0)
        
        # 10. Local density grid (for collision avoidance)
        features.extend(local_grid.flatten().tolist())
        
        # 11. Constellation health
        num_active_neighbors = len(node_detections)
        features.append(np.clip(num_active_neighbors / 8.0, 0, 1))
        
        return np.array(features, dtype=np.float32)


class SatelliteRewardCalculator:
    """
    Reward calculator for satellite constellation management.
    
    Rewards:
    - Coverage maintenance: +20
    - Link quality: +10
    - Collision avoidance: +50
    - Power efficiency: +5
    - Debris avoidance: +15
    - Formation integrity: +10
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.cfg = config.get('rewards', {})
        
        # Satellite-specific rewards
        self.r_coverage = 20.0
        self.r_link = 10.0
        self.r_collision_avoid = 50.0
        self.r_power = 5.0
        self.r_debris_avoid = 15.0
        self.r_formation = 10.0
    
    def compute_reward(
        self,
        node: NodePositionND,
        action: int,
        old_pos: np.ndarray,
        new_pos: np.ndarray,
        collided: bool,
        loop_integrity: float,
        **domain_context
    ) -> float:
        """Compute satellite reward."""
        reward = 0.0
        
        # 1. Coverage score improvement
        old_coverage = getattr(node, '_prev_coverage', 0.0)
        new_coverage = getattr(node, 'coverage_score', 0.0)
        node._prev_coverage = new_coverage
        
        coverage_delta = new_coverage - old_coverage
        reward += self.r_coverage * coverage_delta
        
        # 2. Link quality
        link_quality = getattr(node, 'link_quality', 0.5)
        reward += self.r_link * link_quality
        
        # 3. Collision avoidance (inter-satellite)
        min_sat_distance = domain_context.get('min_satellite_distance', 1.0)
        
        if min_sat_distance > 0.15:  # Safe distance
            reward += self.r_collision_avoid
        elif min_sat_distance < 0.05:  # Critical proximity!
            reward -= self.r_collision_avoid * 2
        
        if collided:
            reward -= 100.0  # Catastrophic failure
        
        # 4. Debris avoidance
        min_debris_distance = domain_context.get('min_debris_distance', 1.0)
        
        if min_debris_distance > 0.10:
            reward += self.r_debris_avoid
        elif min_debris_distance < 0.03:
            reward -= self.r_debris_avoid * 2
        
        # 5. Power efficiency
        power_level = getattr(node, 'power_level', 80.0)
        
        # Reward maintaining power above threshold
        if power_level > 70.0:
            reward += self.r_power
        elif power_level < 30.0:
            reward -= self.r_power
        
        # Penalize wasteful thrust
        thrust_magnitude = np.linalg.norm(new_pos - old_pos)
        if thrust_magnitude > 0.01:  # Large maneuver
            reward -= self.r_power * 0.5
        
        # 6. Formation integrity (constellation coordination)
        reward += self.r_formation * loop_integrity
        
        # 7. Orbital stability
        # Penalize orbits that decay too much
        new_radius = np.linalg.norm(new_pos)
        old_radius = np.linalg.norm(old_pos)
        
        if new_radius < 0.2:  # Too close to Earth!
            reward -= 50.0
        
        radius_change = abs(new_radius - old_radius)
        if radius_change < 0.005:  # Stable orbit
            reward += 3.0
        
        return reward


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def create_domain_extractors(domain: str, config: Dict[str, Any]):
    """
    Factory function to create feature extractor and reward calculator.
    
    Args:
        domain: 'warehouse', 'traffic', or 'satellite'
        config: Configuration dictionary
    
    Returns:
        (feature_extractor, reward_calculator) tuple
    """
    extractors = {
        'warehouse': (WarehouseFeatureExtractor, WarehouseRewardCalculator),
        'traffic': (TrafficFeatureExtractor, TrafficRewardCalculator),
        'satellite': (SatelliteFeatureExtractor, SatelliteRewardCalculator),
    }
    
    if domain not in extractors:
        raise ValueError(
            f"Unknown domain '{domain}'. "
            f"Supported: {list(extractors.keys())}"
        )
    
    ExtractorClass, RewardClass = extractors[domain]
    
    feature_extractor = ExtractorClass()
    reward_calculator = RewardClass(config)
    
    return feature_extractor, reward_calculator


# =============================================================================
# USAGE EXAMPLE
# =============================================================================

if __name__ == '__main__':
    """
    Example: How to use domain extractors with FLOWRRA backend
    """
    
    # Example configuration
    config = {
        'rewards': {
            'r_flow': 5.0,
            'r_collision': 30.0,
            # ... other rewards
        }
    }
    
    # Create extractors for warehouse domain
    warehouse_extractor, warehouse_rewards = create_domain_extractors('warehouse', config)
    
    print("✅ Warehouse extractors created!")
    print(f"   Feature extractor: {warehouse_extractor.__class__.__name__}")
    print(f"   Reward calculator: {warehouse_rewards.__class__.__name__}")
    
    # Create extractors for satellite domain
    satellite_extractor, satellite_rewards = create_domain_extractors('satellite', config)
    
    print("\n✅ Satellite extractors created!")
    print(f"   Feature extractor: {satellite_extractor.__class__.__name__}")
    print(f"   Reward calculator: {satellite_rewards.__class__.__name__}")
    
    # Now use with FLOWRRA orchestrator:
    # from holon.core import FLOWRRA_Orchestrator
    # 
    # orchestrator = FLOWRRA_Orchestrator(
    #     config=config,
    #     feature_extractor=warehouse_extractor,
    #     reward_calculator=warehouse_rewards,
    #     mode='training'
    # )
