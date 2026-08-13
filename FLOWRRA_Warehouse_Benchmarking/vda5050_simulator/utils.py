import datetime
import math

def get_timestamp():
    """Get current timestamp in ISO8601 format"""
    now = datetime.datetime.utcnow()
    return now.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"

def get_topic_type(path):
    """Extract topic type from MQTT topic path"""
    parts = path.rsplit('/', 1)
    if len(parts) > 1:
        return parts[1]
    return path

def check_deviation_range(node_x, node_y, vehicle_x, vehicle_y, deviation_range):
    """Check if vehicle is within deviation range of node"""
    distance = math.sqrt((node_x - vehicle_x) ** 2 + (node_y - vehicle_y) ** 2)
    return distance <= deviation_range

def iterate_position(current_x, current_y, current_z, target_x, target_y, target_z, speed):
    """3D EXTENSION: Calculate next 3D position using Manhattan logic (Elevator Style)"""
    dx = target_x - current_x
    dy = target_y - current_y
    dist_xy = math.sqrt(dx**2 + dy**2)
    
    # PHASE 1: Drive horizontally to the destination (or elevator shaft)
    if dist_xy > 0.001: # Using a small threshold to prevent float precision bugs
        ratio = min(speed / dist_xy, 1.0)
        next_x = current_x + (dx * ratio)
        next_y = current_y + (dy * ratio)
        next_z = current_z  # Do not change height while driving!
        
        # Keep standard 2D yaw for theta (rotation on the XY plane)
        angle = math.atan2(dy, dx)
        return (next_x, next_y, next_z, angle)
        
    # PHASE 2: Once positioned exactly on the correct XY spot, use the elevator (Z axis)
    dz = target_z - current_z
    dist_z = abs(dz)
    
    if dist_z > 0.001:
        ratio = min(speed / dist_z, 1.0)
        next_x = target_x  # Lock into the exact X target
        next_y = target_y  # Lock into the exact Y target
        next_z = current_z + (dz * ratio)
        
        # Maintain the last known rotation while going up or down
        angle = 0.0 
        return (next_x, next_y, next_z, angle)
        
    # PHASE 3: We have arrived exactly at the 3D target!
    return (target_x, target_y, target_z, 0.0)

def get_distance(x1, y1, z1, x2, y2, z2):
    """3D EXTENSION: Calculate 3D Manhattan distance for warehouse grids"""
    return abs(x1 - x2) + abs(y1 - y2) + abs(z1 - z2)