import numpy as np
import random
import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation

# Constants
GRID_SIZE = 30
NUM_NODES = 12
ANGLE_STEPS = 36 # 360/36 = 10, so the arrows move 10 degrees at once
ROTATION_SPEED = 2 # How many angle_idx steps an arrow rotates per frame (2 means 20 degrees/frame)
NODE_MOVE_SPEED = 1 # How many grid steps a node moves per frame

# --- Helper Functions ---

def eye_direction_from_angle_idx(angle_idx):
    """Converts an angle index (0-35) to a (dx, dy) vector."""
    angle_deg = angle_idx * (360 / ANGLE_STEPS)
    rad = np.deg2rad(angle_deg)
    return np.cos(rad), np.sin(rad)

def get_valid_positions(num_nodes, grid_size):
    """Generates unique random positions for nodes."""
    positions = set()
    while len(positions) < num_nodes:
        # Ensure positions are within bounds for node size and arrow length
        x, y = random.randint(1, grid_size - 2), random.randint(1, grid_size - 2)
        positions.add((x, y))
    return list(positions) # Return as a list to maintain a consistent order for the loop


def calculate_angle_from_reference(reference_point, target_point):
    """Calculates the angle of target_point relative to reference_point (in radians)."""
    dx = target_point[0] - reference_point[0]
    dy = target_point[1] - reference_point[1]
    return math.atan2(dy, dx) # Returns angle from -0 to 1

def positional_shape(node_positions):
    """Generates the loop shape desired for the nodes"""
    
    if not node_positions:
        return []

    final_list = [node_positions[0]]
    chosen_point = node_positions[0]
    list_to_consider = list(node_positions[1:]) # Make a copy to modify

    for _ in range(len(node_positions) - 1):
        list_comparisons = []
        for node in list_to_consider:
            distance = math.hypot(node[0] - chosen_point[0], node[1] - chosen_point[1])
            # Calculate angle relative to the *current chosen_point*
            angle = calculate_angle_from_reference(chosen_point, node)
            list_comparisons.append((distance, angle, node)) # Store distance, angle, and the node
        
        # Sort by distance first, then by angle
        # If distances are equal, the angle will decide.
        chosen_point = sorted(list_comparisons)[0][2] # Get the node from the tuple
        
        final_list.append(chosen_point)
        list_to_consider.remove(chosen_point) # Remove the *actual node tuple*
    
    final_list.append(node_positions[0])
    return final_list

def calculate_target_angle_idx(from_pos, to_pos, angle_steps):
    """
    Calculates the target angle index for an arrow pointing from from_pos to to_pos.
    """
    x1, y1 = from_pos
    x2, y2 = to_pos
    
    # Calculate vector from current node to next node
    dx, dy = x2 - x1, y2 - y1

    # Handle cases where dx or dy is zero to avoid division by zero in arctan2 if it were used incorrectly
    # np.arctan2 handles this correctly, but good to be aware of the edge case.
    if dx == 0 and dy == 0: # If nodes are at the same position, no clear direction
        return 0 # Or handle as an error/no change

    angle_rad = np.arctan2(dy, dx)
    angle_deg = np.rad2deg(angle_rad)

    # Normalize angle to be between 0 and 360 degrees, then map to index
    angle_deg_normalized = angle_deg % 360
    
    # Convert to angle_idx, rounding to the nearest step
    target_idx = int(round(angle_deg_normalized / (360 / angle_steps))) % angle_steps
    return target_idx

def move_node(node_info, grid_size, all_nodes_data):
    """
    Calculates the next valid position for a single node.
    Avoids collision with grid boundaries and other nodes.
    """
    current_x, current_y = node_info['pos']
    possible_moves = [(0, NODE_MOVE_SPEED), (0, -NODE_MOVE_SPEED), (NODE_MOVE_SPEED, 0), (-NODE_MOVE_SPEED, 0)] # N, S, E, W
    random.shuffle(possible_moves) # Randomize direction preference

    for dx, dy in possible_moves:
        new_x, new_y = current_x + dx, current_y + dy

        # Check grid boundaries
        if not (0 <= new_x < grid_size and 0 <= new_y < grid_size):
            continue

        # Check collision with other nodes in their *current* positions
        is_colliding_with_other_node = False
        for other_node_info in all_nodes_data:
            if other_node_info['pos'] == (new_x, new_y) and other_node_info is not node_info:
                is_colliding_with_other_node = True
                break
        
        if is_colliding_with_other_node:
            continue

        # If a valid move is found, update the node's position
        node_info['pos'] = (new_x, new_y)
        return True # Successfully moved

    return False # No valid move found for this step (node stays in place)


# --- Initialization ---

# 1. Get positions for all nodes
node_positions = get_valid_positions(num_nodes=NUM_NODES, grid_size=GRID_SIZE)

# 1-2. Loop the positions for all nodes in desired right order 

node_positions_2 = positional_shape(node_positions)

# 2. Initialize nodes data (using a list of dictionaries for better structure)
nodes_data = []
for i, pos in enumerate(node_positions_2):
    nodes_data.append({
        'id': i,
        'pos': pos,
        'eye_angle_idx': random.randint(0, ANGLE_STEPS - 1), # Start with a random angle
        'target_angle_idx': 0, # Placeholder, will be calculated dynamically
        'patch_disk': None,    # Matplotlib patch for the disk
        'patch_arrow': None,   # Matplotlib patch for the arrow
        'patch_line': None     # Matplotlib patch for the connecting line
    })

# --- Matplotlib Setup ---
fig, ax = plt.subplots(figsize=(8, 8)) # Slightly larger figure for clarity
ax.set_xlim(0, GRID_SIZE)
ax.set_ylim(0, GRID_SIZE)
ax.set_title("Environment-A: Moving Disks with Dynamic Eye Direction & Loop")
ax.set_xticks([])
ax.set_yticks([])

# Draw nodes (disks) and initial arrows
all_patches = [] # To collect all patches for animation blitting

for node in nodes_data:
    x_center, y_center = node['pos'][0] + 0.5, node['pos'][1] + 0.5 # Center of the grid cell

    # Draw Disk
    disk_patch = patches.Circle((x_center, y_center), 0.4, fc='skyblue', ec='blue', lw=0.8, alpha=0.9)
    ax.add_patch(disk_patch)
    node['patch_disk'] = disk_patch
    all_patches.append(disk_patch)

    # Draw Arrow (initially pointing in a random direction)
    dx, dy = eye_direction_from_angle_idx(node['eye_angle_idx'])
    arrow_patch = patches.Arrow(x_center, y_center, 0.3 * dx, 0.3 * dy, 
                                width=0.2, fc='darkorange', ec='red', lw=1.0, zorder=2)
    ax.add_patch(arrow_patch)
    node['patch_arrow'] = arrow_patch
    all_patches.append(arrow_patch)

# Initialize Dotted Lines for the loop
# We need to store line objects so we can update their data in the animation
for i in range(NUM_NODES):
    current_node = nodes_data[i]
    next_node = nodes_data[i + 1]

    x1_center, y1_center = current_node['pos'][0] + 0.5, current_node['pos'][1] + 0.5
    x2_center, y2_center = next_node['pos'][0] + 0.5, next_node['pos'][1] + 0.5

    line, = ax.plot([x1_center, x2_center], [y1_center, y2_center], 
                    'k:', alpha=0.4, lw=1.0, zorder=1) # 'k:' for black dotted line
    current_node['patch_line'] = line # Store the line patch for the 'current_node'
    all_patches.append(line)

plt.gca().set_aspect('equal', adjustable='box')
plt.tight_layout()

# --- Animation Function ---

def animate_moving_loop(frame):
    """
    This function is called repeatedly by the animation.
    It moves nodes, rotates arrows, and updates connecting lines.
    """
    updated_patches = [] # Collect all patches that are modified in this frame

    # 1. Move all nodes
    # Create a copy of positions to avoid issues with simultaneous updates
    # (though move_node handles current position checks, this can be safer for complex interactions)
    
    # Simple movement: each node tries to move
    for node in nodes_data:
        move_node(node, GRID_SIZE, nodes_data) # Pass all_nodes_data for collision detection
        
        # Update disk position
        x_center, y_center = node['pos'][0] + 0.5, node['pos'][1] + 0.5
        node['patch_disk'].set_center((x_center, y_center))
        updated_patches.append(node['patch_disk'])

    # 2. Recalculate target angles and rotate arrows
    for i in range(NUM_NODES):
        current_node = nodes_data[i]
        next_node = nodes_data[(i + 1) % NUM_NODES] # Get the next node in the loop

        # Recalculate target angle based on NEW positions
        target_idx = calculate_target_angle_idx(current_node['pos'], next_node['pos'], ANGLE_STEPS)
        current_node['target_angle_idx'] = target_idx # Update the target for this frame

        # Rotate arrow towards target
        current_idx = current_node['eye_angle_idx']
        
        if current_idx != target_idx:
            # Shortest rotation path
            diff = target_idx - current_idx
            if diff > ANGLE_STEPS / 2:
                diff -= ANGLE_STEPS
            elif diff < -ANGLE_STEPS / 2:
                diff += ANGLE_STEPS

            if diff > 0:
                current_node['eye_angle_idx'] = (current_idx + ROTATION_SPEED) % ANGLE_STEPS
            elif diff < 0:
                current_node['eye_angle_idx'] = (current_idx - ROTATION_SPEED + ANGLE_STEPS) % ANGLE_STEPS
            
            # Update the arrow's direction vector
            x_center, y_center = current_node['pos'][0] + 0.5, current_node['pos'][1] + 0.5
            dx, dy = eye_direction_from_angle_idx(current_node['eye_angle_idx'])
            
            current_node['patch_arrow'].set_data(x_center, y_center, 0.3 * dx, 0.3 * dy)
            updated_patches.append(current_node['patch_arrow'])
        
        # 3. Update connecting lines (must be done after all nodes have moved)
        # The line connects current_node to next_node
        x1_center, y1_center = current_node['pos'][0] + 0.5, current_node['pos'][1] + 0.5
        x2_center, y2_center = next_node['pos'][0] + 0.5, next_node['pos'][1] + 0.5
        
        current_node['patch_line'].set_data([x1_center, x2_center], [y1_center, y2_center])
        updated_patches.append(current_node['patch_line'])

    return updated_patches # Return all patches that were modified

# Create the animation
# frames: number of frames (e.g., 400 steps for continuous movement)
# interval: delay between frames in milliseconds
ani = animation.FuncAnimation(fig, animate_moving_loop, frames=100, interval=1000, blit=False, repeat=True)

plt.show()