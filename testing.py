my_list = [25,10,20,30,40,44,48,32,22, 50]

starting_element = my_list[0]


my_list_2 = sorted(my_list[1:])

right_list =[]
left_list = []
for i in my_list_2:
    if i > starting_element:
        right_list.append(i)

    elif i < starting_element:
        left_list.append(i)

final_list = [starting_element] + sorted(right_list) + sorted(left_list,reverse=True)

print(final_list)

import math
from scipy.spatial import ConvexHull

node_positions = [(10,10), (10,15), (10,5), (20,25), (30,35), (45,50)]


def calculate_angle_from_reference(reference_point, target_point):
    """Calculates the angle of target_point relative to reference_point (in radians)."""
    dx = target_point[0] - reference_point[0]
    dy = target_point[1] - reference_point[1]
    angle_radians = math.atan2(dy, dx)
    return abs(math.cos(angle_radians)) # Returns angle from -0 to 1

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

# Test
ordered_loop = positional_shape(node_positions)
print(ordered_loop)