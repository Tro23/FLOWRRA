"""
blender_visualizer.py

Enhanced Blender visualization for FLOWRRA deployment.

Features:
- Node spheres with trails (comet tails)
- Loop connections with break indicators
- Static and moving obstacles
- Coherence heatmap overlay
- Camera animation

Usage in Blender:
1. Open Blender
2. Switch to Scripting workspace
3. Load this script
4. Update JSON_PATH to point to your deployment_viz.json
5. Run script
"""

import json
import math

import bpy
import mathutils
from mathutils import Color, Vector

"""
blender_visualizer.py

Enhanced Blender visualization for FLOWRRA deployment.

Features:
- Node spheres with trails (comet tails)
- Loop connections with break indicators
- Static and moving obstacles
- Coherence heatmap overlay
- Camera animation

Usage in Blender:
1. Open Blender
2. Switch to Scripting workspace
3. Load this script
4. Update JSON_PATH to point to your deployment_viz.json
5. Run script
"""

import json
import math

import bpy
import mathutils
from mathutils import Color, Vector

# ============================================================================
# CONFIGURATION
# ============================================================================

JSON_PATH = "deployment_viz.json"  # Update this path!

# Visual settings
NODE_RADIUS = 0.15
TRAIL_LENGTH = 20  # Number of past positions to show
OBSTACLE_MATERIAL_COLOR = (0.8, 0.2, 0.1, 0.6)  # Red, semi-transparent
CONNECTION_THICKNESS = 0.05
BROKEN_CONNECTION_COLOR = (1, 0, 0)  # Red
INTACT_CONNECTION_COLOR = (0, 1, 0.3)  # Green

# Animation settings
FRAME_RATE = 30  # FPS
TIME_SCALE = 1.0  # Speed multiplier

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================


def clear_scene():
    """Remove all mesh objects from the scene."""
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)

    # Clear materials
    for material in bpy.data.materials:
        bpy.data.materials.remove(material)


def create_material(name, color, emission=False, alpha=1.0):
    """Create a material with given color."""
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    nodes.clear()

    # Principled BSDF
    bsdf = nodes.new(type="ShaderNodeBsdfPrincipled")
    bsdf.inputs["Base Color"].default_value = (*color[:3], 1.0)
    bsdf.inputs["Metallic"].default_value = 0.3
    bsdf.inputs["Roughness"].default_value = 0.4
    bsdf.inputs["Alpha"].default_value = alpha

    if emission:
        bsdf.inputs["Emission"].default_value = (*color[:3], 1.0)
        bsdf.inputs["Emission Strength"].default_value = 2.0

    # Output
    output = nodes.new(type="ShaderNodeOutputMaterial")
    mat.node_tree.links.new(bsdf.outputs["BSDF"], output.inputs["Surface"])

    # Enable transparency if needed
    if alpha < 1.0:
        mat.blend_method = "BLEND"

    return mat


def create_sphere(location, radius, material, name="Sphere"):
    """Create a sphere at given location."""
    bpy.ops.mesh.primitive_uv_sphere_add(radius=radius, location=location)
    obj = bpy.context.active_object
    obj.name = name

    if material:
        if obj.data.materials:
            obj.data.materials[0] = material
        else:
            obj.data.materials.append(material)

    return obj


def create_cylinder_between_points(p1, p2, radius, material, name="Connection"):
    """Create a cylinder connecting two points."""
    # Calculate direction and length
    direction = Vector(p2) - Vector(p1)
    length = direction.length

    if length < 0.001:
        return None

    # Create cylinder
    bpy.ops.mesh.primitive_cylinder_add(
        radius=radius, depth=length, location=Vector(p1) + direction * 0.5
    )

    obj = bpy.context.active_object
    obj.name = name

    # Rotate to align with direction
    direction.normalize()
    up = Vector((0, 0, 1))

    if abs(direction.dot(up)) < 0.99:
        axis = up.cross(direction)
        angle = math.acos(up.dot(direction))
        obj.rotation_mode = "AXIS_ANGLE"
        obj.rotation_axis_angle = (angle, *axis)
    elif direction.dot(up) < 0:
        obj.rotation_euler = (math.pi, 0, 0)

    if material:
        if obj.data.materials:
            obj.data.materials[0] = material
        else:
            obj.data.materials.append(material)

    return obj


def create_trail(positions, radius, material, name="Trail"):
    """Create a comet trail from list of positions."""
    if len(positions) < 2:
        return None

    # Create curve
    curve_data = bpy.data.curves.new(name=name, type="CURVE")
    curve_data.dimensions = "3D"
    curve_data.bevel_depth = radius
    curve_data.bevel_resolution = 4

    # Create spline
    spline = curve_data.splines.new("BEZIER")
    spline.bezier_points.add(len(positions) - 1)

    for i, pos in enumerate(positions):
        point = spline.bezier_points[i]
        point.co = pos
        point.handle_left_type = "AUTO"
        point.handle_right_type = "AUTO"

    # Create object
    obj = bpy.data.objects.new(name, curve_data)
    bpy.context.collection.objects.link(obj)

    if material:
        obj.data.materials.append(material)

    return obj


def coherence_to_color(coherence):
    """Map coherence value to color (red=low, green=high)."""
    # Red -> Yellow -> Green gradient
    r = 1.0 - coherence
    g = coherence
    b = 0.0
    return (r, g, b)


# ============================================================================
# MAIN VISUALIZATION
# ============================================================================


def visualize_flowrra(json_path):
    """Main visualization function."""
    print(f"Loading data from {json_path}...")

    with open(json_path, "r") as f:
        data = json.load(f)

    config = data["config"]
    frames = data["frames"]

    world_bounds = config["world_bounds"]
    num_nodes = config["num_nodes"]

    print(f"Loaded {len(frames)} frames with {num_nodes} nodes")

    # Clear scene
    clear_scene()

    # Set up scene
    bpy.context.scene.frame_start = 0
    bpy.context.scene.frame_end = len(frames) - 1
    bpy.context.scene.render.fps = FRAME_RATE

    # Create materials
    node_materials = {}
    for i in range(num_nodes):
        hue = i / num_nodes
        color = Color()
        color.hsv = (hue, 0.8, 0.9)
        mat = create_material(f"Node_{i}", color.to_tuple(), emission=True)
        node_materials[i] = mat

    trail_materials = {}
    for i in range(num_nodes):
        hue = i / num_nodes
        color = Color()
        color.hsv = (hue, 0.6, 0.7)
        mat = create_material(f"Trail_{i}", color.to_tuple(), alpha=0.5)
        trail_materials[i] = mat

    intact_conn_mat = create_material(
        "IntactConnection", INTACT_CONNECTION_COLOR, alpha=0.7
    )
    broken_conn_mat = create_material(
        "BrokenConnection", BROKEN_CONNECTION_COLOR, alpha=0.5
    )
    obstacle_mat = create_material(
        "Obstacle", OBSTACLE_MATERIAL_COLOR[:3], alpha=OBSTACLE_MATERIAL_COLOR[3]
    )

    # Create ground plane with coherence texture
    bpy.ops.mesh.primitive_plane_add(
        size=max(world_bounds) * 1.2, location=(0, 0, -0.5)
    )
    ground = bpy.context.active_object
    ground.name = "Ground"
    ground_mat = create_material("GroundMat", (0.1, 0.1, 0.15), alpha=0.8)
    ground.data.materials.append(ground_mat)

    # Create static obstacles (persistent across frames)
    if frames[0].get("obstacles"):
        for obs in frames[0]["obstacles"]:
            if obs["is_static"]:
                pos = obs["pos"] + [-0.3]  # Slightly below ground
                obs_obj = create_sphere(
                    pos, obs["radius"], obstacle_mat, name=f"Obstacle_{obs['id']}"
                )

    # Animate nodes and connections
    print("Creating animation...")

    node_objects = {}
    trail_objects = {}
    connection_objects = {}

    # Initialize node objects
    for i in range(num_nodes):
        initial_pos = frames[0]["nodes"][i]["pos"] + [0]  # Add Z coordinate
        node_obj = create_sphere(
            initial_pos, NODE_RADIUS, node_materials[i], name=f"Node_{i}"
        )
        node_objects[i] = node_obj

    # Animate frame by frame
    for frame_idx, frame in enumerate(frames):
        bpy.context.scene.frame_set(frame_idx)

        # Update node positions
        node_positions_history = {}

        for node_data in frame["nodes"]:
            node_id = node_data["id"]
            pos = Vector(node_data["pos"] + [0])

            if node_id in node_objects:
                node_obj = node_objects[node_id]
                node_obj.location = pos
                node_obj.keyframe_insert(data_path="location", frame=frame_idx)

                # Track position history for trails
                if node_id not in node_positions_history:
                    node_positions_history[node_id] = []
                node_positions_history[node_id].append(pos)

        # Create/update trails (every 5 frames to reduce clutter)
        if frame_idx % 5 == 0 and frame_idx > TRAIL_LENGTH:
            for node_id in range(num_nodes):
                # Get last N positions
                trail_positions = []
                for past_frame_idx in range(
                    max(0, frame_idx - TRAIL_LENGTH), frame_idx
                ):
                    if past_frame_idx < len(frames):
                        past_pos = frames[past_frame_idx]["nodes"][node_id]["pos"] + [0]
                        trail_positions.append(Vector(past_pos))

                if len(trail_positions) > 1:
                    trail_name = f"Trail_{node_id}_F{frame_idx}"
                    trail_obj = create_trail(
                        trail_positions,
                        NODE_RADIUS * 0.3,
                        trail_materials[node_id],
                        name=trail_name,
                    )
                    trail_objects[trail_name] = trail_obj

        # Update connections
        if "connections" in frame:
            # Clear old connection objects
            for conn_name, conn_obj in list(connection_objects.items()):
                bpy.data.objects.remove(conn_obj, do_unlink=True)
            connection_objects.clear()

            for conn in frame["connections"]:
                node_a_id = conn["node_a"]
                node_b_id = conn["node_b"]
                is_broken = conn["broken"]

                # Get positions
                pos_a = Vector(frame["nodes"][node_a_id]["pos"] + [0])
                pos_b = Vector(frame["nodes"][node_b_id]["pos"] + [0])

                # Create connection cylinder
                mat = broken_conn_mat if is_broken else intact_conn_mat
                conn_obj = create_cylinder_between_points(
                    pos_a,
                    pos_b,
                    CONNECTION_THICKNESS,
                    mat,
                    name=f"Conn_{node_a_id}_{node_b_id}_F{frame_idx}",
                )

                if conn_obj:
                    connection_objects[conn_obj.name] = conn_obj

        # Update moving obstacles
        if "obstacles" in frame:
            for obs in frame["obstacles"]:
                if not obs["is_static"]:
                    obs_name = f"Obstacle_{obs['id']}"
                    if obs_name in bpy.data.objects:
                        obs_obj = bpy.data.objects[obs_name]
                    else:
                        pos = obs["pos"] + [-0.3]
                        obs_obj = create_sphere(
                            pos, obs["radius"], obstacle_mat, name=obs_name
                        )

                    obs_obj.location = Vector(obs["pos"] + [-0.3])
                    obs_obj.keyframe_insert(data_path="location", frame=frame_idx)

        # Progress indicator
        if frame_idx % 50 == 0:
            progress = (frame_idx / len(frames)) * 100
            print(f"  Progress: {progress:.1f}% ({frame_idx}/{len(frames)} frames)")

    # Add camera
    cam_loc = (
        max(world_bounds) * 0.7,
        -max(world_bounds) * 0.7,
        max(world_bounds) * 0.8,
    )
    bpy.ops.object.camera_add(location=cam_loc)
    camera = bpy.context.active_object
    camera.rotation_euler = (math.radians(60), 0, math.radians(45))
    bpy.context.scene.camera = camera

    # Add lighting
    bpy.ops.object.light_add(type="SUN", location=(5, 5, 10))
    sun = bpy.context.active_object
    sun.data.energy = 3.0

    bpy.ops.object.light_add(type="AREA", location=(0, 0, 5))
    area = bpy.context.active_object
    area.data.energy = 500
    area.data.size = max(world_bounds)

    print("✅ Visualization complete!")
    print(f"   Total frames: {len(frames)}")
    print(f"   Duration: {len(frames) / FRAME_RATE:.1f} seconds")
    print("   Press SPACE to play animation")


# ============================================================================
# RUN
# ============================================================================

if __name__ == "__main__":
    visualize_flowrra(JSON_PATH)
# ============================================================================
# CONFIGURATION
# ============================================================================

JSON_PATH = "deployment_viz.json"  # Update this path!

# Visual settings
NODE_RADIUS = 0.15
TRAIL_LENGTH = 20  # Number of past positions to show
OBSTACLE_MATERIAL_COLOR = (0.8, 0.2, 0.1, 0.6)  # Red, semi-transparent
CONNECTION_THICKNESS = 0.05
BROKEN_CONNECTION_COLOR = (1, 0, 0)  # Red
INTACT_CONNECTION_COLOR = (0, 1, 0.3)  # Green

# Animation settings
FRAME_RATE = 30  # FPS
TIME_SCALE = 1.0  # Speed multiplier

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================


def clear_scene():
    """Remove all mesh objects from the scene."""
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)

    # Clear materials
    for material in bpy.data.materials:
        bpy.data.materials.remove(material)


def create_material(name, color, emission=False, alpha=1.0):
    """Create a material with given color."""
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    nodes.clear()

    # Principled BSDF
    bsdf = nodes.new(type="ShaderNodeBsdfPrincipled")
    bsdf.inputs["Base Color"].default_value = (*color[:3], 1.0)
    bsdf.inputs["Metallic"].default_value = 0.3
    bsdf.inputs["Roughness"].default_value = 0.4
    bsdf.inputs["Alpha"].default_value = alpha

    if emission:
        bsdf.inputs["Emission"].default_value = (*color[:3], 1.0)
        bsdf.inputs["Emission Strength"].default_value = 2.0

    # Output
    output = nodes.new(type="ShaderNodeOutputMaterial")
    mat.node_tree.links.new(bsdf.outputs["BSDF"], output.inputs["Surface"])

    # Enable transparency if needed
    if alpha < 1.0:
        mat.blend_method = "BLEND"

    return mat


def create_sphere(location, radius, material, name="Sphere"):
    """Create a sphere at given location."""
    bpy.ops.mesh.primitive_uv_sphere_add(radius=radius, location=location)
    obj = bpy.context.active_object
    obj.name = name

    if material:
        if obj.data.materials:
            obj.data.materials[0] = material
        else:
            obj.data.materials.append(material)

    return obj


def create_cylinder_between_points(p1, p2, radius, material, name="Connection"):
    """Create a cylinder connecting two points."""
    # Calculate direction and length
    direction = Vector(p2) - Vector(p1)
    length = direction.length

    if length < 0.001:
        return None

    # Create cylinder
    bpy.ops.mesh.primitive_cylinder_add(
        radius=radius, depth=length, location=Vector(p1) + direction * 0.5
    )

    obj = bpy.context.active_object
    obj.name = name

    # Rotate to align with direction
    direction.normalize()
    up = Vector((0, 0, 1))

    if abs(direction.dot(up)) < 0.99:
        axis = up.cross(direction)
        angle = math.acos(up.dot(direction))
        obj.rotation_mode = "AXIS_ANGLE"
        obj.rotation_axis_angle = (angle, *axis)
    elif direction.dot(up) < 0:
        obj.rotation_euler = (math.pi, 0, 0)

    if material:
        if obj.data.materials:
            obj.data.materials[0] = material
        else:
            obj.data.materials.append(material)

    return obj


def create_trail(positions, radius, material, name="Trail"):
    """Create a comet trail from list of positions."""
    if len(positions) < 2:
        return None

    # Create curve
    curve_data = bpy.data.curves.new(name=name, type="CURVE")
    curve_data.dimensions = "3D"
    curve_data.bevel_depth = radius
    curve_data.bevel_resolution = 4

    # Create spline
    spline = curve_data.splines.new("BEZIER")
    spline.bezier_points.add(len(positions) - 1)

    for i, pos in enumerate(positions):
        point = spline.bezier_points[i]
        point.co = pos
        point.handle_left_type = "AUTO"
        point.handle_right_type = "AUTO"

    # Create object
    obj = bpy.data.objects.new(name, curve_data)
    bpy.context.collection.objects.link(obj)

    if material:
        obj.data.materials.append(material)

    return obj


def coherence_to_color(coherence):
    """Map coherence value to color (red=low, green=high)."""
    # Red -> Yellow -> Green gradient
    r = 1.0 - coherence
    g = coherence
    b = 0.0
    return (r, g, b)


# ============================================================================
# MAIN VISUALIZATION
# ============================================================================


def visualize_flowrra(json_path):
    """Main visualization function."""
    print(f"Loading data from {json_path}...")

    with open(json_path, "r") as f:
        data = json.load(f)

    config = data["config"]
    frames = data["frames"]

    world_bounds = config["world_bounds"]
    num_nodes = config["num_nodes"]

    print(f"Loaded {len(frames)} frames with {num_nodes} nodes")

    # Clear scene
    clear_scene()

    # Set up scene
    bpy.context.scene.frame_start = 0
    bpy.context.scene.frame_end = len(frames) - 1
    bpy.context.scene.render.fps = FRAME_RATE

    # Create materials
    node_materials = {}
    for i in range(num_nodes):
        hue = i / num_nodes
        color = Color()
        color.hsv = (hue, 0.8, 0.9)
        mat = create_material(f"Node_{i}", color.to_tuple(), emission=True)
        node_materials[i] = mat

    trail_materials = {}
    for i in range(num_nodes):
        hue = i / num_nodes
        color = Color()
        color.hsv = (hue, 0.6, 0.7)
        mat = create_material(f"Trail_{i}", color.to_tuple(), alpha=0.5)
        trail_materials[i] = mat

    intact_conn_mat = create_material(
        "IntactConnection", INTACT_CONNECTION_COLOR, alpha=0.7
    )
    broken_conn_mat = create_material(
        "BrokenConnection", BROKEN_CONNECTION_COLOR, alpha=0.5
    )
    obstacle_mat = create_material(
        "Obstacle", OBSTACLE_MATERIAL_COLOR[:3], alpha=OBSTACLE_MATERIAL_COLOR[3]
    )

    # Create ground plane with coherence texture
    bpy.ops.mesh.primitive_plane_add(
        size=max(world_bounds) * 1.2, location=(0, 0, -0.5)
    )
    ground = bpy.context.active_object
    ground.name = "Ground"
    ground_mat = create_material("GroundMat", (0.1, 0.1, 0.15), alpha=0.8)
    ground.data.materials.append(ground_mat)

    # Create static obstacles (persistent across frames)
    if frames[0].get("obstacles"):
        for obs in frames[0]["obstacles"]:
            if obs["is_static"]:
                pos = obs["pos"] + [-0.3]  # Slightly below ground
                obs_obj = create_sphere(
                    pos, obs["radius"], obstacle_mat, name=f"Obstacle_{obs['id']}"
                )

    # Animate nodes and connections
    print("Creating animation...")

    node_objects = {}
    trail_objects = {}
    connection_objects = {}

    # Initialize node objects
    for i in range(num_nodes):
        initial_pos = frames[0]["nodes"][i]["pos"] + [0]  # Add Z coordinate
        node_obj = create_sphere(
            initial_pos, NODE_RADIUS, node_materials[i], name=f"Node_{i}"
        )
        node_objects[i] = node_obj

    # Animate frame by frame
    for frame_idx, frame in enumerate(frames):
        bpy.context.scene.frame_set(frame_idx)

        # Update node positions
        node_positions_history = {}

        for node_data in frame["nodes"]:
            node_id = node_data["id"]
            pos = Vector(node_data["pos"] + [0])

            if node_id in node_objects:
                node_obj = node_objects[node_id]
                node_obj.location = pos
                node_obj.keyframe_insert(data_path="location", frame=frame_idx)

                # Track position history for trails
                if node_id not in node_positions_history:
                    node_positions_history[node_id] = []
                node_positions_history[node_id].append(pos)

        # Create/update trails (every 5 frames to reduce clutter)
        if frame_idx % 5 == 0 and frame_idx > TRAIL_LENGTH:
            for node_id in range(num_nodes):
                # Get last N positions
                trail_positions = []
                for past_frame_idx in range(
                    max(0, frame_idx - TRAIL_LENGTH), frame_idx
                ):
                    if past_frame_idx < len(frames):
                        past_pos = frames[past_frame_idx]["nodes"][node_id]["pos"] + [0]
                        trail_positions.append(Vector(past_pos))

                if len(trail_positions) > 1:
                    trail_name = f"Trail_{node_id}_F{frame_idx}"
                    trail_obj = create_trail(
                        trail_positions,
                        NODE_RADIUS * 0.3,
                        trail_materials[node_id],
                        name=trail_name,
                    )
                    trail_objects[trail_name] = trail_obj

        # Update connections
        if "connections" in frame:
            # Clear old connection objects
            for conn_name, conn_obj in list(connection_objects.items()):
                bpy.data.objects.remove(conn_obj, do_unlink=True)
            connection_objects.clear()

            for conn in frame["connections"]:
                node_a_id = conn["node_a"]
                node_b_id = conn["node_b"]
                is_broken = conn["broken"]

                # Get positions
                pos_a = Vector(frame["nodes"][node_a_id]["pos"] + [0])
                pos_b = Vector(frame["nodes"][node_b_id]["pos"] + [0])

                # Create connection cylinder
                mat = broken_conn_mat if is_broken else intact_conn_mat
                conn_obj = create_cylinder_between_points(
                    pos_a,
                    pos_b,
                    CONNECTION_THICKNESS,
                    mat,
                    name=f"Conn_{node_a_id}_{node_b_id}_F{frame_idx}",
                )

                if conn_obj:
                    connection_objects[conn_obj.name] = conn_obj

        # Update moving obstacles
        if "obstacles" in frame:
            for obs in frame["obstacles"]:
                if not obs["is_static"]:
                    obs_name = f"Obstacle_{obs['id']}"
                    if obs_name in bpy.data.objects:
                        obs_obj = bpy.data.objects[obs_name]
                    else:
                        pos = obs["pos"] + [-0.3]
                        obs_obj = create_sphere(
                            pos, obs["radius"], obstacle_mat, name=obs_name
                        )

                    obs_obj.location = Vector(obs["pos"] + [-0.3])
                    obs_obj.keyframe_insert(data_path="location", frame=frame_idx)

        # Progress indicator
        if frame_idx % 50 == 0:
            progress = (frame_idx / len(frames)) * 100
            print(f"  Progress: {progress:.1f}% ({frame_idx}/{len(frames)} frames)")

    # Add camera
    cam_loc = (
        max(world_bounds) * 0.7,
        -max(world_bounds) * 0.7,
        max(world_bounds) * 0.8,
    )
    bpy.ops.object.camera_add(location=cam_loc)
    camera = bpy.context.active_object
    camera.rotation_euler = (math.radians(60), 0, math.radians(45))
    bpy.context.scene.camera = camera

    # Add lighting
    bpy.ops.object.light_add(type="SUN", location=(5, 5, 10))
    sun = bpy.context.active_object
    sun.data.energy = 3.0

    bpy.ops.object.light_add(type="AREA", location=(0, 0, 5))
    area = bpy.context.active_object
    area.data.energy = 500
    area.data.size = max(world_bounds)

    print("✅ Visualization complete!")
    print(f"   Total frames: {len(frames)}")
    print(f"   Duration: {len(frames) / FRAME_RATE:.1f} seconds")
    print("   Press SPACE to play animation")


# ============================================================================
# RUN
# ============================================================================

if __name__ == "__main__":
    visualize_flowrra(JSON_PATH)
