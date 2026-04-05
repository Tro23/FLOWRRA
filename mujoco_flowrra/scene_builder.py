# scene_builder.py
import random


def generate_swarm_xml(
    num_nodes: int,
    num_static_obs: int = 5,
    num_moving_obs: int = 3,
    # FIX #5 — Default mutable/random args are evaluated ONCE at import time in
    # Python, not at each call.  Every run would use the same spacing value for
    # the entire lifetime of the program.  Use None and compute inside instead.
    spacing: float = None,
) -> str:
    if spacing is None:
        spacing = random.uniform(1.5, 3.0)

    xml = """<mujoco model="FLOWRRA_Swarm">
        <option timestep="0.01" gravity="0 0 -9.81"/>
        <asset>
            <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="512"/>
            <texture name="grid" type="2d" builtin="checker" rgb1="0.1 0.2 0.3" rgb2="0.2 0.3 0.4" width="512" height="512"/>
            <material name="grid_mat" texture="grid" texrepeat="10 10" texuniform="true" reflectance="0.2"/>
        </asset>
        <worldbody>
            <light pos="0 0 10" dir="0 0 -1" diffuse="1 1 1"/>
            <geom name="floor" type="plane" size="20 20 0.1" material="grid_mat"/>
    """

    # --- 1. STATIC OBSTACLES (Pillars) ---
    for i in range(num_static_obs):
        ox = random.choice([random.uniform(-10, -3), random.uniform(3, 10)])
        oy = random.choice([random.uniform(-10, -3), random.uniform(3, 10)])
        xml += f"""
            <geom name="obs_{i}" type="cylinder" pos="{ox} {oy} 2.5" size="0.5 2.5" rgba="0.8 0.2 0.2 1"/>
        """

    # --- 2. MOVING OBSTACLES (Giant Bouncy Spheres) ---
    for i in range(num_moving_obs):
        ox = random.uniform(-8, 8)
        oy = random.uniform(-8, 8)
        oz = random.uniform(2, 5)
        xml += f"""
            <body name="mov_obs_{i}" pos="{ox} {oy} {oz}">
                <freejoint/>
                <geom type="sphere" size="0.4" mass="3.0" rgba="0.8 0.5 0.1 1"/>
            </body>
        """

    # --- 3. SPAWN THE DRONES ---
    grid_size = int(num_nodes**0.5) + 1
    for i in range(num_nodes):
        x = (i % grid_size) * spacing - (grid_size * spacing / 2)
        y = (i // grid_size) * spacing - (grid_size * spacing / 2)
        z = 3.0

        xml += f"""
            <body name="drone_{i}" pos="{x} {y} {z}">
                <freejoint/> <geom type="box" size="0.2 0.2 0.05" mass="1.0" rgba="0.2 0.5 0.8 1"/>
                <site name="t1_{i}" pos="0.2 0.2 0" type="cylinder" size="0.05 0.01" rgba="1 0 0 1"/>
                <site name="t2_{i}" pos="-0.2 0.2 0" type="cylinder" size="0.05 0.01" rgba="0 1 0 1"/>
                <site name="t3_{i}" pos="-0.2 -0.2 0" type="cylinder" size="0.05 0.01" rgba="0 1 0 1"/>
                <site name="t4_{i}" pos="0.2 -0.2 0" type="cylinder" size="0.05 0.01" rgba="1 0 0 1"/>
            </body>
        """

    xml += """
        </worldbody>
        <actuator>
    """

    for i in range(num_nodes):
        xml += f"""
            <motor name="m1_{i}" site="t1_{i}" gear="0 0 1 0 0 0" ctrlrange="0 15"/>
            <motor name="m2_{i}" site="t2_{i}" gear="0 0 1 0 0 0" ctrlrange="0 15"/>
            <motor name="m3_{i}" site="t3_{i}" gear="0 0 1 0 0 0" ctrlrange="0 15"/>
            <motor name="m4_{i}" site="t4_{i}" gear="0 0 1 0 0 0" ctrlrange="0 15"/>
        """

    xml += """
        </actuator>
    </mujoco>
    """
    return xml
