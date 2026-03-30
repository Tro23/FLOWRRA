import time

import mujoco
import mujoco.viewer
import numpy as np
from motor_mixer import QuadcopterMixer

# 1. Import our Modular Architecture
from pid_controller import DronePID

# 2. Load the Universe
model = mujoco.MjModel.from_xml_path("drone.xml")
data = mujoco.MjData(model)

# 3. Initialize the Mechanics
pid = DronePID()
mixer = QuadcopterMixer(motor_limit=15.0)

# 4. Set up a flight path (Waypoints: [X, Y, Z] AND Target Velocities: [Vx, Vy, Vz])
waypoints = [
    np.array([0.0, 0.0, 2.0]),  # Takeoff and hover at 2m
    np.array([2.0, 0.0, 2.0]),  # Fly forward 2m
    np.array([2.0, -2.0, 2.0]),  # Fly right 2m
    np.array([0.0, 0.0, 1.0]),  # Return to center, drop to 1m
]

# For the hardcoded test, we tell it to completely stop (0m/s) at each waypoint.
# agent.py will soon replace this with dynamic momentum vectors!
target_velocities = [
    np.array([0.0, 0.0, 0.0]),
    np.array([0.0, 0.0, 0.0]),
    np.array([0.0, 0.0, 0.0]),
    np.array([0.0, 0.0, 0.0]),
]

current_wp_idx = 0
target_pos = waypoints[current_wp_idx]
target_vel = target_velocities[current_wp_idx]

# --- START SIMULATION ---
with mujoco.viewer.launch_passive(model, data) as viewer:
    print("Universe loaded. Commencing 6D autonomous flight test...")
    last_time = data.time

    while viewer.is_running():
        # A. Calculate exact timestep (dt)
        current_time = data.time
        dt = current_time - last_time
        last_time = current_time
        if dt == 0:
            dt = 0.01

        # B. SENSE: Get absolute truth from the Universe
        current_pos = data.qpos[0:3]
        current_vel = data.qvel[0:3]

        # --- SAFETY RESET ---
        if current_pos[2] < 0.1 or np.linalg.norm(current_pos) > 10.0:
            mujoco.mj_resetData(model, data)
            mujoco.mj_forward(model, data)
            pid.integral_error = np.zeros(3)
            pid.last_error = np.zeros(3)
            current_wp_idx = 0
            target_pos = waypoints[current_wp_idx]
            target_vel = target_velocities[current_wp_idx]
            last_time = data.time
            continue

        # C. WAYPOINT LOGIC (To be replaced by agent.py soon!)
        distance_to_target = np.linalg.norm(target_pos - current_pos)
        if distance_to_target < 0.1 and current_wp_idx < len(waypoints) - 1:
            current_wp_idx += 1
            target_pos = waypoints[current_wp_idx]
            target_vel = target_velocities[
                current_wp_idx
            ]  # <--- Grab the new target velocity
            print(f"Target reached! Moving to Waypoint {current_wp_idx}: {target_pos}")

        # D. THINK: PID calculates required forces using BOTH position and velocity errors
        thrust, roll, pitch, yaw = pid.compute(
            current_pos=current_pos,
            target_pos=target_pos,
            current_vel=current_vel,
            target_vel=target_vel,  # <--- Pass it into the PID
            dt=dt,
        )

        # E. REFLEX: Mixer translates forces to physical motors
        m1, m2, m3, m4 = mixer.mix(thrust, roll, pitch, yaw)

        # F. ACT: Apply power and step physics
        data.ctrl[0] = m1
        data.ctrl[1] = m2
        data.ctrl[2] = m3
        data.ctrl[3] = m4

        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(0.01)
