# pid_controller.py

import numpy as np


class DronePID:
    def __init__(self):
        # OUTER LOOP: Position/Velocity to Target Attitude (Roll/Pitch)
        self.kp_xy = 1.2
        self.kd_xy = 0.7

        # INNER LOOP: Gyroscope / Angular Rate limits
        self.kp_rate = 0.2  # Dampens the rotation speed
        self.max_tilt = 0.6  # Radians (~23 degrees) to prevent flips

        self.kp_z = 4.0
        self.ki_z = 0.1
        self.kd_z = 2.5

        self.integral_error = np.zeros(3)
        self.base_thrust = 9.81 / 4.0

    def compute(
        self,
        current_pos,
        target_pos,
        current_vel,
        target_vel,
        current_rpy,
        current_gyro,
        dt,
    ):
        """
        current_rpy: [roll, pitch, yaw] in radians
        current_gyro: [roll_rate, pitch_rate, yaw_rate] in rad/s
        """
        error = target_pos - current_pos
        d_error = target_vel - current_vel

        # 1. Z-Axis (Altitude) - Standard PID
        self.integral_error[2] += error[2] * dt
        self.integral_error[2] = np.clip(self.integral_error[2], -2.0, 2.0)
        thrust_adj = (
            (self.kp_z * error[2])
            + (self.ki_z * self.integral_error[2])
            + (self.kd_z * d_error[2])
        ) / 4.0

        # NEW: Hard-cap the thrust adjustment to prevent physics explosions (NaN)
        thrust_adj = np.clip(thrust_adj, -self.base_thrust, self.base_thrust * 2.0)
        total_thrust = self.base_thrust + thrust_adj

        # 2. OUTER LOOP: Determine desired tilt based on positional/velocity error
        target_pitch = (self.kp_xy * error[0]) + (self.kd_xy * d_error[0])
        target_roll = -(self.kp_xy * error[1]) - (self.kd_xy * d_error[1])

        # Cap the target tilt so it never commands an upside-down maneuver
        target_pitch = np.clip(target_pitch, -self.max_tilt, self.max_tilt)
        target_roll = np.clip(target_roll, -self.max_tilt, self.max_tilt)

        # 3. INNER LOOP (The Spinal Cord Gyro): Correct against angular momentum
        # Instead of feeding tilt directly to motors, we subtract the current angular velocity (gyro)
        pitch_cmd = target_pitch - current_rpy[1] - (self.kp_rate * current_gyro[1])
        roll_cmd = target_roll - current_rpy[0] - (self.kp_rate * current_gyro[0])
        yaw_cmd = 0.0 - (self.kp_rate * current_gyro[2])  # Keep yaw stable

        return total_thrust, roll_cmd, pitch_cmd, yaw_cmd
