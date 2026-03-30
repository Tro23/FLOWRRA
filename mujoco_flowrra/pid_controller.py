# pid_controller.py
import numpy as np


class DronePID:
    def __init__(self):
        # XY (Pitch/Roll) Gains: Snappy but with high damping (D) to prevent swinging
        self.kp_xy = 0.8
        self.ki_xy = 0.0
        self.kd_xy = 0.5

        # Z (Altitude) Gains: Very strong to fight gravity accurately
        self.kp_z = 3.0
        self.ki_z = 0.1
        self.kd_z = 2.0

        self.integral_error = np.zeros(3)
        self.last_error = np.zeros(3)

        # THE HOVER FIX: 9.81 N total required to hover / 4 motors = 2.4525 N per motor
        self.base_thrust = 9.81 / 4.0

    def compute(self, current_pos, target_pos, current_vel, target_vel, dt):
        error = target_pos - current_pos

        # 1. Proportional Term (Distance)
        p_out = error

        # 2. Integral Term (Accumulated error, with anti-windup limit)
        self.integral_error += error * dt
        self.integral_error = np.clip(self.integral_error, -2.0, 2.0)
        i_out = self.integral_error

        # 3. Derivative Term (Velocity Error)
        d_out = target_vel - current_vel

        # --- CALCULATE CORRECTIONS ---
        # Thrust adjustment needs to be divided across the 4 motors too
        thrust_adj = (
            (self.kp_z * p_out[2]) + (self.ki_z * i_out[2]) + (self.kd_z * d_out[2])
        ) / 4.0
        total_thrust = self.base_thrust + thrust_adj

        pitch = (
            (self.kp_xy * p_out[0]) + (self.ki_xy * i_out[0]) + (self.kd_xy * d_out[0])
        )
        roll = -(
            (self.kp_xy * p_out[1]) + (self.ki_xy * i_out[1]) + (self.kd_xy * d_out[1])
        )

        # Limit maximum tilt to 30 degrees (approx 0.5 rad) to prevent flipping
        pitch = np.clip(pitch, -0.5, 0.5)
        roll = np.clip(roll, -0.5, 0.5)
        yaw = 0.0

        self.last_error = error

        return total_thrust, roll, pitch, yaw
