# motor_mixer.py

class QuadcopterMixer:
    def __init__(self, motor_limit=10.0):
        # We add a motor limit to prevent the physics engine from applying infinite force
        self.motor_limit = motor_limit

    def mix(self, base_thrust, roll, pitch, yaw):
        """
        Translates aerodynamic forces into 4 specific motor speeds.
        Assumes standard X-configuration quadcopter.
        """
        m1 = base_thrust - roll - pitch + yaw  # Front Right
        m2 = base_thrust + roll - pitch - yaw  # Front Left
        m3 = base_thrust + roll + pitch + yaw  # Back Left
        m4 = base_thrust - roll + pitch - yaw  # Back Right
        
        # Ensure motors don't spin backward (min 0) and don't exceed max power
        m1 = max(0.0, min(self.motor_limit, m1))
        m2 = max(0.0, min(self.motor_limit, m2))
        m3 = max(0.0, min(self.motor_limit, m3))
        m4 = max(0.0, min(self.motor_limit, m4))
        
        return m1, m2, m3, m4