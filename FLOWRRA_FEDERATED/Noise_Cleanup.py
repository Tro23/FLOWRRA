"""
noise_cleanup.py

State-of-the-art sensor fusion and noise filtering for FLOWRRA.

FLOWRRA operates on PROCESSED sensor data. This module handles:
- Kalman filtering for position/velocity estimation
- Particle filtering for multi-modal distributions
- Consensus filtering for distributed sensor fusion
- Outlier rejection (RANSAC, Mahalanobis distance)
- Temporal smoothing (Savitzky-Golay, exponential smoothing)
- Communication link quality assessment

Architecture:
    Raw Sensors → Noise Filters → Sensor Fusion → FLOWRRA Orchestrator

Key Papers Implemented:
- Kalman Filter (Kalman 1960)
- Particle Filter (Gordon et al. 1993)
- Consensus Kalman Filter (Olfati-Saber 2007)
- RANSAC (Fischler & Bolles 1981)
"""

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.signal import savgol_filter
from scipy.spatial.distance import mahalanobis

# =============================================================================
# KALMAN FILTER - Position/Velocity Estimation
# =============================================================================


class KalmanFilter:
    """
    Extended Kalman Filter for node state estimation.

    State vector: [x, y, vx, vy] (2D) or [x, y, z, vx, vy, vz] (3D)

    Handles:
    - Position noise (GPS, visual odometry)
    - Velocity estimation from noisy positions
    - Prediction during sensor dropout
    """

    def __init__(self, dimensions: int = 2, dt: float = 1.0):
        """
        Args:
            dimensions: 2D or 3D
            dt: Time step
        """
        self.dims = dimensions
        self.dt = dt
        self.state_dim = dimensions * 2  # [pos, vel]

        # State: [x, y, (z), vx, vy, (vz)]
        self.x = np.zeros(self.state_dim)

        # State covariance matrix
        self.P = np.eye(self.state_dim) * 0.1

        # Process noise covariance (how much we trust dynamics model)
        self.Q = np.eye(self.state_dim) * 0.01
        self.Q[dimensions:, dimensions:] *= 0.05  # Higher velocity uncertainty

        # Measurement noise covariance (how much we trust sensors)
        self.R = np.eye(dimensions) * 0.01  # Position measurements only

        # State transition matrix (constant velocity model)
        self.F = self._build_transition_matrix()

        # Measurement matrix (we only measure position, not velocity)
        self.H = np.zeros((dimensions, self.state_dim))
        self.H[:dimensions, :dimensions] = np.eye(dimensions)

        # Track measurement history for outlier detection
        self.innovation_history = deque(maxlen=10)

    def _build_transition_matrix(self) -> np.ndarray:
        """Build state transition matrix for constant velocity model."""
        F = np.eye(self.state_dim)
        # Position updates: x(t+1) = x(t) + vx(t) * dt
        for i in range(self.dims):
            F[i, self.dims + i] = self.dt
        return F

    def predict(self) -> np.ndarray:
        """
        Prediction step (when no measurement available).

        Returns:
            Predicted state
        """
        # State prediction: x = F * x
        self.x = self.F @ self.x

        # Covariance prediction: P = F * P * F^T + Q
        self.P = self.F @ self.P @ self.F.T + self.Q

        return self.x[: self.dims]  # Return position only

    def update(self, measurement: np.ndarray) -> np.ndarray:
        """
        Update step (when measurement available).

        Args:
            measurement: Noisy position measurement [x, y, (z)]

        Returns:
            Filtered position estimate
        """
        # Innovation: y = z - H * x
        innovation = measurement - (self.H @ self.x)

        # Innovation covariance: S = H * P * H^T + R
        S = self.H @ self.P @ self.H.T + self.R

        # Outlier detection: Mahalanobis distance
        try:
            mahal_dist = np.sqrt(innovation.T @ np.linalg.inv(S) @ innovation)

            # Reject if too far (likely outlier)
            if mahal_dist > 3.0:  # 3-sigma rule
                # Use prediction instead of corrupted measurement
                return self.predict()
        except np.linalg.LinAlgError:
            pass  # Singular matrix, skip outlier check

        # Kalman gain: K = P * H^T * S^-1
        K = self.P @ self.H.T @ np.linalg.inv(S)

        # State update: x = x + K * y
        self.x = self.x + K @ innovation

        # Covariance update: P = (I - K * H) * P
        I_KH = np.eye(self.state_dim) - K @ self.H
        self.P = I_KH @ self.P

        # Track innovation for adaptive tuning
        self.innovation_history.append(np.linalg.norm(innovation))

        return self.x[: self.dims]  # Return filtered position

    def get_velocity(self) -> np.ndarray:
        """Get estimated velocity."""
        return self.x[self.dims :]

    def reset(self, initial_state: np.ndarray):
        """Reset filter with new initial state."""
        if len(initial_state) == self.dims:
            # Only position provided, assume zero velocity
            self.x[: self.dims] = initial_state
            self.x[self.dims :] = 0.0
        else:
            self.x = initial_state

        self.P = np.eye(self.state_dim) * 0.1


# =============================================================================
# PARTICLE FILTER - Multi-Modal Distribution Handling
# =============================================================================


class ParticleFilter:
    """
    Particle filter for non-Gaussian noise and multi-modal distributions.

    Useful when:
    - Sensor noise is non-Gaussian (e.g., outliers, skewed)
    - Multiple possible states (ambiguous detections)
    - Need to handle severe nonlinearities
    """

    def __init__(
        self,
        dimensions: int = 2,
        num_particles: int = 100,
        process_noise_std: float = 0.02,
    ):
        """
        Args:
            dimensions: State space dimensions
            num_particles: Number of particles (more = better, slower)
            process_noise_std: Motion model uncertainty
        """
        self.dims = dimensions
        self.num_particles = num_particles
        self.process_noise_std = process_noise_std

        # Particles: [num_particles, dims]
        self.particles = np.random.randn(num_particles, dimensions) * 0.1

        # Particle weights (initially uniform)
        self.weights = np.ones(num_particles) / num_particles

        # Effective sample size threshold for resampling
        self.resample_threshold = num_particles / 2

    def predict(self, velocity: np.ndarray, dt: float = 1.0):
        """
        Propagate particles forward using motion model.

        Args:
            velocity: Expected velocity [vx, vy, (vz)]
            dt: Time step
        """
        # Add motion
        self.particles += velocity * dt

        # Add process noise (random walk)
        self.particles += (
            np.random.randn(self.num_particles, self.dims) * self.process_noise_std
        )

        # Toroidal boundary wrapping
        self.particles = np.mod(self.particles, 1.0)

    def update(self, measurement: np.ndarray, measurement_noise_std: float = 0.01):
        """
        Update particle weights based on measurement likelihood.

        Args:
            measurement: Sensor reading [x, y, (z)]
            measurement_noise_std: Sensor noise standard deviation
        """
        # Calculate likelihood for each particle
        # Using Gaussian likelihood: p(z|x) ~ exp(-||z - x||^2 / (2*sigma^2))
        distances = np.linalg.norm(self.particles - measurement, axis=1)
        likelihoods = np.exp(-(distances**2) / (2 * measurement_noise_std**2))

        # Update weights: w = w * p(z|x)
        self.weights *= likelihoods

        # Normalize weights
        weight_sum = np.sum(self.weights)
        if weight_sum > 0:
            self.weights /= weight_sum
        else:
            # All weights zero (measurement very far from all particles)
            # Reset to uniform
            self.weights = np.ones(self.num_particles) / self.num_particles

        # Check if resampling needed
        n_eff = 1.0 / np.sum(self.weights**2)  # Effective sample size
        if n_eff < self.resample_threshold:
            self._resample()

    def _resample(self):
        """
        Low variance resampling (Thrun et al. 2005).

        Prevents particle depletion while maintaining diversity.
        """
        # Cumulative sum of weights
        cumsum = np.cumsum(self.weights)

        # Generate starting point
        u = np.random.uniform(0, 1.0 / self.num_particles)

        # Low variance sampling
        new_particles = []
        for i in range(self.num_particles):
            u_i = u + i / self.num_particles
            idx = np.searchsorted(cumsum, u_i)
            new_particles.append(self.particles[idx].copy())

        self.particles = np.array(new_particles)

        # Reset weights to uniform
        self.weights = np.ones(self.num_particles) / self.num_particles

    def estimate(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get current state estimate.

        Returns:
            mean: Weighted mean position
            covariance: Weighted covariance matrix
        """
        # Weighted mean
        mean = np.average(self.particles, weights=self.weights, axis=0)

        # Weighted covariance
        diff = self.particles - mean
        cov = np.cov(diff.T, aweights=self.weights)

        return mean, cov

    def reset(self, initial_position: np.ndarray, spread: float = 0.05):
        """Reset particles around initial position."""
        self.particles = (
            initial_position + np.random.randn(self.num_particles, self.dims) * spread
        )
        self.particles = np.mod(self.particles, 1.0)
        self.weights = np.ones(self.num_particles) / self.num_particles


# =============================================================================
# CONSENSUS FILTER - Distributed Multi-Agent Fusion
# =============================================================================


class ConsensusKalmanFilter:
    """
    Distributed Kalman Filter using consensus algorithms.

    Each node maintains local estimate and shares with neighbors
    to reach consensus on global state.

    Based on: Olfati-Saber (2007) "Distributed Kalman Filtering for
    Sensor Networks"
    """

    def __init__(self, node_id: int, dimensions: int = 2, consensus_gain: float = 0.5):
        """
        Args:
            node_id: This node's ID
            dimensions: State dimensions
            consensus_gain: Consensus convergence rate (0-1)
        """
        self.node_id = node_id
        self.dims = dimensions
        self.gamma = consensus_gain

        # Local Kalman filter
        self.kf = KalmanFilter(dimensions)

        # Consensus state (shared with neighbors)
        self.consensus_state = np.zeros(dimensions)

        # Neighbor information buffer
        self.neighbor_states: Dict[int, np.ndarray] = {}

    def predict(self):
        """Prediction step (local)."""
        return self.kf.predict()

    def update(self, measurement: np.ndarray):
        """Update with local measurement."""
        return self.kf.update(measurement)

    def receive_neighbor_state(self, neighbor_id: int, state: np.ndarray):
        """
        Receive state estimate from neighbor.

        Args:
            neighbor_id: Neighbor's ID
            state: Neighbor's state estimate
        """
        self.neighbor_states[neighbor_id] = state

    def consensus_step(self) -> np.ndarray:
        """
        Perform consensus iteration to fuse neighbor information.

        Returns:
            Updated consensus state
        """
        if not self.neighbor_states:
            # No neighbors, use local estimate
            self.consensus_state = self.kf.x[: self.dims]
            return self.consensus_state

        # Consensus update: x_i = x_i + gamma * sum(x_j - x_i)
        local_state = self.kf.x[: self.dims]

        # Average neighbor states
        neighbor_mean = np.mean(list(self.neighbor_states.values()), axis=0)

        # Consensus iteration
        self.consensus_state = local_state + self.gamma * (neighbor_mean - local_state)

        return self.consensus_state

    def get_fused_estimate(self) -> np.ndarray:
        """Get final fused estimate after consensus."""
        return self.consensus_state

    def clear_neighbor_buffer(self):
        """Clear neighbor states (call after each consensus round)."""
        self.neighbor_states.clear()


# =============================================================================
# OUTLIER DETECTOR - RANSAC + Mahalanobis Distance
# =============================================================================


class OutlierDetector:
    """
    Robust outlier detection for sensor readings.

    Combines:
    - Mahalanobis distance for statistical outliers
    - RANSAC for geometric outliers
    - Temporal consistency checks
    """

    def __init__(
        self,
        history_length: int = 10,
        mahalanobis_threshold: float = 3.0,
        temporal_threshold: float = 0.5,
    ):
        """
        Args:
            history_length: How many past measurements to track
            mahalanobis_threshold: Outlier threshold (standard deviations)
            temporal_threshold: Max allowed position jump
        """
        self.history = deque(maxlen=history_length)
        self.mahal_thresh = mahalanobis_threshold
        self.temporal_thresh = temporal_threshold

        self.num_outliers_detected = 0

    def is_outlier(
        self, measurement: np.ndarray, predicted: Optional[np.ndarray] = None
    ) -> bool:
        """
        Check if measurement is an outlier.

        Args:
            measurement: Current sensor reading
            predicted: Predicted value (if available)

        Returns:
            True if outlier detected
        """
        if len(self.history) < 3:
            # Not enough history, accept measurement
            self.history.append(measurement)
            return False

        # 1. Temporal consistency check
        if predicted is not None:
            temporal_error = np.linalg.norm(measurement - predicted)
            if temporal_error > self.temporal_thresh:
                self.num_outliers_detected += 1
                return True

        # 2. Statistical outlier check (Mahalanobis distance)
        if len(self.history) >= 5:
            history_array = np.array(list(self.history))
            mean = np.mean(history_array, axis=0)
            cov = np.cov(history_array.T)

            try:
                # Regularize covariance for numerical stability
                cov += np.eye(len(cov)) * 1e-6

                dist = mahalanobis(measurement, mean, np.linalg.inv(cov))

                if dist > self.mahal_thresh:
                    self.num_outliers_detected += 1
                    return True
            except (np.linalg.LinAlgError, ValueError):
                # Singular covariance, skip check
                pass

        # 3. RANSAC-style check (for neighbor detections)
        # Check if measurement is consistent with linear motion model
        if len(self.history) >= 4:
            # Fit linear trajectory to recent history
            recent = np.array(list(self.history)[-4:])

            # Simple linear fit: x(t) = x0 + v*t
            times = np.arange(len(recent))
            velocity = (recent[-1] - recent[0]) / (len(recent) - 1)

            # Predict next position
            predicted_pos = recent[-1] + velocity

            # Check if measurement is close to prediction
            prediction_error = np.linalg.norm(measurement - predicted_pos)
            if prediction_error > self.temporal_thresh:
                self.num_outliers_detected += 1
                return True

        # Passed all checks, not an outlier
        self.history.append(measurement)
        return False

    def get_statistics(self) -> Dict:
        """Get outlier detection statistics."""
        return {
            "total_outliers": self.num_outliers_detected,
            "history_length": len(self.history),
        }


# =============================================================================
# TEMPORAL SMOOTHER - Savitzky-Golay Filter
# =============================================================================


class TemporalSmoother:
    """
    Smooth noisy signals using Savitzky-Golay filter.

    Better than moving average because it preserves peaks and trends
    while removing high-frequency noise.
    """

    def __init__(
        self, window_length: int = 11, poly_order: int = 3, dimensions: int = 2
    ):
        """
        Args:
            window_length: Filter window size (must be odd)
            poly_order: Polynomial order for fitting
            dimensions: Data dimensions
        """
        if window_length % 2 == 0:
            window_length += 1  # Must be odd

        self.window_length = window_length
        self.poly_order = poly_order
        self.dims = dimensions

        # Circular buffer for history
        self.buffer = deque(maxlen=window_length * 2)

    def add_sample(self, sample: np.ndarray):
        """Add new sample to buffer."""
        self.buffer.append(sample.copy())

    def get_smoothed(self) -> Optional[np.ndarray]:
        """
        Get smoothed value at current time.

        Returns:
            Smoothed value, or None if insufficient data
        """
        if len(self.buffer) < self.window_length:
            # Not enough data, return latest value
            return self.buffer[-1] if self.buffer else None

        # Extract recent history
        recent = np.array(list(self.buffer)[-self.window_length :])

        # Apply Savitzky-Golay filter per dimension
        smoothed = np.zeros(self.dims)
        for d in range(self.dims):
            smoothed[d] = savgol_filter(
                recent[:, d], self.window_length, self.poly_order
            )[-1]  # Get last (current) value

        return smoothed


# =============================================================================
# ADAPTIVE NOISE ESTIMATOR
# =============================================================================


class AdaptiveNoiseEstimator:
    """
    Estimates sensor noise covariance online and adapts filter parameters.

    Uses innovation sequence to detect changes in noise characteristics.
    """

    def __init__(self, dimensions: int = 2, window_size: int = 50):
        """
        Args:
            dimensions: State dimensions
            window_size: Window for noise estimation
        """
        self.dims = dimensions
        self.window_size = window_size

        # Innovation buffer (prediction errors)
        self.innovations = deque(maxlen=window_size)

        # Estimated noise covariance
        self.R_estimated = np.eye(dimensions) * 0.01

    def add_innovation(self, innovation: np.ndarray):
        """Add innovation (prediction error) to buffer."""
        self.innovations.append(innovation.copy())

    def update_noise_estimate(self) -> np.ndarray:
        """
        Update noise covariance estimate based on innovations.

        Returns:
            Estimated measurement noise covariance matrix
        """
        if len(self.innovations) < 10:
            return self.R_estimated

        # Calculate sample covariance of innovations
        innov_array = np.array(list(self.innovations))
        self.R_estimated = np.cov(innov_array.T)

        # Regularize to prevent singular matrix
        self.R_estimated += np.eye(self.dims) * 1e-6

        return self.R_estimated

    def get_noise_level(self) -> float:
        """Get scalar noise level metric."""
        return np.trace(self.R_estimated) / self.dims


# =============================================================================
# INTEGRATED SENSOR PROCESSOR - Main Interface for FLOWRRA
# =============================================================================


class SensorProcessor:
    """
    Main sensor processing pipeline for FLOWRRA.

    Combines all filtering techniques into a unified interface.
    Automatically selects best method based on data quality.
    """

    def __init__(
        self,
        node_id: int,
        dimensions: int = 2,
        use_consensus: bool = True,
        filter_mode: str = "auto",  # "kalman", "particle", "auto"
    ):
        """
        Args:
            node_id: Node identifier
            dimensions: Spatial dimensions
            use_consensus: Enable distributed consensus fusion
            filter_mode: Filtering strategy
        """
        self.node_id = node_id
        self.dims = dimensions
        self.filter_mode = filter_mode

        # Initialize filters
        self.kalman = KalmanFilter(dimensions)
        self.particle = ParticleFilter(dimensions, num_particles=50)

        # Consensus (if multi-agent)
        if use_consensus:
            self.consensus = ConsensusKalmanFilter(node_id, dimensions)
        else:
            self.consensus = None

        # Outlier detection
        self.outlier_detector = OutlierDetector()

        # Temporal smoothing
        self.smoother = TemporalSmoother(window_length=7, dimensions=dimensions)

        # Adaptive noise estimation
        self.noise_estimator = AdaptiveNoiseEstimator(dimensions)

        # Current best estimate
        self.current_estimate = np.zeros(dimensions)
        self.current_velocity = np.zeros(dimensions)

        # Performance tracking
        self.total_measurements = 0
        self.outliers_rejected = 0

    def process_measurement(
        self, raw_measurement: np.ndarray, measurement_confidence: float = 1.0
    ) -> np.ndarray:
        """
        Process raw sensor measurement through full pipeline.

        Args:
            raw_measurement: Noisy sensor reading [x, y, (z)]
            measurement_confidence: Confidence score (0-1)

        Returns:
            Filtered position estimate
        """
        self.total_measurements += 1

        # 1. OUTLIER DETECTION
        predicted = self.kalman.predict()
        is_outlier = self.outlier_detector.is_outlier(raw_measurement, predicted)

        if is_outlier:
            self.outliers_rejected += 1
            # Use prediction instead of corrupted measurement
            self.current_estimate = predicted
            return self.current_estimate

        # 2. SELECT FILTER BASED ON NOISE LEVEL
        if self.filter_mode == "auto":
            noise_level = self.noise_estimator.get_noise_level()
            use_particle = noise_level > 0.05  # High noise → particle filter
        elif self.filter_mode == "particle":
            use_particle = True
        else:
            use_particle = False

        # 3. APPLY FILTER
        if use_particle:
            # Particle filter for non-Gaussian/high noise
            self.particle.predict(self.current_velocity)
            self.particle.update(raw_measurement, measurement_noise_std=0.02)
            filtered_pos, _ = self.particle.estimate()
        else:
            # Kalman filter for Gaussian noise
            filtered_pos = self.kalman.update(raw_measurement)

        # 4. TEMPORAL SMOOTHING
        self.smoother.add_sample(filtered_pos)
        smoothed_pos = self.smoother.get_smoothed()
        if smoothed_pos is not None:
            filtered_pos = smoothed_pos

        # 5. UPDATE NOISE ESTIMATE (adaptive)
        innovation = raw_measurement - predicted
        self.noise_estimator.add_innovation(innovation)
        self.noise_estimator.update_noise_estimate()

        # 6. CONSENSUS FUSION (if enabled)
        if self.consensus is not None:
            self.consensus.kf.x[: self.dims] = filtered_pos
            # Note: consensus_step() called separately after neighbor exchange

        # Update internal state
        self.current_estimate = filtered_pos
        self.current_velocity = self.kalman.get_velocity()

        return self.current_estimate

    def exchange_with_neighbors(
        self, neighbor_estimates: Dict[int, np.ndarray]
    ) -> np.ndarray:
        """
        Perform consensus fusion with neighbor estimates.

        Args:
            neighbor_estimates: Dict mapping neighbor_id -> position estimate

        Returns:
            Fused consensus estimate
        """
        if self.consensus is None:
            return self.current_estimate

        # Share with consensus filter
        for neighbor_id, estimate in neighbor_estimates.items():
            self.consensus.receive_neighbor_state(neighbor_id, estimate)

        # Perform consensus iteration
        fused_estimate = self.consensus.consensus_step()

        # Clear buffer for next round
        self.consensus.clear_neighbor_buffer()

        self.current_estimate = fused_estimate
        return fused_estimate

    def get_position(self) -> np.ndarray:
        """Get current filtered position estimate."""
        return self.current_estimate

    def get_velocity(self) -> np.ndarray:
        """Get current velocity estimate."""
        return self.current_velocity

    def reset(self, initial_position: np.ndarray):
        """Reset all filters."""
        self.kalman.reset(initial_position)
        self.particle.reset(initial_position)
        self.current_estimate = initial_position
        self.current_velocity = np.zeros(self.dims)

    def get_statistics(self) -> Dict:
        """Get processing statistics."""
        return {
            "node_id": self.node_id,
            "total_measurements": self.total_measurements,
            "outliers_rejected": self.outliers_rejected,
            "outlier_rate": self.outliers_rejected / max(1, self.total_measurements),
            "current_noise_level": self.noise_estimator.get_noise_level(),
            "current_position": self.current_estimate.tolist(),
            "current_velocity": self.current_velocity.tolist(),
        }


# =============================================================================
# INTEGRATION EXAMPLE
# =============================================================================


def integrate_with_flowrra_example():
    """
    Example of integrating SensorProcessor with FLOWRRA Orchestrator.
    """
    from holon.node import NodePositionND

    # Create node with sensor processor
    node = NodePositionND(id=0, pos=np.array([0.5, 0.5]), dimensions=2)

    # Attach sensor processor
    sensor_processor = SensorProcessor(
        node_id=node.id, dimensions=2, use_consensus=True, filter_mode="auto"
    )

    # Simulation loop
    for step in range(100):
        # 1. Get raw (noisy) sensor reading
        raw_gps = node.pos + np.random.normal(0, 0.02, 2)  # Noisy GPS

        # 2. Process through sensor fusion pipeline
        filtered_pos = sensor_processor.process_measurement(raw_gps)

        # 3. Update node with filtered position
        node.pos = filtered_pos

        # 4. (Optional) Consensus with neighbors
        if step % 10 == 0:
            # Simulate neighbor communication
            neighbor_estimates = {1: np.array([0.52, 0.48]), 2: np.array([0.48, 0.52])}
            consensus_pos = sensor_processor.exchange_with_neighbors(neighbor_estimates)
            node.pos = consensus_pos

    # Get statistics
    stats = sensor_processor.get_statistics()
    print(f"Processed {stats['total_measurements']} measurements")
    print(
        f"Rejected {stats['outliers_rejected']} outliers ({stats['outlier_rate'] * 100:.1f}%)"
    )
    print(f"Final noise level: {stats['current_noise_level']:.4f}")


if __name__ == "__main__":
    # Run integration example
    integrate_with_flowrra_example()
