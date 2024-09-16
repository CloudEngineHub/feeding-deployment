import numpy as np
import math
import pinocchio as pin
from scipy.spatial.transform import Rotation as R
import os
import time

class Arm:

    def __init__(self):
        self.DAMPING_FACTOR = 0.01
        self.POLICY_CONTROL_PERIOD = 0.1
        self.ALPHA = 0.01
        self.DT = 0.001

        self.K_r = np.diag([0.4, 0.4, 0.4, 0.4, 0.2, 0.2])
        self.K_l = np.diag([160.0, 160.0, 160.0, 160.0, 100.0, 100.0])
        self.K_lp = np.diag([10.0, 10.0, 10.0, 10.0, 7.5, 7.5])
        self.K_r_inv = np.linalg.inv(self.K_r)
        self.K_r_K_l = self.K_r @ self.K_l
    
        self.K_T_p = np.diag([100.0, 100.0, 100.0, 400.0, 400.0, 400.0])
        self.K_T_d = np.diag([10, 10, 10, 40, 40, 40])

        self.file_path = os.path.dirname(os.path.realpath(__file__))
        self.model = pin.buildModelFromUrdf(
            os.path.join(self.file_path, "hack_gen3_robotiq_2f_85_feeding.urdf")
        )
        self.data = self.model.createData()
        self.q_pin = np.zeros(self.model.nq)
        self.tool_frame_id = self.model.getFrameId("fork_tip")
        self.n_compliant_dofs = 6

        self.q = np.zeros(self.n_compliant_dofs)
        self.dq = np.zeros(self.n_compliant_dofs)
        self.tau = np.zeros(self.n_compliant_dofs)
        self.q_n = np.zeros(self.n_compliant_dofs)
        self.dq_n = np.zeros(self.n_compliant_dofs)
        self.q_s = np.zeros(self.n_compliant_dofs)
        self.x = np.zeros(7)
        self.dx = np.zeros(7)
        self.x_d = np.zeros(7)
        self.gripper_pos = 0

        pin.framesForwardKinematics(self.model, self.data, self.q_pin)
        tool_pose = self.data.oMf[self.tool_frame_id]
        self.x_d[:3] = tool_pose.translation.copy()
        self.x_d[3:] = R.from_matrix(tool_pose.rotation).as_quat()

    def get_fk(self, q):
        # Pinocchio joint configuration
        q_pin = np.array(
            [
                math.cos(q[0]),
                math.sin(q[0]),
                q[1],
                math.cos(q[2]),
                math.sin(q[2]),
                q[3],
                math.cos(q[4]),
                math.sin(q[4]),
                math.cos(q[5]),
                math.sin(q[5]),
            ]
        )
        pin.computeJointJacobians(self.model, self.data, q_pin)
        pin.framesForwardKinematics(self.model, self.data, q_pin)
        tool_pose = self.data.oMf[self.tool_frame_id]

        # Copy the rotation matrix
        rotation_matrix = tool_pose.rotation.copy()

        # # Make first and second columns negative to account for axis convention
        # rotation_matrix[:, 0] = -rotation_matrix[:, 0]
        # rotation_matrix[:, 1] = -rotation_matrix[:, 1]

        # Construct pose
        pose = np.zeros(7)
        pose[:3] = tool_pose.translation.copy()
        pose[3:] = R.from_matrix(rotation_matrix).as_quat()
        
        J = pin.getFrameJacobian(
            self.model, self.data, self.tool_frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
        )
        
        return pose, J
    
    def gravity(self):
        return pin.computeGeneralizedGravity(self.model, self.data, self.q_pin)

    def original_function(self, q, tau_s_f, dq_s):

        g = self.gravity()

        x_n, J_n = self.get_fk(q)

        pos_error = x_n[:3] - self.x_d[:3]

        # Convert to Rotation objects
        R_n = R.from_quat(x_n[3:])
        R_d = R.from_quat(self.x_d[3:])

        # Adjust quaternions to be on the same hemisphere
        if np.dot(R_d.as_quat(), R_n.as_quat()) < 0.0:
            R_n = R.from_quat(-R_n.as_quat())

        # Compute error rotation
        error_rotation = R_n.inv() * R_d

        # Convert error rotation to quaternion
        error_quat = error_rotation.as_quat()

        # Extract vector part
        orient_error_vector = error_quat[:3]

        # Get rotation matrix of nominal pose
        R_n_matrix = R_n.as_matrix()

        # Compute orientation error
        orient_error = -R_n_matrix @ orient_error_vector

        # Assemble error
        error = np.zeros(6)
        error[:3] = pos_error
        error[3:] = orient_error 

        damping_lambda = self.DAMPING_FACTOR * np.eye(self.n_compliant_dofs)
        J_n_damped = np.linalg.inv(J_n.T @ J_n + damping_lambda) @ J_n.T

        tau_task = J_n_damped @ (-self.K_T_p @ error - self.K_T_d @ (J_n @ self.dq_n)) + g

        # Nominal motor plant
        ddq_n = self.K_r_inv @ (tau_task - tau_s_f)
        self.dq_n += ddq_n * self.DT
        self.q_n += self.dq_n * self.DT

        # Nominal friction
        tau_f = self.K_r_K_l @ ((self.dq_n - dq_s) + self.K_lp @ (self.q_n - self.q_s))


def _main():
    rng = np.random.default_rng(0)
    num_samples = 10000
    arm = Arm()
    lower_limits = [arm.model.lowerPositionLimit[i] for i in arm.model.idx_qs]
    upper_limits = [arm.model.upperPositionLimit[i] for i in arm.model.idx_qs]
    tau_s_f = np.zeros(6)
    dq_s = np.zeros(6)
    durations = []
    for _ in range(num_samples):
        q = rng.uniform(lower_limits, upper_limits)
        start_time = time.perf_counter()
        arm.original_function(q, tau_s_f, dq_s)
        duration = time.perf_counter() - start_time
        durations.append(duration)
    ms_durations = np.array(durations) * 1000
    print(f"Results over {num_samples} samples")
    print(f"Mean duration: {np.mean(ms_durations)} ms")
    print(f"Max duration: {np.max(ms_durations)} ms")
    print(f"Percentage over 1ms: {100 * np.sum(ms_durations > 1) / num_samples}%")
    print(f"Percentage over 4ms: {100 * np.sum(ms_durations > 4) / num_samples}%")
    import ipdb; ipdb.set_trace()

if __name__ == "__main__":
    _main()
