
import pybullet as p
from pybullet_helpers.joint import JointPositions
from pybullet_helpers.geometry import Pose

class FeedingPyBulletUtensil:

    def __init__(self, utensil_urdf_path: str, base_pose: Pose, home_joint_positions: JointPositions, physics_client_id: int) -> None:

        self.physics_client_id = physics_client_id
    
        self.utensil_id = p.loadURDF(
                str(utensil_urdf_path),
                useFixedBase=True,
                physicsClientId=self.physics_client_id,
            )

        p.resetBasePositionAndOrientation(
            self.utensil_id,
            base_pose.position,
            base_pose.orientation,
            physicsClientId=self.physics_client_id,
        )

        self.utensil_joints = []
        for i in range(p.getNumJoints(self.utensil_id)):
            joint_info = p.getJointInfo(self.utensil_id, i)
            if joint_info[2] != 4:
                self.utensil_joints.append(i)

        self.set_joints(home_joint_positions)

    def set_joints(self, joint_positions: JointPositions) -> None:

        joint_velocities = [0] * len(joint_positions)
        for joint_id, joint_pos, joint_vel in zip(
            self.utensil_joints, joint_positions, joint_velocities, strict=True
        ):
            p.resetJointState(
                self.utensil_id,
                joint_id,
                targetValue=joint_pos,
                targetVelocity=joint_vel,
                physicsClientId=self.physics_client_id,
            )

    def get_joint_positions(self) -> JointPositions:

        joint_positions = []
        for joint_id in self.utensil_joints:
            joint_state = p.getJointState(self.utensil_id, joint_id, physicsClientId=self.physics_client_id)
            joint_positions.append(joint_state[0])

        return joint_positions
    