import warnings
import gymnasium as gym
import numpy as np
import pybullet as p
from urdfenvs.robots.generic_urdf.generic_diff_drive_robot import GenericDiffDriveRobot
from urdfenvs.urdf_common.urdf_env import UrdfEnv


def run_albert(n_steps=1000, render=False, goal=True, obstacles=True):
    verbose = True
    robots = [
        GenericDiffDriveRobot(
            urdf="albert.urdf",
            mode="vel",
            actuated_wheels=["wheel_right_joint", "wheel_left_joint"],
            castor_wheels=["rotacastor_right_joint", "rotacastor_left_joint"],
            wheel_radius = 0.08,
            wheel_distance = 0.494,
            spawn_rotation = 0,
            facing_direction = '-y',
        ),
    ]
    env: UrdfEnv = UrdfEnv(
        dt=0.01, robots=robots, render=render
    )
    action = np.zeros(env.n())
    action[0] = 0.2
    action[1] = 0.0
    action[5] = -0.1
    ob = env.reset(
        pos=np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, -1.5, 0.0, 1.8, 0.5])
    )
    print(f"Initial observation : {ob}")
    
    robot =  env._robots[0]
    ee_link_name = "mmrobot_hand"
        # FILTER OUT GRIPPER JOINTS - only include actual arm joints
    arm_idxs = []
    for idx in range(2, robot.n() - 1):  # exclude base (0,1) and last gripper joint
        joint_name = robot._joint_names[idx]
        print(f"Joint index {idx}: {joint_name}")
        # Skip gripper/finger joints
        if 'finger' in joint_name.lower() or 'gripper' in joint_name.lower():
            if verbose:
                print(f"Skipping gripper joint at index {idx}: {joint_name}")
            continue
        arm_idxs.append(idx)
    
    if verbose:
        print(f"Arm joint indices (action space): {arm_idxs}")
        print(f"Arm joint names: {[robot._joint_names[i] for i in arm_idxs]}")

    def _end_link_index():
        if ee_link_name:
            try:
                for j in range(p.getNumJoints(robot._robot)):
                    link_name = p.getJointInfo(robot._robot, j)[12].decode("UTF-8")
                    if link_name == ee_link_name:
                        return j
            except Exception:
                pass
        try:
            # Use last arm joint (non-gripper)
            return robot._robot_joints[arm_idxs[-1]]
        except Exception:
            return None

    end_link_index = _end_link_index()
    
    if verbose and end_link_index is not None:
        print(f"End-effector link index: {end_link_index}")
        link_info = p.getJointInfo(robot._robot, end_link_index)
        print(f"EE link name: {link_info[12].decode('UTF-8')}")

    # Read current full joint positions/velocities using PyBullet joint indices
    arm_pb_idxs = [robot._robot_joints[i] for i in arm_idxs]
    print("Reading arm joint states using PyBullet joint indices:", arm_pb_idxs)
    arm_states = p.getJointStates(robot._robot, arm_pb_idxs)
    print(f"Initial arm joint states from PyBullet API: {arm_states}")
    
    arm_idxs2 = robot._robot_joints[2:9]  # Assuming base=3, arm=7
    print("Reading arm joint states using indices:", arm_idxs2)
    for i, joint_idx in enumerate(arm_idxs2):
        p.getJointState(robot._robot, joint_idx)
    # ee_target = np.asarray(arm_target).reshape(3)
    
        
    # 3. Get joint limits
    lower_limits = []
    upper_limits = []
    joint_ranges = []
    rest_poses = []
    
    for jid in arm_pb_idxs:
        info = p.getJointInfo(env._robots[0]._robot, jid)
        lo = info[8]
        hi = info[9]
        
        lower_limits.append(lo)
        upper_limits.append(hi)
        joint_ranges.append(hi - lo)
        print(f"Joint {jid} limits: [{lo}, {hi}]")
    
    history = []
    for _ in range(n_steps):
        observation, reward, done, truncated, info = env.step(action)
        arm_states = p.getJointStates(robot._robot, arm_pb_idxs)
        # print(robot._joint_names)
        pos_obs = observation['robot_0']["joint_state"]["position"][3:9]
        # print(f"Arm joint positions from observation: {pos_obs}")
        vel_obs = observation['robot_0']["joint_state"]["velocity"][3:9]
        #print(f"Arm joint velocities from observation: {vel_obs}")
        #print("="*40)
        q_arm = np.array([st[0] for st in arm_states])
        qd_arm = np.array([st[1] for st in arm_states])
        #print(f"Arm joint positions from pybullet api: {q_arm}")
        #print(f"Arm joint velocities from pybullet api: {qd_arm}")
        #print("="*60)
        print("="*60)
        history.append(ob)
    env.close()
    return history


if __name__ == "__main__":
    show_warnings = False
    warning_flag = "default" if show_warnings else "ignore"
    with warnings.catch_warnings():
        warnings.filterwarnings(warning_flag)
        run_albert(render=True)
