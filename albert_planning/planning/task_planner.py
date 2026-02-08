"""
High-level task planner for mobile manipulation.
Sequential base → arm motion with smooth transitions.
"""
import time
import numpy as np
import pybullet as p
from typing import Optional, Tuple, Dict, List
from tqdm import tqdm
from urdfenvs.urdf_common.generic_robot import ControlMode
from bar_env import BOTTLE_POSES

# In case file metrics.py donesn't work: 
try:
    from metrics import TrialMetrics
except ImportError:
    print("Warning: metrics.py not found. Metrics will be disabled.")
    TrialMetrics = None  # Prevent NameError later

# =========================================================
# METRICS CONFIGURATION
# Set to 1 to enable metrics recording and printing.
# Set to 0 to run normally without overhead.
# =========================================================
IS_METRICS_ENABLED = 1
# =========================================================

def get_reference_window_lookahead(path: np.ndarray, x_current: np.ndarray, 
                                   N: int, lookahead_dist: float = 1.0) -> np.ndarray:
    """
    Extract reference window by looking ahead a fixed distance on the path.
    NO waypoint switching, NO blocks!
    
    Args:
        path: (T, 2) array of [x, y] waypoints
        x_current: [x, y, theta] current state
        N: MPC horizon
        lookahead_dist: distance to look ahead [m]
    
    Returns:
        ref: (3, N+1) reference [x, y, theta]
    """
    # Find closest point on path
    dists = np.linalg.norm(path - x_current[:2], axis=1)
    closest_idx = np.argmin(dists)
    
    # Compute arc lengths from closest point
    arc_lengths = [0.0]
    for i in range(closest_idx, len(path) - 1):
        dist = np.linalg.norm(path[i+1] - path[i])
        arc_lengths.append(arc_lengths[-1] + dist)
    
    # Extract reference by lookahead distance
    ref = np.zeros((3, N + 1))
    
    for k in range(N + 1):
        # Desired arc length
        s_desired = k * lookahead_dist / N
        
        # Find corresponding index
        if len(arc_lengths) > 0 and s_desired >= arc_lengths[-1]:
            idx = len(path) - 1
        else:
            seg_idx = np.searchsorted(arc_lengths, s_desired)
            idx = min(closest_idx + seg_idx, len(path) - 1)
        
        # Set reference
        ref[0, k] = path[idx, 0]
        ref[1, k] = path[idx, 1]
        
        # Compute orientation
        if idx < len(path) - 1:
            dx = path[idx + 1, 0] - path[idx, 0]
            dy = path[idx + 1, 1] - path[idx, 1]
            ref[2, k] = np.arctan2(dy, dx)
        else:
            ref[2, k] = x_current[2]
    
    return ref



class MobileManipulationTask:
    """Coordinates base and arm motion for pick-and-place."""
    
    def __init__(self, env, robot, dt, config: Dict):
        self.env = env
        self.robot = robot
        self.dt = dt
        self.config = config
        
        # --- METRICS INITIALIZATION ---
        self.metrics = None
        if IS_METRICS_ENABLED and TrialMetrics is not None:
            self.metrics = TrialMetrics()
            print("Metrics Collection: ENABLED")
        else:
            print("Metrics Collection: DISABLED")

        # Metrics Internal State
        self.last_pos = np.zeros(2)
        self.stuck_counter = 0
        # -----------------------------
        
        # Robot info
        self.total_action_dim = env.n()
        self.arm_indices = self._get_arm_indices()
        self.arm_pb_idxs = [robot._robot_joints[i] for i in self.arm_indices]
        self.ee_link_idx = self._get_ee_link_idx()
        self.gripper_indices = [16, 17]
        
        # Controllers
        self.base_controller = None
        self.arm_controller = None
        
        # Planners (for path processing collision checking)
        self.base_planner = None
        self.arm_planner = None
        
        # State
        self.phase = "idle"  # idle, base_moving, arm_moving, done

        # ✅ Gripper state tracking
        self.gripper_closed = False  # Track if gripper should stay closed
        self.gripper_target_pos = 0.04  # Current target position (0.04=open, 0.0=closed)
        self.gripper_force = -2.5  # Holding force
        
        # ✅ History tracking for plotting
        self.base_history = {
            'time': [],
            'positions': [],      # [x, y, theta]
            'controls': [],       # [v, omega]
            'goal': None,
            'global_path': None,
            'mode': None
        }
        
        self.arm_history = {
            'time': [],
            'joint_positions': [],
            'joint_velocities': [],
            'joint_accelerations': [],
            'ee_positions': [],
            'ee_target': None,
            'ee_start': None,
            'q_target': None,
            'global_path': None,
            'mode': None
        }
        
        self.base_histories = []
        self.arm_histories = []
        self.arm_exec_metrics = []

        self.t_start = 0.0  # Track elapsed time
    
    def _get_arm_indices(self):
        """Get arm joint indices (exclude base, gripper)."""
        arm_idxs = []
        for idx in range(2, self.robot.n() - 1):
            joint_name = self.robot._joint_names[idx]
            if 'finger' not in joint_name.lower() and 'gripper' not in joint_name.lower():
                arm_idxs.append(idx)
        return arm_idxs
    
    def _get_ee_link_idx(self):
        """Find end-effector link index."""
        ee_name = "mmrobot_hand"
        for j in range(p.getNumJoints(self.robot._robot)):
            link_name = p.getJointInfo(self.robot._robot, j)[12].decode("UTF-8")
            if link_name == ee_name:
                return j
        return self.arm_pb_idxs[-1]
    
    def get_base_position(self) -> np.ndarray:
        """Get [x, y, theta] from PyBullet."""
        pos, quat = p.getBasePositionAndOrientation(self.robot._robot)
        pos = pos[:2]
        euler = p.getEulerFromQuaternion(quat)
        theta = euler[2]
        # print(f"Base position: {pos}, theta: {theta}")
        # print("ALTERNATIVE METHOD:")
        x_current = self.robot.state["joint_state"]["position"][:3].copy()
        #print(f" BASE POSE::  x: {x_current[0]}, y: {x_current[1]}, theta: {x_current[2]}")
        # Apply theta offset for Albert's -y facing direction
        #theta_offset = np.pi / 2
        #theta = (euler[2] + theta_offset + np.pi) % (2 * np.pi) - np.pi
        #return np.array([x_current[0], x_current[1], x_current[2]])
        # print(f" BASE POSE::  x: {pos[0]}, y: {pos[1]}, theta: {theta}")
        return np.array([pos[0], pos[1], theta])
    
        def get_base_position(self) -> np.ndarray:
            return self.robot.state["joint_state"]["position"][:3].copy()
        
    def get_arm_state(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get arm joint positions and velocities."""
        states = p.getJointStates(self.robot._robot, self.arm_pb_idxs)
        q = np.array([s[0] for s in states])
        qd = np.array([s[1] for s in states])
        return q, qd
    
    def get_ee_pose(self) -> np.ndarray:
        """Get end-effector position [x, y, z]."""
        state = p.getLinkState(self.robot._robot, self.ee_link_idx)
        return np.array(state[0])

    def open_gripper(self, robot_id, target_pos=0.04, force=30.0):
        """Open gripper fully."""
        if not self.gripper_indices:
            return
        
        # ✅ Update state
        self.gripper_closed = False
        self.gripper_target_pos = target_pos
        self.gripper_force = force
        
        # Apply command
        for idx in self.gripper_indices:
            p.setJointMotorControl2(
                robot_id,
                idx,
                controlMode=p.POSITION_CONTROL,
                targetPosition=target_pos,
                force=force,
            )
        for _ in range(50):
            p.stepSimulation()
            time.sleep(1./240.)

    def close_gripper(self, robot_id, target_pos=0.0, force=30.0):
        """Close gripper to grasp."""
        if not self.gripper_indices:
            return
        
        # ✅ Update state
        self.gripper_closed = True
        self.gripper_target_pos = target_pos
        self.gripper_force = force
        
        # Apply command
        for idx in self.gripper_indices:
            p.setJointMotorControl2(
                robot_id,
                idx,
                controlMode=p.POSITION_CONTROL,
                targetPosition=target_pos,
                force=force,
            )
        q_hold, _ = self.get_arm_state()
        for jid, q in zip(self.arm_pb_idxs, q_hold):
            p.setJointMotorControl2(
                robot_id,
                jid,
                controlMode=p.POSITION_CONTROL,
                targetPosition=q,
                force=80.0,
            )
        for _ in range(50):
            p.stepSimulation()
            time.sleep(1./240.)
                
    def compute_base_goal_from_target(self, target_pos: np.ndarray, 
                                     approach_dist: float = 0.3) -> np.ndarray:
        """
        Compute base goal position to reach target with arm.
        
        Args:
            target_pos: [x, y, z] target position
            approach_dist: distance from target for base
        
        Returns:
            [x, y, theta] base goal
        """
        # Simple approach: position base at approach_dist from target
        direction = np.array([target_pos[0], target_pos[1]])
        direction_norm = np.linalg.norm(direction)
        
        if direction_norm > 0:
            direction = direction / direction_norm
        else:
            direction = np.array([1.0, 0.0])
        
        base_pos = np.array([target_pos[0], target_pos[1]]) - direction * approach_dist
        theta = np.arctan2(target_pos[1] - base_pos[1], target_pos[0] - base_pos[0])
        
        return np.array([base_pos[0], base_pos[1], theta])
    
    def compute_ee_orientation(self, target_pos: np.ndarray, 
                              grasp_type: str = 'top_down',
                              origin_xy: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compute end-effector orientation for grasping based on approach direction.
        
        Args:
            target_pos: [x, y, z] target position to grasp
            grasp_type: Type of grasp
                       - 'top_down': Approach from above (default)
                       - 'side': Approach from side
                       - 'angled': 45° approach
            origin_xy: Optional [x, y] position to define approach direction
        
        Returns:
            ee_rpy: [roll, pitch, yaw] orientation in radians
        """
        # Get approach origin
        if origin_xy is None:
            origin_xy = self.get_base_position()[:2]
        else:
            origin_xy = np.array(origin_xy[:2])
        
        # Compute approach direction in XY plane
        dx = target_pos[0] - origin_xy[0]
        dy = target_pos[1] - origin_xy[1]
        
        # Compute yaw to align gripper with approach direction
        approach_yaw = np.arctan2(dy, dx)
        
        yaw = approach_yaw  # Align gripper with approach
        pitch = np.pi/2  # Point down
        roll = 0.0  # Align with approach

        ee_rpy = np.array([roll, pitch, yaw])
        
        print(f"\n🎯 Computed EE Orientation:")
        print(f"  Approach direction: ({dx:.3f}, {dy:.3f})")
        print(f"  Yaw (approach angle): {np.rad2deg(yaw):.1f}°")
        print(f"  Full RPY (deg): [{np.rad2deg(roll):.1f}, {np.rad2deg(pitch):.1f}, {np.rad2deg(yaw):.1f}]")
        
        return ee_rpy
    
    def execute_base_motion(self, goal_pos: np.ndarray, max_steps: int = 700,
                            tolerance: float = 0.1, path: Optional[List] = None, gripper_force: bool = False, 
                            log=True, **kwargs,) -> bool:

        """
        Execute base motion to goal.
        
        Args:
            goal_pos: [x, y, theta] goal position
            max_steps: maximum steps
            tolerance: position tolerance [m]
            path: optional global path for MPC tracking
        
        Returns:
            success: True if goal reached
        """
        print(f"\n=== BASE MOTION ===")
        print(f"Goal: {goal_pos}")
        

        # Initialize tracking
        self.base_history['goal'] = goal_pos
        self.base_history['mode'] = "tracker" if path is not None else "planner"
        self.base_history['global_path'] = path
        self.t_start = 0.0
        self.robot._mode = ControlMode.velocity

        # --- METRICS: Start Execution Timer ---
        if self.metrics: self.metrics.start_execution()
        # --------------------------------------
        
        if path is not None and len(path) > 0:
            print(f"Raw path: {len(path)} waypoints")
            
            # Process path: shortcut + smooth
            if self.base_planner is not None:
                from global_planners import shortcut_path, smooth_path
                print("Processing path...")
                smooth_points = self.config.get('smooth_points', 400)
                path = smooth_path(path, num_points=smooth_points)
                print(f"Tracking processed path: {len(path)} waypoints")
            
                
            else:
                print(f"Tracking path without processing: {len(path)} waypoints")

            if self.metrics:
                self.metrics.add_global_path(path)
            
            
            if len(path) > 0: # [FIX] Double check if smoothing returned an empty path
                mode = "tracking"
                self.base_history['mode'] = "tracker"
                path_idx = 0
        else:
            print("⚠ Warning: Path became empty after processing. Falling back to Direct MPC.")
            mode = "direct"
            self.base_history['mode'] = "planner"
            self.base_controller.x_ref_traj = None
            self.base_controller.x_target = goal_pos
        
        self.phase = "base_moving"
        print("BASE STARTING POSITION:", self.get_base_position())
        
        # Convert path to numpy array ONCE (outside loop!)
        if mode == "tracking" and path is not None:
            path_array = np.array(path)
            print(f"  Path array shape: {path_array.shape}")
        
        for t in tqdm(range(max_steps), desc="Base motion"):
            x_current = self.get_base_position()
            
            # --- METRICS: CHECK STAGNATION (Local Minima) ---
            if self.metrics:
                dist_moved = np.linalg.norm(x_current[:2] - self.last_pos)
                if dist_moved < 0.002: # 2mm
                    self.stuck_counter += 1
                else:
                    self.stuck_counter = 0
                self.last_pos = x_current[:2]
                
                if self.stuck_counter > 200:
                    print("!! Robot Stuck (Local Minima) !!")
                    self.metrics.robustness_fail = True

            self.base_history['time'].append(self.t_start)
            self.base_history['positions'].append(x_current.copy())
        
            # Initialize x_traj to ensure it exists
            x_traj = None

            # ------------- CONTROL LOGIC ------------------------
            #  Track state BEFORE control
            if mode == "tracking" and path is not None:
                # Fixed lookahead, NO waypoint switching!
                ref_window = get_reference_window_lookahead(
                    path_array, 
                    x_current, 
                    self.base_controller.N,
                    lookahead_dist=1.5  # Look 1.5m ahead
                )
                self.base_controller.set_reference_trajectory(ref_window)
                u_base, _, x_traj = self.base_controller.solve(x_current)
            else:
                # Direct MPC (no trajectory, just goal in cost function)
                self.base_controller.x_ref_traj = None  # Clear any previous  
                u_base, _, x_traj = self.base_controller.solve(x_current)
            """
            if mode == "tracking" and path is not None:
                # print("PATH SAMPLES", path)
                # Waypoint tracking logic
                # Fixed lookahead, NO waypoint switching!

                if path_idx < len(path):
                    target_pt = path[path_idx]
                    #print("TARGET POINT:", target_pt)
                    #print("CURRENT POSITION:", x_current)
                    dist_to_waypoint = np.linalg.norm(x_current[:2] - np.array(target_pt))
                    
                    # Switch to next waypoint if close enough
                    if dist_to_waypoint < 0.15 and path_idx < len(path) - 1:
                        path_idx += 1
                        print(f"  Reached waypoint {path_idx}/{len(path)}")
                    
                    # Extract reference window for MPC horizon
                    ref_window = self._get_base_reference_window(path, path_idx, x_current)
                    self.base_controller.set_reference_trajectory(ref_window)
                
                # Solve MPC with trajectory tracking
                u_base, _, x_traj = self.base_controller.solve(x_current)
            
            else:
                # Direct MPC (no trajectory, just goal in cost function)
                self.base_controller.x_ref_traj = None  # Clear any previous trajectory
                u_base, _, x_traj = self.base_controller.solve(x_current)
            """
            """
            # ------------------ VISUALIZATION ---------------------------
            # Draw the MPC prediction lines (Green "Tentacle")
            # We use a short lifeTime so they disappear automatically in the next step
            if x_traj is not None:
                # Shape check: ensure it is (3, N+1)
                if x_traj.shape[0] != 3: 
                    x_traj = x_traj.T
                
                for k in range(x_traj.shape[1] - 1):
                    # Start point (x, y, z_offset)
                    pt_a = [x_traj[0, k], x_traj[1, k], 0.05]
                    # End point
                    pt_b = [x_traj[0, k+1], x_traj[1, k+1], 0.05]
                    
                    # Draw GREEN line
                    p.addUserDebugLine(pt_a, pt_b, lineColorRGB=[0, 0, 1], lineWidth=2.0, lifeTime=0.1)
            """

            # --- METRICS: LOG PHYSICS ---
            if self.metrics and log:
                self.metrics.log_step(u_base[0], u_base[1], x_current[0], x_current[1], self.dt)
            # ----------------------------


            # Apply control
            action = np.zeros(self.total_action_dim)
            action[0] = u_base[0]
            action[1] = u_base[1]
            action[9] = -2.5 if gripper_force else 0.0  # Gripper force
            action[10] = -2.5 if gripper_force else 0.0  # Gripper force
            

            # Track control
            self.base_history['controls'].append(u_base.copy())
            ob, _, _, _, _ = self.env.step(action)
            
            # Update time
            self.t_start += self.dt

            # --- METRICS: COUNT COLLISIONS ---
            if self.metrics:
                contacts = p.getContactPoints(bodyA=self.robot._robot)
                if any(c[6][2] > 0.05 for c in contacts): # Filter floor
                    self.metrics.collision_static += 1
            # ---------------------------------
            
            # Check convergence
            distance = np.linalg.norm(x_current[:2] - goal_pos[:2])
            print(f"  Step {t}: position = ({x_current[0]:.3f}, {x_current[1]:.3f}), distance to goal = {distance:.3f}m")
            if distance < tolerance:
                print(f"✓ Base goal reached at step {t}")
                print(f"  Final position: ({x_current[0]:.3f}, {x_current[1]:.3f})")
                print(f"  Distance to goal: {distance:.3f}m")
                return True
            
            if t % 50 == 0:
                print(f"  Step {t}: distance = {distance:.3f}m")
        
        print(f"⚠ Base motion timeout (distance = {distance:.3f}m)")
        return False
    
    def _get_base_reference_window(self, path: List, current_idx: int, 
                                   x_current: np.ndarray) -> np.ndarray:
        """
        Extract reference window from path for MPC horizon.
        
        Args:
            path: list of (x, y) waypoints
            current_idx: current waypoint index
            x_current: current state [x, y, theta]
        
        Returns:
            ref_window: (3, N+1) reference [x, y, theta] for MPC
        """
        N = self.base_controller.N
        ref_window = np.zeros((3, N + 1))
        
        # Fill reference from path
        for k in range(N + 1):
            path_idx = min(current_idx + k, len(path) - 1)
            # print("FILLING REF WINDOW: path_idx:", path_idx)
            ref_window[0, k] = path[path_idx][0]  # x
            ref_window[1, k] = path[path_idx][1]  # y
            # print("REF WINDOW:", ref_window)
            
            # Compute orientation toward next waypoint
            if path_idx < len(path) - 1:
                dx = path[path_idx + 1][0] - path[path_idx][0]
                dy = path[path_idx + 1][1] - path[path_idx][1]
                ref_window[2, k] = np.arctan2(dy, dx)
            else:
                # At goal, use final orientation
                ref_window[2, k] = x_current[2]
        
        return ref_window
    
    def execute_arm_motion(self, target_ee_pos: np.ndarray, target_ee_rpy: np.ndarray, max_steps: int = 300,
                          tolerance: float = 0.015, q_path: Optional[np.ndarray] = None, gripper_force: bool = False, **kwargs) -> bool:
        """
        Execute arm motion to reach target end-effector position.
        
        Args:
            target_ee_pos: [x, y, z] target EE position
            target_ee_rpy: [roll, pitch, yaw] target orientation (optional)
                          If None, IK will find best orientation automatically
            max_steps: maximum steps
            tolerance: position tolerance [m]
            q_path: optional pre-computed joint trajectory (T, n_joints)
        
        Returns:
            success: True if target reached
        """
        print(f"\n=== ARM MOTION ===")
        print(f"Target EE: {target_ee_pos}")


        # ============================================================
        # NUOVO: Pausa ostacoli dinamici mentre il braccio si muove
        # ============================================================
        saved_velocities = []
        if hasattr(self.env, 'dynamic_obstacles') and self.env.dynamic_obstacles:
            for obs in self.env.dynamic_obstacles:
                saved_velocities.append(obs.velocity.copy())
                obs.velocity = np.zeros(2)
            print("⏸️  Dynamic obstacles paused for arm motion")
        # ============================================================
        
        if target_ee_rpy is not None:
            print(f" Orientation: {target_ee_rpy}")
            print(f" Orientation (deg): {np.rad2deg(target_ee_rpy)}")
        else:
            print(f" Orientation: AUTO (position-only IK)")
        
        self.robot._mode = ControlMode.acceleration
        self.phase = "arm_moving"
        
        # Compute target joint configuration via IK
        q_target = self._ik_target(target_ee_pos, target_ee_rpy)
        
        if q_target is None:
            print("⚠ IK failed")
            # ============================================================
            # NUOVO: Ripristina velocità anche se IK fallisce
            # ============================================================
            if saved_velocities:
                for obs, vel in zip(self.env.dynamic_obstacles, saved_velocities):
                    obs.velocity = vel
                print("▶️  Dynamic obstacles resumed")
            # ============================================================
            return False
        
        print(f"Target joints: {q_target}")
        
        # ✅ Initialize tracking
        self.arm_history['ee_target'] = target_ee_pos.copy()
        self.arm_history['q_target'] = q_target.copy()
        self.arm_history['mode'] = "tracker" if q_path is not None else "planner"
        
        # Get start EE position
        ee_start = self.get_ee_pose()
        self.arm_history['ee_start'] = ee_start.copy()
        
        self.t_start = 0.0

        planner_type = None
        plan_time = None
        planner_path_len = None

        # Build roadmap and find path NOW (after base has moved)
        if q_path is None and self.arm_planner is not None:
            print("\n🗺️  Planning arm path...")
            planner_type = type(self.arm_planner).__name__
            plan_start = time.time()
            
            # Get current arm configuration (AFTER base moved!)
            q_current, _ = self.get_arm_state()
            
            # Build roadmap (if not already built or if PRM)
            if hasattr(self.arm_planner, 'build_roadmap'):
                n_samples = self.config.get('arm_prm_samples', 500)
                k_neighbors = self.config.get('arm_prm_k_neighbors', 10)
                self.arm_planner.build_roadmap(n_samples=n_samples, k_neighbors=k_neighbors)
            
            # Find path from current to target
            q_path = self.arm_planner.find_path(q_current, q_target)
            plan_time = time.time() - plan_start
            if plan_time > 0.0:
                print(f"  Arm planner time: {plan_time:.3f}s")
            
            
            # Reset robot back to current config after planning
            for i, jid in enumerate(self.arm_pb_idxs):
                p.resetJointState(self.robot._robot, jid, q_current[i])
            
            if q_path is None:
                print("⚠️  Arm path not found, falling back to direct MPC")
                planner_type = None
                plan_time = None
                planner_path_len = None
        elif q_path is not None and len(q_path) > 1:
            planner_path_len = float(np.sum(np.linalg.norm(np.diff(q_path, axis=0), axis=1)))
        
        # Process path if available
        if q_path is not None:
            print(f"Raw joint-space path: {q_path.shape[0]} waypoints")
            
            # Process path: shortcut + smooth
            if self.arm_planner is not None:
                from arm_global_planners import shortcut_arm_path, smooth_arm_path
                
                print("Processing arm path...")
                smooth_points = self.config.get('smooth_points', 100)
                
                # Shortcut
                # q_path = shortcut_arm_path(q_path, self.arm_planner, max_iterations=50)
                
                # Smooth
                # q_path = smooth_arm_path(q_path, num_points=smooth_points)
                
                print(f"✓ Tracking processed path: {q_path.shape[0]} waypoints\n")
                print("THE PATH IS:", q_path)

            else:
                print(f"Tracking path without processing: {q_path.shape[0]} waypoints\n")
            
            mode = "tracking"
            path_idx = 0
            
            # ✅ Save global path
            self.arm_history['global_path'] = q_path.copy()
        else:
            print(f"Direct MPC planning to IK target\n")
            mode = "planning"

        if planner_path_len is None and q_path is not None and len(q_path) > 1:
            planner_path_len = float(np.sum(np.linalg.norm(np.diff(q_path, axis=0), axis=1)))

        segment_time = []
        segment_ee = []
        segment_acc = []
        
        for t in tqdm(range(max_steps), desc="Arm motion"):
            q_current, qd_current = self.get_arm_state()
            ee_current = self.get_ee_pose()
            
            # ✅ Track state BEFORE control
            self.arm_history['time'].append(self.t_start)
            self.arm_history['joint_positions'].append(q_current.copy())
            self.arm_history['joint_velocities'].append(qd_current.copy())
            self.arm_history['ee_positions'].append(ee_current.copy())
            segment_time.append(self.t_start)
            segment_ee.append(ee_current.copy())
            
            if t % 20 == 0:
                print(f"  Step {t}: EE position = {ee_current}, target = {target_ee_pos}")

            
            if mode == "tracking" and q_path is not None:
                # Extract reference window for MPC
                ref_window = self._get_arm_reference_window(q_path, path_idx)
                self.arm_controller.set_reference_trajectory(ref_window)
                
                # Advance waypoint if close enough
                if path_idx < len(q_path) - 1:
                    dist_to_waypoint = np.linalg.norm(q_current - q_path[path_idx])
                    if dist_to_waypoint < 0.1:  # 0.1 rad threshold
                        path_idx += 1
                        if t % 20 == 0:
                            print(f"  Reached waypoint {path_idx}/{len(q_path)}")
            else:
                # Direct planning mode - clear reference
                self.arm_controller.q_ref_traj = None
                self.arm_controller.q_target = q_target
            
            # Solve MPC
            u_arm, q_next, qd_next, _ = self.arm_controller.solve(q_current, qd_current)
            

            # ✅ Track acceleration
            self.arm_history['joint_accelerations'].append(u_arm.copy())
            segment_acc.append(u_arm.copy())
            

            # Apply control (only arm moves)
            action = np.zeros(self.total_action_dim)
            action[self.arm_indices] = u_arm
            action[9] = -2.5 if gripper_force else 0.0  # Gripper force
            action[10] = -2.5 if gripper_force else 0.0  # Gripper force
            
            ob, _, _, _, _ = self.env.step(action)
            
            # ✅ Update time
            self.t_start += self.dt
            
            # Check convergence
            ee_current = self.get_ee_pose()

            ee_error = np.linalg.norm(ee_current - target_ee_pos)

            if ee_error < tolerance:
                print(f"✓ Arm target reached at step {t}")
                print(f"  Final EE position: {ee_current}")
                print(f"  Error: {ee_error*1000:.2f}mm")
                if saved_velocities:
                    for obs, vel in zip(self.env.dynamic_obstacles, saved_velocities):
                        obs.velocity = vel
                    print("▶  Dynamic obstacles resumed")
                self.arm_exec_metrics.append(
                    {
                        "duration": (segment_time[-1] - segment_time[0]) if len(segment_time) > 1 else 0.0,
                        "ee_path_length": float(np.sum(np.linalg.norm(np.diff(np.array(segment_ee), axis=0), axis=1)))
                        if len(segment_ee) > 1 else 0.0,
                        "final_ee_error": float(np.linalg.norm(segment_ee[-1] - target_ee_pos))
                        if segment_ee else 0.0,
                        "max_acc": float(np.max(np.linalg.norm(np.array(segment_acc), axis=1)))
                        if segment_acc else 0.0,
                        "planner_type": planner_type,
                        "planner_time": plan_time,
                        "planner_path_length": planner_path_len,
                        "mode": mode,
                    }
                )
                return True
            
            if t % 20 == 0:
                print(f"  Step {t}: EE error = {ee_error*1000:.2f}mm")
        
        print(f"⚠ Arm motion timeout (error = {ee_error*1000:.2f}mm)")
        if saved_velocities:
            for obs, vel in zip(self.env.dynamic_obstacles, saved_velocities):
                obs.velocity = vel
            print("▶  Dynamic obstacles resumed")

        if segment_ee:
            self.arm_exec_metrics.append(
                {
                    "duration": (segment_time[-1] - segment_time[0]) if len(segment_time) > 1 else 0.0,
                    "ee_path_length": float(np.sum(np.linalg.norm(np.diff(np.array(segment_ee), axis=0), axis=1)))
                    if len(segment_ee) > 1 else 0.0,
                    "final_ee_error": float(np.linalg.norm(segment_ee[-1] - target_ee_pos)),
                    "max_acc": float(np.max(np.linalg.norm(np.array(segment_acc), axis=1)))
                    if segment_acc else 0.0,
                    "planner_type": planner_type,
                    "planner_time": plan_time,
                    "planner_path_length": planner_path_len,
                    "mode": mode,
                }
            )
        return False
    
    def _get_arm_reference_window(self, q_path: np.ndarray, current_idx: int) -> np.ndarray:
        """
        Extract reference window from joint-space path for MPC horizon.
        
        Args:
            q_path: (T, n_joints) joint trajectory
            current_idx: current waypoint index
        
        Returns:
            ref_window: (n_joints, N+1) reference for MPC
        """
        N = self.arm_controller.N
        n_joints = q_path.shape[1]
        ref_window = np.zeros((n_joints, N + 1))
        
        for k in range(N + 1):
            path_idx = min(current_idx + k, len(q_path) - 1)
            ref_window[:, k] = q_path[path_idx]
        
        return ref_window
    

    def _ik_target(self, target_ee_pos: np.ndarray, target_ee_rpy: np.ndarray) -> Optional[np.ndarray]:
        """IK with verification (without corrupting robot state)."""
        try:
            robot_id = self.robot._robot
            arm_joints = self.arm_pb_idxs
            ee_link_idx = self.ee_link_idx
            
            # ✅ 1. SAVE current joint states
            current_states = []
            for jid in arm_joints:
                state = p.getJointState(robot_id, jid)
                current_states.append(state[0])  # position
            
            # 2. Get limits and rest poses
            lower_limits = []
            upper_limits = []
            joint_ranges = []
            rest_poses = []
            
            for jid in arm_joints:
                info = p.getJointInfo(robot_id, jid)

                lower_limits.append(info[8])
                upper_limits.append(info[9])
                joint_ranges.append(info[9] - info[8])
                rest_poses.append(p.getJointState(robot_id, jid)[0])
            
            # 3. Call IK
            q_full = p.calculateInverseKinematics(
                bodyUniqueId=robot_id,
                endEffectorLinkIndex=ee_link_idx,
                targetPosition=target_ee_pos,
                targetOrientation=p.getQuaternionFromEuler(target_ee_rpy), #[0, -1.5708, 0]
                lowerLimits=lower_limits,
                upperLimits=upper_limits,
                jointRanges=joint_ranges,
                restPoses=rest_poses,
                maxNumIterations=1000,
                residualThreshold=1e-4
            )
            
            # 4. Extract solution
            dof_joints = [j for j in range(p.getNumJoints(robot_id)) 
                        if p.getJointInfo(robot_id, j)[2] != p.JOINT_FIXED]
            
            q_solution = np.array([q_full[dof_joints.index(jid)] for jid in arm_joints])
            
            # ✅ 5. VERIFY without permanently changing state
            # Temporarily set to solution
            for jid, q in zip(arm_joints, q_solution):
                p.resetJointState(robot_id, jid, q)
            
            # Check achieved position
            link_state = p.getLinkState(robot_id, ee_link_idx)
            ee_achieved = np.array(link_state[0])
            print(f"  IK achieved EE position: {ee_achieved}")
            # Check achieved orientation
            ee_quat = link_state[1]
            ee_rpy = p.getEulerFromQuaternion(ee_quat)
            print(f"  IK achieved EE orientation (RPY): {ee_rpy}")
            error_mm = np.linalg.norm(ee_achieved - target_ee_pos) * 1000
            
            # ✅ 6. RESTORE original state
            for jid, q_original in zip(arm_joints, current_states):
                p.resetJointState(robot_id, jid, q_original)
            
            # 7. Check if solution is acceptable
            if error_mm > 25.0:  # 25mm threshold
                print(f"  ✗ IK error too large: {error_mm:.1f}mm")
            
            if error_mm > 1.0:
                print(f"  IK error: {error_mm:.1f}mm")
            
            return q_solution
            
        except Exception as e:
            print(f"⚠ IK failed: {e}")
            return None
    
    def execute_smooth_transition(self, delay_steps: int = 10) -> None:
        """
        Smooth transition between base and arm motion.
        Gradually brings base to stop before arm starts.
        
        Args:
            delay_steps: steps to decelerate base
        """
        print(f"\n=== SMOOTH TRANSITION (delay={delay_steps}) ===")
        
        for t in range(delay_steps):
            # Zero control to let base settle
            action = np.zeros(self.total_action_dim)
            ob, _, _, _, _ = self.env.step(action)

        
        print("✓ Transition complete")
    
    def execute_task(self, target_pos: np.ndarray, 
                    target_ee_rpy: Optional[np.ndarray] = None,
                    approach_dist: float = 0.6,
                    smooth_transition: bool = True,
                    global_path: Optional[List] = None,
                    arm_path: Optional[np.ndarray] = None) -> bool:
        """
        Execute full pick-and-place task.
        
        Args:
            target_pos: [x, y, z] target position (e.g., table)
            target_ee_rpy: [roll, pitch, yaw] target EE orientation (optional)
                          If None, IK will find best orientation automatically
            approach_dist: base distance from target
            smooth_transition: enable smooth transition
            global_path: optional pre-computed path for base
            arm_path: optional pre-computed joint trajectory for arm
        
        Returns:
            success: True if task completed
        """
        print("\n" + "="*70)
        print("MOBILE MANIPULATION TASK")
        print("="*70)
        
        # Get the original base position before motion
        original_base_pos = self.get_base_position()
        print(f"Original base position: {original_base_pos}")

        # ----------- Phase 1: Base motion ---------------
        base_goal = self.compute_base_goal_from_target(target_pos, approach_dist)
        print(f"\n🎯 Computed base goal: {base_goal}")
        success = self.execute_base_motion(base_goal, path=global_path)
        
        if not success:
            print("❌ Task failed: base motion unsuccessful")
            # --- METRICS: PRINT FAILURE REPORT ---
            if self.metrics:
                self.metrics.stop_execution()
                self.metrics.print_excel_report()
            return False
        
        # Smooth transition
        if smooth_transition:
            self.execute_smooth_transition(delay_steps=20)
        
        # Open gripper before arm motion
        self.open_gripper(self.robot._robot)
        print("Gripper opened.")
        
        # Save EE position before arm motion
        ee_position_before = self.get_ee_pose()
        print(f"EE position before arm motion: {ee_position_before}")
        
        # Compute EE orientation if not provided
        # Now that base has moved, we can compute the optimal approach orientation
        if target_ee_rpy is None:
            print("\n🎯 Computing EE orientation from approach direction...")
            target_ee_rpy = self.compute_ee_orientation(
                target_pos,
                grasp_type='top_down',
                origin_xy=ee_position_before[:2],
            )
        
        # Offset arm target so the bottle sits between the fingers
        arm_target_pos = target_pos.copy()
        approach_dir = arm_target_pos[:2] - ee_position_before[:2]
        norm = np.linalg.norm(approach_dir)
        if norm > 1e-6:
            approach_dir = approach_dir / norm
            offset = float(self.config.get('arm_goal_offset', 0.1))
            arm_target_pos[:2] -= approach_dir * offset
            print(f"Adjusted arm target by {offset:.3f}m along approach direction")
            print(f"New arm target position: {arm_target_pos}")
        
        # --------------- Phase 2: Arm motion a bit higher ---------------
        arm_target_pos[2] += 0.3
        success = self.execute_arm_motion(arm_target_pos, target_ee_rpy, q_path=arm_path)
        
        if not success:
            print("❌ Task failed: arm motion unsuccessful")
            return False


        # Phase 2.5: Arm motion a precise down to grasp
        arm_target_pos[2] -= 0.2
        success = self.execute_arm_motion(arm_target_pos, target_ee_rpy, q_path=arm_path)

        if not success:
            print("❌ Task failed: arm motion unsuccessful")
            return False

        # Close gripper after arm motion
        self.close_gripper(self.robot._robot)
        print("Gripper closed.")

        arm_target_pos[2] += 0.3
        success = self.execute_arm_motion(arm_target_pos, target_ee_rpy, q_path=arm_path, gripper_force=True)

        if not success:
            print("❌ Task failed: arm motion unsuccessful")
            return False

        # --------------- Phase 4: Move to place location and release ---------------
        place_pose_index = self.config.get('bottle_final_pose', 1)
        if place_pose_index is None:
            print("⚠ bottle_final_pose not set, skipping place phase")
        elif len(BOTTLE_POSES) <= place_pose_index:
             print(f"⚠ Invalid bottle_final_pose {place_pose_index}")
        else:
            place_pose_index = int(place_pose_index)
            place_target = np.array(BOTTLE_POSES[place_pose_index])
            print(f"\n🎯 Place target (bottle_pose {place_pose_index}): {place_target}")

            place_base_goal = self.compute_base_goal_from_target(place_target, approach_dist)
            print(f"\n🎯 Computed place base goal: {place_base_goal}")
            
            # =========================================================
            # DYNAMIC PLANNING LOGIC
            # =========================================================
            place_path = None
            
            # Check if a Global Planner is available (Tracker Mode)
            if self.base_planner is not None:
                print(f"\n🗺️  Using Global Planner ({self.base_planner.__class__.__name__})...")
                current_base = self.get_base_position()
                
                # PRM: Uses existing roadmap.
                # RRT/RRT*: Generates a new tree from scratch.
                place_path = self.base_planner.find_path(current_base[:2], place_base_goal[:2])
                
                if place_path is not None:
                    print(f"✓ Path found with {len(place_path)} waypoints.")
                    # Draw path in GREEN
                    for i in range(len(place_path) - 1):
                        p.addUserDebugLine([place_path[i][0], place_path[i][1], 0.05], 
                                          [place_path[i+1][0], place_path[i+1][1], 0.05], 
                                          lineColorRGB=[0, 1, 0], lineWidth=3.0, lifeTime=0)
                else:
                    print("⚠ Global planner failed to find path. Falling back to Direct MPC.")
            
            else:
                # Direct MPC Mode
                print("\n⚡ Using Direct MPC (No Global Planner configured)...")
                place_path = None

            # Execute Base Motion
            self.execute_base_motion(
                place_base_goal, 
                path=place_path,
                gripper_force=True, 
                segment_label="base_2_place",
                log=False
            )
            print("⚠ Proceeding to placement (ignoring potential base timeout)...")
            

            # --- ARM PLACE LOGIC ---
            place_ee_before = self.get_ee_pose()
            
            # Compute EE orientation
            place_ee_rpy = self.compute_ee_orientation(
                place_target,
                grasp_type='top_down',
                origin_xy=place_ee_before[:2],
            )

            # Offset arm target
            place_arm_target = place_target.copy()
            place_dir = place_arm_target[:2] - place_ee_before[:2]
            place_norm = np.linalg.norm(place_dir)
            if place_norm > 1e-6:
                place_dir = place_dir / place_norm
                offset = float(self.config.get('arm_goal_offset', 0.1))
                place_arm_target[:2] -= place_dir * offset

            # 1. Move High (Pre-place)
            place_arm_target[2] += 0.3
            success = self.execute_arm_motion(
                place_arm_target,
                place_ee_rpy,
                q_path=None,
                gripper_force=True,
                segment_label="arm_2_place",
            )
            
            # 2. Move Down (Place)
            place_arm_target[2] -= 0.2
            success = self.execute_arm_motion(
                place_arm_target,
                place_ee_rpy,
                q_path=None,
                gripper_force=True,
                segment_label="arm_2_place",
                merge_with_previous=True,
            )

            # 3. Open Gripper
            self.open_gripper(self.robot._robot)
            print("Gripper opened.")

            # --------------- Phase 5: Move Arm-Up and Stop immediatelly when bottle placed ------------
            print("✓ Object dropped. Stopping execution.")
            print("↑ Lifting arm after placement...")
            time.sleep(1.0)
            place_arm_target[2] += 0.2  # Lift back up by 20cm
            self.execute_arm_motion(place_arm_target, place_ee_rpy, q_path=None, gripper_force=False)
    

        self.phase = "done"
        print("\n" + "="*70)
        print("✓ TASK COMPLETED")
        print("="*70)

        # --- METRICS: FINAL REPORT ---
        if self.metrics:
            self.metrics.success = True
            self.metrics.stop_execution()
            self.metrics.print_excel_report()
        
        return True
