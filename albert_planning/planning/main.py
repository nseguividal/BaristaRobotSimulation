"""
Main entry point for Albert mobile manipulation simulation.
"""
import argparse
import numpy as np
import warnings
from urdfenvs.robots.generic_urdf.generic_diff_drive_robot import GenericDiffDriveRobot
from bar_env import BarEnvironment, BOTTLE_POSES

from dynamics import DifferentialDriveDynamics, ManipulatorDynamics
from global_planners import get_planner
from arm_global_planners import get_arm_planner
from mpc_controllers import BaseMPC, ArmMPC
from task_planner import MobileManipulationTask
import pybullet as p

from metrics import TrialMetrics 


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Albert Mobile Manipulator Simulation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use preset configuration
  python main.py --config fast_test --render
  
  # Custom PLANNER mode
  python main.py --base_mode planner --target_x 3.0 --target_y 2.0
  
  # TRACKER mode with PRM
  python main.py --base_mode tracker --base_global_planner prm --render
  
  # List all presets
  python main.py --list_configs
        """
    )
    
    # Config preset
    parser.add_argument('--config', type=str, default='default',
                       help='Use preset configuration (see --list_configs)')
    parser.add_argument('--list_configs', action='store_true',
                       help='List all available preset configurations and exit')
    
    # Environment
    parser.add_argument('--dt', type=float, default=0.05, help='Timestep [s]')
    parser.add_argument('--render', action='store_true', default=True, help='Enable PyBullet rendering')
    
    # Task
    parser.add_argument('--initial_x', type=float, default=0.0, help='Starting X position [m]')
    parser.add_argument('--initial_y', type=float, default=0.0, help='Starting Y position [m]')
    parser.add_argument('--initial_z', type=float, default=0.15, help='Starting Y position [m]')
    parser.add_argument('--initial_theta', type=float, default=0.0, help='Starting orientation [rad]')
    parser.add_argument('--target_x', type=float, default=-0.3, help='Target X position [m]')
    parser.add_argument('--target_y', type=float, default=-0.5, help='Target Y position [m]')
    parser.add_argument('--target_z', type=float, default=1.0, help='Target Z position [m]')
    parser.add_argument('--bottle_pose', type=int, default=None,
                       help='Use bottle pose index for target and spawn (from BOTTLE_POSES)')
    parser.add_argument('--bottle_final_pose', type=int, default=1,
                       help='Use bottle pose index for final place target (from BOTTLE_POSES)')
    parser.add_argument('--approach_dist', type=float, default=0.5, 
                       help='Base distance from target [m]')
    
    # Planning
    parser.add_argument('--base_mode', type=str, default='planner',
                       choices=['planner', 'tracker'],
                       help='Base MPC mode: planner (goal in cost) or tracker (follow path)')
    parser.add_argument('--base_global_planner', type=str, default='prm',
                       choices=['prm', 'rrt', 'rrt*'],
                       help='Global planner for base (when base_mode=tracker)')
    
    parser.add_argument('--arm_mode', type=str, default='planner',
                       choices=['planner', 'tracker'],
                       help='Arm MPC mode: planner (goal in cost) or tracker (follow trajectory)')
    parser.add_argument('--arm_global_planner', type=str, default='prm',
                       choices=['prm', 'rrt', 'rrt*'],
                       help='Global planner for arm (when arm_mode=tracker)')
    
    # Deprecated (kept for backward compatibility)
    parser.add_argument('--use_global_planner', action='store_true',
                       help='[DEPRECATED] Use --base_mode=tracker instead')
    
    # Base MPC
    parser.add_argument('--base_horizon', type=int, default=20, help='Base MPC horizon')
    parser.add_argument('--base_max_iter', type=int, default=1000, help='Base MPC max iterations')
    parser.add_argument('--base_wx', type=float, default=50.0, help='Base state weight')
    parser.add_argument('--base_wu', type=float, default=20.0, help='Base input weight')
    
    # Arm MPC
    parser.add_argument('--arm_horizon', type=int, default=20, help='Arm MPC horizon')
    parser.add_argument('--arm_max_iter', type=int, default=10, help='Arm MPC max iterations')
    parser.add_argument('--arm_wq', type=float, default=50.0, help='Arm position weight')
    parser.add_argument('--arm_wv', type=float, default=5.0, help='Arm velocity weight')
    parser.add_argument('--arm_wa', type=float, default=0.5, help='Arm acceleration weight')
    parser.add_argument('--no_torque_constraints', action='store_true',
                       help='Disable torque constraints in arm MPC')
    
    # Global planner (PRM/RRT)
    parser.add_argument('--prm_samples', type=int, default=1000, help='Base PRM samples')
    parser.add_argument('--prm_k_neighbors', type=int, default=10, help='Base PRM k-nearest')
    parser.add_argument('--rrt_max_iter', type=int, default=5000, help='Base RRT max iterations')
    
    parser.add_argument('--arm_prm_samples', type=int, default=500, help='Arm PRM samples')
    parser.add_argument('--arm_prm_k_neighbors', type=int, default=10, help='Arm PRM k-nearest')
    parser.add_argument('--arm_rrt_max_iter', type=int, default=2000, help='Arm RRT max iterations')
    
    # Limits
    parser.add_argument('--base_max_steps', type=int, default=800, help='Base max steps')
    parser.add_argument('--arm_max_steps', type=int, default=300, help='Arm max steps')
    parser.add_argument('--base_tolerance', type=float, default=0.1, help='Base position tolerance [m]')
    parser.add_argument('--arm_tolerance', type=float, default=0.01, help='Arm EE tolerance [m]')

    # Dynamic obstacles
    parser.add_argument('--dynamic_obstacles', type=int, default=0,
                       help='Number of dynamic obstacles to spawn (0 = disabled)')
    parser.add_argument('--dynamic_obs_speed', type=float, default=0.4,
                       help='Speed of dynamic obstacles [m/s]')
    parser.add_argument('--dynamic_obs_radius', type=float, default=0.25,
                       help='Radius of dynamic obstacles [m]')
    
    args = parser.parse_args()
    
    # Handle --list_configs
    if args.list_configs:
        from configs import list_configs
        list_configs()
        exit(0)
    
    # Load preset if specified
    if args.config:
        from configs import get_config
        print(f"\nLoading configuration: {args.config}")
        preset = get_config(args.config)
        
        # Apply preset values (only if not explicitly overridden)
        for key, value in preset.items():
            if key == 'description':
                print(f"  {value}")
                continue
            if hasattr(args, key) and getattr(args, key) == parser.get_default(key):
                setattr(args, key, value)
                print(f"  {key} = {value}")

    if args.bottle_pose is not None:
        if args.bottle_pose < 0 or args.bottle_pose >= len(BOTTLE_POSES):
            raise ValueError(f"Invalid bottle_pose {args.bottle_pose}, available: 0-{len(BOTTLE_POSES) - 1}")
        bottle_pos = BOTTLE_POSES[args.bottle_pose]
        args.target_x, args.target_y, args.target_z = bottle_pos
        print(f"Using bottle pose {args.bottle_pose}: {bottle_pos}")

    if args.bottle_final_pose is not None:
        if args.bottle_final_pose < 0 or args.bottle_final_pose >= len(BOTTLE_POSES):
            raise ValueError(
                f"Invalid bottle_final_pose {args.bottle_final_pose}, available: 0-{len(BOTTLE_POSES) - 1}"
            )
        print(f"Using bottle_final_pose {args.bottle_final_pose}: {BOTTLE_POSES[args.bottle_final_pose]}")
    
    return args


def parse_obstacles(obstacle_str):
    """Parse obstacle string to list of (x, y) tuples."""
    if obstacle_str is None:
        return []
    obstacles = []
    for obs in obstacle_str.split(';'):
        x, y = map(float, obs.split(','))
        obstacles.append((x, y))
    return obstacles


def get_obstacles_from_env(env):
    """
    Get obstacles from environment.
    
    Returns:
        list of (x, y) tuples for base collision avoidance
    """
    try:
        obs_list = env.get_obstacles()
        if isinstance(obs_list, dict):
            obs_list = obs_list.values()
        for obs in obs_list:
            print(f"Detected obstacle: {obs}")
            print("obstacle names:", obs.name())
        obstacles = []
        for obs in obs_list:
            d = getattr(obs, "content_dict", None) or getattr(obs, "_content_dict", None)
            if isinstance(d, dict):
                pos = d["geometry"]["position"]
                obstacles.append((pos[0], pos[1]))
        return obstacles
    
    except AttributeError:
        # get_obstacles() not available
        print("⚠ Environment doesn't have get_obstacles() method")
        return []
    except Exception as e:
        print(f"⚠ Error getting obstacles: {e}")
        return []


def setup_environment(args):
    """Create environment and robot."""
    robots = [
        GenericDiffDriveRobot(
            urdf="albert.urdf",
            mode="vel",
            actuated_wheels=["wheel_right_joint", "wheel_left_joint"],
            castor_wheels=["rotacastor_right_joint", "rotacastor_left_joint"],
            wheel_radius=0.08,
            wheel_distance=0.494,
            spawn_rotation=0,
            facing_direction='-y',
        )
    ]
    
    bottle_pose_index = args.bottle_pose if args.bottle_pose is not None else 0
    env = BarEnvironment(
        dt=args.dt,
        robots=robots,
        render=args.render,
        bottle_pose_index=bottle_pose_index,
    )
    ob, info = env.reset(
        pos=np.array([-4.0, 8.0, 0.2, 0.0, 0.0, 0.0, -1.5, 0.0, 1.8, 0.5])
    )

    # Spawn dynamic obstacles if requested
    if args.dynamic_obstacles > 0:
        robot_id = robots[0]._robot
        env.spawn_dynamic_obstacles(
            n_obstacles=args.dynamic_obstacles,
            speed=args.dynamic_obs_speed,
            radius=args.dynamic_obs_radius,
            robot_id=robot_id
        )
        print(f"\n  Spawned {args.dynamic_obstacles} dynamic obstacles (speed={args.dynamic_obs_speed} m/s)")

    return env, robots[0], ob


def setup_base_controller(args, env, robot, task):
    """Setup base controller based on mode."""
    base_dynamics = DifferentialDriveDynamics(dt=args.dt, facing_direction='-y')
    
    # Get obstacles from environment
    obstacles = get_obstacles_from_env(env)
    if obstacles:
        print(f"\n📍 Detected {len(obstacles)} obstacles from environment:")
        for i, (x, y) in enumerate(obstacles):
            print(f"  Obstacle {i+1}: ({x:.2f}, {y:.2f})")
    else:
        print(f"\n📍 No obstacles detected in environment")
    
    # Compute goal from target
    target_pos = np.array([args.target_x, args.target_y, args.target_z])
    base_goal = task.compute_base_goal_from_target(target_pos, args.approach_dist)
    print(f"\nBase goal position: ({base_goal[0]:.2f}, {base_goal[1]:.2f}, {base_goal[2]:.2f})")
    
    robot_id = env._robots[0]._robot
    
    # start_pos = [-2.5, -6.5, 0.2]
    start_pos = [args.initial_x, args.initial_y, args.initial_z]
    p.resetBasePositionAndOrientation(robot_id, start_pos, p.getQuaternionFromEuler([0, 0, args.initial_theta]))
    
    if args.base_mode == 'tracker':
        # Base TRACKING MODE: Global planner + MPC tracking
        print(f"\n=== Base: TRACKING MODE ===")
        print(f"  Global planner: {args.base_global_planner.upper()}")
        print(f"  MPC will track reference trajectory")
        
        print("Initializing Planner...")
        # Hidding robot underground so that planner doesn't hit it
        # p.resetBasePositionAndOrientation(robot_id, [0, 0, -10], [0,0,0,1])
        # Setup global planner
        bounds = [-4.5, 4.5, -9.5, 9.5]
        planner = get_planner(args.base_global_planner, bounds, robot_radius=0.35, robot_id=robot_id)
        
        # ✅ Store planner in task (for path processing collision checking)
        task.base_planner = planner
        
        if args.base_global_planner == 'prm':
            planner.build_roadmap(n_samples=args.prm_samples, k_neighbors=args.prm_k_neighbors)
            print(f"  ✓ PRM roadmap built with {args.prm_samples} samples")
            if args.render:
                planner.draw_roadmap() # drawing roadmap 
        
        # Get current position and plan path
        # current_pos = task.get_base_position()
        pos, quat = p.getBasePositionAndOrientation(robot._robot)
        current_pos = pos[:2]
        print(f"Current base position before planning: ({current_pos[0]:.2f}, {current_pos[1]:.2f})")
        path = planner.find_path((current_pos[0], current_pos[1]), 
                                (base_goal[0], base_goal[1]))
        
        # Bring robot back to start
        #print(f"Teleporting robot to start position...")
        # p.resetBasePositionAndOrientation(robot_id, start_pos, p.getQuaternionFromEuler([0, 0, 0]))
        
        if not path:
            raise RuntimeError("Global planner failed to find path")
        
        print(f"  ✓ Path found: {len(path)} waypoints")
        
        if path and args.render:
            # Draw final path in green
            planner.draw_path(path)
        
        
        # Create MPC for trajectory tracking
        base_mpc = BaseMPC(
            base_dynamics, 
            args.base_horizon, 
            x_target=base_goal,  # Fallback if tracking fails
            wx=args.base_wx, 
            wu=args.base_wu,
            max_iter=args.base_max_iter, 
            env=env,
        )
        
        return base_mpc, path
    
    else:  # args.base_mode == 'planner'
        # PLANNING MODE: Direct MPC (goal in cost function)
        print(f"\n=== Base: PLANNING MODE ===")
        print(f"  MPC will plan trajectory to goal directly")
        print(f"  Goal: ({base_goal[0]:.2f}, {base_goal[1]:.2f}, {base_goal[2]:.2f})")
        
        base_mpc = BaseMPC(
            base_dynamics, 
            args.base_horizon, 
            x_target=base_goal,
            wx=args.base_wx, 
            wu=args.base_wu,
            max_iter=args.base_max_iter, 
            obstacles=obstacles,
            env=env,
        )
        
        return base_mpc, None


def setup_arm_controller(args, robot, task):
    """Setup arm controller based on mode."""
    # Get arm indices
    arm_indices = []
    for idx in range(2, robot.n() - 1):
        joint_name = robot._joint_names[idx]
        if 'finger' not in joint_name.lower() and 'gripper' not in joint_name.lower():
            arm_indices.append(idx)
    
    # Get PyBullet indices
    arm_pb_idxs = [robot._robot_joints[i] for i in arm_indices]
    
    # Create dynamics model
    arm_dynamics = ManipulatorDynamics(robot, arm_indices, args.dt)
    
    arm_path = None
    
    if args.arm_mode == 'tracker':
        # TRACKING MODE: Use joint-space global planner
        print(f"\n=== Arm: TRACKING MODE ===")
        print(f"  Global planner: {args.arm_global_planner.upper()}")
        
        try:
            # Setup joint-space planner (DON'T build roadmap yet!)
            arm_planner = get_arm_planner(
                args.arm_global_planner,
                robot,
                arm_pb_idxs,
                arm_dynamics.q_min,
                arm_dynamics.q_max
            )
            
            # ✅ Store planner in task (for path processing + planning later)
            task.arm_planner = arm_planner
            
            
            print(f"  ✓ Arm planner ready")
            print(f"  ℹ️  Roadmap will be built when arm motion starts (after base moves)")
            
        except Exception as e:
            print(f"  ⚠ Arm planner setup failed: {e}")
            print(f"  Falling back to PLANNING MODE")
            args.arm_mode = 'planner'
    
    if args.arm_mode == 'planner':
        print(f"\n=== Arm: PLANNING MODE ===")
        print(f"  MPC will plan trajectory to IK target directly")
    
    # Create MPC (target will be set via IK during execution)
    arm_mpc = ArmMPC(
        model=arm_dynamics,
        N=args.arm_horizon,
        q_target=None,  # Will be set via IK
        w_q=args.arm_wq,
        w_v=args.arm_wv,
        w_a=args.arm_wa,
        max_iter=args.arm_max_iter,
        use_torque_constraints=not args.no_torque_constraints
    )
    
    return arm_mpc, arm_path


def main():
    """Main execution."""
    args = parse_args()
    
    # Handle deprecated flag
    if args.use_global_planner:
        print("⚠ Warning: --use_global_planner is deprecated, use --base_mode=tracker")
        args.base_mode = 'tracker'
    
    # Suppress warnings if not debugging
    warnings.filterwarnings("ignore")
    
    print("="*70)
    print("ALBERT MOBILE MANIPULATOR SIMULATION")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Target: ({args.target_x:.2f}, {args.target_y:.2f}, {args.target_z:.2f})")
    print(f"  Base mode: {args.base_mode.upper()}")
    if args.base_mode == 'tracker':
        print(f"    → Global planner: {args.base_global_planner.upper()}")
    print(f"  Arm mode: {args.arm_mode.upper()}")
    if args.arm_mode == 'tracker':
        print(f"    → Global planner: {args.arm_global_planner.upper()}")
    print(f"  Base horizon: {args.base_horizon}")
    print(f"  Arm horizon: {args.arm_horizon}")
    
    # Setup
    env, robot, ob = setup_environment(args)
    
    # Create task planner
    config = {
        'use_global_planner': args.use_global_planner,
        'base_max_steps': args.base_max_steps,
        'arm_max_steps': args.arm_max_steps,
        'base_tolerance': args.base_tolerance,
        'arm_tolerance': args.arm_tolerance,
        'smooth_points': 100,  # For path processing
        'arm_prm_samples': args.arm_prm_samples,  # For roadmap building in execute_arm_motion
        'arm_prm_k_neighbors': args.arm_prm_k_neighbors,
        'bottle_final_pose': args.bottle_final_pose,
    }

    task = MobileManipulationTask(env, robot, args.dt, config)
    
    base_controller, global_path = setup_base_controller(args, env, robot, task)
    arm_controller, arm_path = setup_arm_controller(args, robot, task) # ← Pass task!
    
    # Assign controllers to task
    task.base_controller = base_controller
    task.arm_controller = arm_controller

    print(f"\n✓ Setup complete")
    
    print(f"\n✓ Setup complete")
    print(f"\nExecution Plan:")
    print(f"  1. Base motion: {args.base_mode}")
    if args.base_mode == 'tracker':
        print(f"     → Following {len(global_path)} waypoints from {args.base_global_planner.upper()}")
    else:
        print(f"     → Direct MPC planning to goal")
    print(f"  2. Smooth transition (20 steps)")
    print(f"  3. Arm motion: {args.arm_mode}")
    if args.arm_mode == 'tracker' and arm_path:
        print(f"     → Following joint-space trajectory from {args.arm_global_planner.upper()}")
    else:
        print(f"     → Direct MPC planning to IK target")
    

    # --- EXECUTE TASK ---
    # Define Targets
    target_pos = np.array([args.target_x, args.target_y, args.target_z])
    target_ee_rpy = None  # Free orientation
    success = task.execute_task(
        target_pos,
        target_ee_rpy=target_ee_rpy,
        approach_dist=args.approach_dist,
        smooth_transition=True,
        global_path=global_path,  # Pass base path
        arm_path=arm_path  # Pass arm path
    )
    
    # --- PLOTTING ---
    if success:
        print("\n" + "="*70)
        print("GENERATING PLOTS")
        print("="*70)
        
        from plotting import (
            plot_base_results, plot_arm_results, plot_full_task,
            plot_roadmap, BaseHistory, ArmHistory, TaskHistory
        )
        
        #obstacles = getattr(env, 'obstacles', [])
        obstacles = env.get_obstacles()
        #obstacles = get_obstacles_from_env(env)
        # Convert tracked data to history objects (full)
        base_hist = None
        arm_hist = None
        if task.base_history['time']:
            base_hist = BaseHistory(
                time=np.array(task.base_history['time']),
                positions=np.array(task.base_history['positions']),
                controls=np.array(task.base_history['controls']),
                goal=task.base_history['goal'],
                global_path=task.base_history['global_path'],
                mode=task.base_history['mode']
            )
        if task.arm_history['time']:
            arm_hist = ArmHistory(
                time=np.array(task.arm_history['time']),
                joint_positions=np.array(task.arm_history['joint_positions']),
                joint_velocities=np.array(task.arm_history['joint_velocities']),
                joint_accelerations=np.array(task.arm_history['joint_accelerations']),
                ee_positions=np.array(task.arm_history['ee_positions']),
                ee_target=task.arm_history['ee_target'],
                ee_start=task.arm_history['ee_start'],
                q_target=task.arm_history['q_target'],
                global_path=task.arm_history['global_path'],
                mode=task.arm_history['mode']
            )

        # Per-segment plots
        for idx, segment in enumerate(task.base_histories):
            if not segment['time']:
                continue
            base_seg_hist = BaseHistory(
                time=np.array(segment['time']),
                positions=np.array(segment['positions']),
                controls=np.array(segment['controls']),
                goal=segment['goal'],
                global_path=segment['global_path'],
                mode=segment['mode']
            )
            label = segment.get('label', f"base_{idx + 1}").replace(" ", "_")
            plot_base_results(
                base_seg_hist,
                obstacles=obstacles,
                save_path=f"PLOTS/base_results_{idx + 1}_{label}.png"
            )

        joint_names = [f"J{i+1}" for i in range(7)]
        for idx, segment in enumerate(task.arm_histories):
            if not segment['time']:
                continue
            arm_seg_hist = ArmHistory(
                time=np.array(segment['time']),
                joint_positions=np.array(segment['joint_positions']),
                joint_velocities=np.array(segment['joint_velocities']),
                joint_accelerations=np.array(segment['joint_accelerations']),
                ee_positions=np.array(segment['ee_positions']),
                ee_target=segment['ee_target'],
                ee_start=segment['ee_start'],
                q_target=segment['q_target'],
                global_path=segment['global_path'],
                mode=segment['mode']
            )
            label = segment.get('label', f"arm_{idx + 1}").replace(" ", "_")
            plot_arm_results(
                arm_seg_hist,
                robot,
                task.arm_pb_idxs,
                joint_names=joint_names,
                obstacles=obstacles,
                save_path=f"PLOTS/arm_results_{idx + 1}_{label}.png"
            )
        
        # Full task plot
        if task.base_history['time'] or task.arm_history['time']:
            task_hist = TaskHistory(
                base=base_hist if task.base_history['time'] else None,
                arm=arm_hist if task.arm_history['time'] else None,
                obstacles=obstacles
            )
            
            plot_full_task(task_hist, save_path='task_results.png')
        
        # Plot roadmap if available
        if args.base_mode == 'tracker' and task.base_planner is not None:
            plot_roadmap(task.base_planner, path=global_path,
                        start=(args.initial_x, args.initial_y),
                        goal=(task.base_history['goal'][0], task.base_history['goal'][1]),
                        obstacles=obstacles,
                        save_path='base_roadmap.png')

        arm_exec_metrics = getattr(task, "arm_exec_metrics", [])
        if arm_exec_metrics:
            durations = [m.get("duration", 0.0) for m in arm_exec_metrics]
            ee_lengths = [m.get("ee_path_length", 0.0) for m in arm_exec_metrics]
            final_errors = [m.get("final_ee_error", 0.0) for m in arm_exec_metrics]
            max_accels = [m.get("max_acc", 0.0) for m in arm_exec_metrics]

            duration = float(np.sum(durations)) if durations else 0.0
            total_dist = float(np.sum(ee_lengths)) if ee_lengths else 0.0
            final_err = float(np.mean(final_errors)) if final_errors else 0.0
            max_acc = float(np.max(max_accels)) if max_accels else 0.0

            planner_metrics = [
                m for m in arm_exec_metrics
                if m.get("planner_type")
                and any(k in m["planner_type"].lower() for k in ("prm", "rrt"))
            ]

            print("\n" + "="*40)
            print("🦾 ARM METRICS")
            print("="*40)
            print(f"Duration (s):         {duration:.4f}")
            print(f"EE Path Length (m):   {total_dist:.4f}")
            print(f"Final EE Error (m):   {final_err:.4f}")
            print(f"Max Accel (rad/s²):   {max_acc:.4f}")
            if planner_metrics:
                planner_time = float(np.sum([m.get("planner_time") or 0.0 for m in planner_metrics]))
                planner_len = float(np.sum([m.get("planner_path_length") or 0.0 for m in planner_metrics]))
                print(f"Planner Time (s):     {planner_time:.4f}")
            print("="*40 + "\n")
        elif task.arm_history['time']:
            ee_positions = np.array(task.arm_history['ee_positions'])
            total_dist = 0.0
            if ee_positions.shape[0] > 1:
                total_dist = float(np.sum(np.linalg.norm(np.diff(ee_positions, axis=0), axis=1)))
            ee_target = task.arm_history['ee_target']
            final_err = float(np.linalg.norm(ee_positions[-1] - ee_target)) if ee_target is not None else 0.0
            acc = np.array(task.arm_history['joint_accelerations'])
            max_acc = float(np.max(np.linalg.norm(acc, axis=1))) if acc.size else 0.0
            t_arr = np.array(task.arm_history['time'])
            duration = float(t_arr[-1] - t_arr[0]) if t_arr.size > 1 else 0.0

            print("\n" + "="*40)
            print("🦾 ARM METRICS")
            print("="*40)
            print(f"Duration (s):         {duration:.4f}")
            print(f"EE Path Length (m):   {total_dist:.4f}")
            print(f"Final EE Error (m):   {final_err:.4f}")
            print(f"Max Accel (rad/s²):   {max_acc:.4f}")
            print("="*40 + "\n")

        print("✓ All plots generated")
    
    # Cleanup
    env.close()
    if success:
        print("\n✓ Simulation completed successfully")
        return 0
    else:
        print("\n❌ Simulation failed")
        return 1


if __name__ == "__main__":
    exit(main())
