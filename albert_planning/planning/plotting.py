"""
Comprehensive plotting for mobile manipulation MPC - NO OBSTACLE LABELS VERSION
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import pybullet as p
from dataclasses import dataclass
from typing import List, Optional, Tuple


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class BaseHistory:
    """History of base motion."""
    time: np.ndarray           # (T,) timestamps
    positions: np.ndarray      # (T, 3) [x, y, theta]
    controls: np.ndarray       # (T, 2) [v, omega]
    goal: np.ndarray          # (3,) [x, y, theta]
    global_path: Optional[List] = None  # List of (x, y) waypoints
    mode: str = "planner"     # "planner" or "tracker"


@dataclass
class ArmHistory:
    """History of arm motion."""
    time: np.ndarray               # (T,) timestamps
    joint_positions: np.ndarray    # (T, n_joints)
    joint_velocities: np.ndarray   # (T, n_joints)
    joint_accelerations: np.ndarray # (T-1, n_joints)
    ee_positions: np.ndarray       # (T, 3) [x, y, z]
    ee_target: np.ndarray         # (3,) target position
    ee_start: np.ndarray          # (3,) start position
    q_target: np.ndarray          # (n_joints,) target configuration
    global_path: Optional[np.ndarray] = None  # (N, n_joints) if available
    mode: str = "planner"


@dataclass
class TaskHistory:
    """Complete task history."""
    base: Optional[BaseHistory] = None
    arm: Optional[ArmHistory] = None
    obstacles: List = None


# ============================================================================
# OBSTACLE FORMAT CONVERSION
# ============================================================================

def convert_obstacles_to_plot_format(obstacles):
    """
    Convert obstacles from environment format to plotting format.
    
    Args:
        obstacles: Can be:
            - Dict from env.get_obstacles() {id: obstacle_object}
            - List of dicts [{'pos': [...], 'size': [...]}, ...]
            - None
    
    Returns:
        List of dicts with 'pos' and 'size' keys
    """
    if obstacles is None:
        return []
    
    # If already in correct format (list of dicts with 'pos' and 'size')
    if isinstance(obstacles, list):
        if all('pos' in obs and 'size' in obs for obs in obstacles):
            return obstacles
    
    # If dict from env.get_obstacles()
    if isinstance(obstacles, dict):
        converted = []
        for obs_id, obs in obstacles.items():
            try:
                # Try to get position and size from obstacle object
                if hasattr(obs, 'position') and hasattr(obs, 'size'):
                    pos = obs.position()[:2]  # Get x, y (ignore z)
                    size = obs.size()[:2]     # Get width, height (ignore z)
                    converted.append({'pos': pos, 'size': size})
                elif hasattr(obs, 'pos') and hasattr(obs, 'size'):
                    converted.append({'pos': obs.pos[:2], 'size': obs.size[:2]})
            except Exception as e:
                print(f"Warning: Could not convert obstacle {obs_id}: {e}")
                continue
        return converted
    
    print(f"Warning: Unknown obstacle format: {type(obstacles)}")
    return []


# ============================================================================
# BASE PLOTTING
# ============================================================================

def plot_base_results(history: BaseHistory, obstacles: List = None, 
                     save_path: str = 'base_results.png'):
    """
    Plot comprehensive base motion results.
    
    Args:
        history: BaseHistory object
        obstacles: Obstacles (any format - will be auto-converted)
        save_path: Path to save figure
    """
    # Convert obstacles to proper format
    obstacles = convert_obstacles_to_plot_format(obstacles)
    
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    T = len(history.time)
    
    # ============= 2D Trajectory =============
    ax_traj = fig.add_subplot(gs[:2, :2])
    
    # Obstacles (draw first so they're in background)
    if obstacles:
        for i, obs in enumerate(obstacles):
            x_corner = obs['pos'][0] - obs['size'][0] / 2
            y_corner = obs['pos'][1] - obs['size'][1] / 2
            
            rect = Rectangle(
                (x_corner, y_corner),
                obs['size'][0], obs['size'][1],
                facecolor='red', alpha=0.4, edgecolor='darkred',
                linewidth=3, linestyle='--',
                label='Obstacles' if i == 0 else '',
                zorder=1  # Behind trajectory
            )
            ax_traj.add_patch(rect)
    
    # Plot trajectory
    ax_traj.plot(history.positions[:, 0], history.positions[:, 1], 
                'b-', linewidth=2.5, label='Actual trajectory', alpha=0.8, zorder=3)
    
    # Global path if available
    if history.global_path is not None:
        path_array = np.array(history.global_path)
        ax_traj.plot(path_array[:, 0], path_array[:, 1], 
                    'g--', linewidth=1.5, label='Global path', alpha=0.6)
    
    # Start position
    ax_traj.scatter(history.positions[0, 0], history.positions[0, 1],
                   color='green', s=200, marker='o', edgecolors='darkgreen',
                   linewidths=3, label='Start', zorder=10)
    
    # End position
    ax_traj.scatter(history.positions[-1, 0], history.positions[-1, 1],
                   color='blue', s=150, marker='s', edgecolors='darkblue',
                   linewidths=2, label='End', zorder=10)
    
    # Goal position
    ax_traj.scatter(history.goal[0], history.goal[1],
                   color='red', s=250, marker='*', edgecolors='darkred',
                   linewidths=2, label='Goal', zorder=10)
    
    # Orientation arrows (every N points)
    arrow_step = max(1, T // 15)
    for i in range(0, T, arrow_step):
        dx = 0.15 * np.cos(history.positions[i, 2])
        dy = 0.15 * np.sin(history.positions[i, 2])
        ax_traj.arrow(history.positions[i, 0], history.positions[i, 1],
                     dx, dy, head_width=0.08, head_length=0.1,
                     fc='orange', ec='orange', alpha=0.6)
    
    ax_traj.set_xlabel('X [m]', fontsize=11)
    ax_traj.set_ylabel('Y [m]', fontsize=11)
    ax_traj.set_title('Base 2D Trajectory', fontsize=14, fontweight='bold')
    ax_traj.legend(fontsize=10)
    ax_traj.grid(True, alpha=0.3)
    ax_traj.axis('equal')
    
    # ============= Position vs Time =============
    ax_pos = fig.add_subplot(gs[0, 2])
    ax_pos.plot(history.time, history.positions[:, 0], 'r-', label='x', linewidth=2)
    ax_pos.plot(history.time, history.positions[:, 1], 'g-', label='y', linewidth=2)
    ax_pos.axhline(y=history.goal[0], color='r', linestyle='--', alpha=0.5)
    ax_pos.axhline(y=history.goal[1], color='g', linestyle='--', alpha=0.5)
    ax_pos.set_xlabel('Time [s]', fontsize=9)
    ax_pos.set_ylabel('Position [m]', fontsize=9)
    ax_pos.set_title('Position vs Time', fontsize=11, fontweight='bold')
    ax_pos.legend(fontsize=9)
    ax_pos.grid(True, alpha=0.3)
    
    # ============= Orientation vs Time =============
    ax_theta = fig.add_subplot(gs[1, 2])
    ax_theta.plot(history.time, np.rad2deg(history.positions[:, 2]), 
                 'b-', linewidth=2)
    ax_theta.axhline(y=np.rad2deg(history.goal[2]), color='b', 
                    linestyle='--', alpha=0.5)
    ax_theta.set_xlabel('Time [s]', fontsize=9)
    ax_theta.set_ylabel('Orientation [deg]', fontsize=9)
    ax_theta.set_title('Orientation vs Time', fontsize=11, fontweight='bold')
    ax_theta.grid(True, alpha=0.3)
    
    # ============= Control Inputs =============
    ax_ctrl = fig.add_subplot(gs[2, 0])
    ax_ctrl.plot(history.time, history.controls[:, 0], 'r-', 
                label='v (linear)', linewidth=2)
    ax_ctrl.set_xlabel('Time [s]', fontsize=9)
    ax_ctrl.set_ylabel('Linear velocity [m/s]', fontsize=9, color='r')
    ax_ctrl.tick_params(axis='y', labelcolor='r')
    ax_ctrl.grid(True, alpha=0.3)
    
    ax_ctrl2 = ax_ctrl.twinx()
    ax_ctrl2.plot(history.time, history.controls[:, 1], 'b-',
                 label='ω (angular)', linewidth=2)
    ax_ctrl2.set_ylabel('Angular velocity [rad/s]', fontsize=9, color='b')
    ax_ctrl2.tick_params(axis='y', labelcolor='b')
    ax_ctrl.set_title('Control Inputs', fontsize=11, fontweight='bold')
    
    # ============= Distance to Goal =============
    ax_dist = fig.add_subplot(gs[2, 1])
    distance = np.linalg.norm(history.positions[:, :2] - history.goal[:2], axis=1)
    ax_dist.plot(history.time, distance, 'purple', linewidth=2.5)
    ax_dist.axhline(y=0.1, color='red', linestyle='--', linewidth=2, 
                   label='10cm tolerance')
    ax_dist.fill_between(history.time, 0, 0.1, alpha=0.2, color='green')
    ax_dist.set_xlabel('Time [s]', fontsize=9)
    ax_dist.set_ylabel('Distance [m]', fontsize=9)
    ax_dist.set_title('Distance to Goal', fontsize=11, fontweight='bold')
    ax_dist.legend(fontsize=8)
    ax_dist.grid(True, alpha=0.3)
    ax_dist.set_yscale('log')
    
    # ============= Velocity Magnitude =============
    ax_vel = fig.add_subplot(gs[2, 2])
    vel_mag = np.abs(history.controls[:, 0])
    omega_mag = np.abs(history.controls[:, 1])
    ax_vel.plot(history.time, vel_mag, 'r-', label='|v|', linewidth=2)
    ax_vel.plot(history.time, omega_mag, 'b-', label='|ω|', linewidth=2)
    ax_vel.set_xlabel('Time [s]', fontsize=9)
    ax_vel.set_ylabel('Magnitude', fontsize=9)
    ax_vel.set_title('Control Magnitude', fontsize=11, fontweight='bold')
    ax_vel.legend(fontsize=9)
    ax_vel.grid(True, alpha=0.3)
    
    # Overall title
    mode_str = "TRACKER" if history.mode == "tracker" else "PLANNER"
    fig.suptitle(f'Base MPC Results ({mode_str} Mode)', 
                fontsize=16, fontweight='bold')
    
    # Summary
    final_dist = np.linalg.norm(history.positions[-1, :2] - history.goal[:2])
    print("\n" + "="*70)
    print("BASE TRAJECTORY SUMMARY")
    print("="*70)
    print(f"Mode: {mode_str}")
    print(f"Duration: {history.time[-1]:.2f} s")
    print(f"Final distance to goal: {final_dist*1000:.1f} mm")
    print(f"Max linear velocity: {np.max(np.abs(history.controls[:, 0])):.3f} m/s")
    print(f"Max angular velocity: {np.max(np.abs(history.controls[:, 1])):.3f} rad/s")
    print(f"Path length: {np.sum(np.linalg.norm(np.diff(history.positions[:, :2], axis=0), axis=1)):.2f} m")
    print("="*70 + "\n")
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Base results saved to: {save_path}")
    plt.show()


# ============================================================================
# ARM PLOTTING
# ============================================================================

def plot_arm_results(history: ArmHistory, robot, arm_pb_idxs: List[int],
                    joint_names: List[str] = None, 
                    obstacles: List = None,
                    save_path: str = 'arm_results.png'):
    """
    Plot comprehensive arm motion results with 3D EE trajectory.
    
    Args:
        history: ArmHistory object
        robot: Robot instance for FK computation
        arm_pb_idxs: PyBullet joint indices
        joint_names: List of joint names (optional)
        obstacles: Obstacles (any format - will be auto-converted)
        save_path: Path to save figure
    """
    # Convert obstacles to proper format
    obstacles = convert_obstacles_to_plot_format(obstacles)
    
    T = len(history.time)
    n_joints = history.joint_positions.shape[1]
    
    # Create figure
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(4, 3, hspace=0.35, wspace=0.3)
    
    # ============= 3D End-Effector Trajectory =============
    ax_3d = fig.add_subplot(gs[:2, :2], projection='3d')
    
    # Plot trajectory
    ax_3d.plot(history.ee_positions[:, 0], 
              history.ee_positions[:, 1],
              history.ee_positions[:, 2],
              'b-', linewidth=2.5, label='EE trajectory', alpha=0.8)
    
    # Start position
    ax_3d.scatter(*history.ee_start, color='green', s=200, marker='o',
                 edgecolors='darkgreen', linewidths=3, label='Start', zorder=10)
    
    # End position
    ax_3d.scatter(history.ee_positions[-1, 0],
                 history.ee_positions[-1, 1],
                 history.ee_positions[-1, 2],
                 color='blue', s=150, marker='s',
                 edgecolors='darkblue', linewidths=2, label='End', zorder=10)
    
    # Target position
    ax_3d.scatter(*history.ee_target, color='red', s=250, marker='*',
                 edgecolors='darkred', linewidths=2, label='Target', zorder=10)
    
    # Target tolerance sphere (20mm)
    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    radius = 0.02
    x_sphere = history.ee_target[0] + radius * np.cos(u) * np.sin(v)
    y_sphere = history.ee_target[1] + radius * np.sin(u) * np.sin(v)
    z_sphere = history.ee_target[2] + radius * np.cos(v)
    ax_3d.plot_surface(x_sphere, y_sphere, z_sphere, alpha=0.15, color='red')
    
    # Add 3D obstacles (vertical boxes from ground to ceiling)
    if obstacles:
        for i, obs in enumerate(obstacles):
            # Obstacle dimensions
            x_min = obs['pos'][0] - obs['size'][0]/2
            x_max = obs['pos'][0] + obs['size'][0]/2
            y_min = obs['pos'][1] - obs['size'][1]/2
            y_max = obs['pos'][1] + obs['size'][1]/2
            z_min = 0.0  # Ground
            z_max = 1.5  # Height (adjust based on workspace)
            
            # Draw box wireframe
            # Bottom face
            ax_3d.plot([x_min, x_max], [y_min, y_min], [z_min, z_min], 'r-', linewidth=2, alpha=0.7)
            ax_3d.plot([x_max, x_max], [y_min, y_max], [z_min, z_min], 'r-', linewidth=2, alpha=0.7)
            ax_3d.plot([x_max, x_min], [y_max, y_max], [z_min, z_min], 'r-', linewidth=2, alpha=0.7)
            ax_3d.plot([x_min, x_min], [y_max, y_min], [z_min, z_min], 'r-', linewidth=2, alpha=0.7)
            
            # Top face
            ax_3d.plot([x_min, x_max], [y_min, y_min], [z_max, z_max], 'r-', linewidth=2, alpha=0.7)
            ax_3d.plot([x_max, x_max], [y_min, y_max], [z_max, z_max], 'r-', linewidth=2, alpha=0.7)
            ax_3d.plot([x_max, x_min], [y_max, y_max], [z_max, z_max], 'r-', linewidth=2, alpha=0.7)
            ax_3d.plot([x_min, x_min], [y_max, y_min], [z_max, z_max], 'r-', linewidth=2, alpha=0.7)
            
            # Vertical edges
            ax_3d.plot([x_min, x_min], [y_min, y_min], [z_min, z_max], 'r-', linewidth=2, alpha=0.7)
            ax_3d.plot([x_max, x_max], [y_min, y_min], [z_min, z_max], 'r-', linewidth=2, alpha=0.7)
            ax_3d.plot([x_max, x_max], [y_max, y_max], [z_min, z_max], 'r-', linewidth=2, alpha=0.7)
            ax_3d.plot([x_min, x_min], [y_max, y_max], [z_min, z_max], 'r-', linewidth=2, alpha=0.7)
            
            # Fill faces with transparency
            # Front face
            verts_front = [[x_min, y_min, z_min], [x_max, y_min, z_min],
                          [x_max, y_min, z_max], [x_min, y_min, z_max]]
            ax_3d.add_collection3d(Poly3DCollection([verts_front], alpha=0.2, 
                                                    facecolor='red', edgecolor='none'))
            
            # Back face
            verts_back = [[x_min, y_max, z_min], [x_max, y_max, z_min],
                         [x_max, y_max, z_max], [x_min, y_max, z_max]]
            ax_3d.add_collection3d(Poly3DCollection([verts_back], alpha=0.2,
                                                    facecolor='red', edgecolor='none'))
            
            # Left face
            verts_left = [[x_min, y_min, z_min], [x_min, y_max, z_min],
                         [x_min, y_max, z_max], [x_min, y_min, z_max]]
            ax_3d.add_collection3d(Poly3DCollection([verts_left], alpha=0.2,
                                                    facecolor='red', edgecolor='none'))
            
            # Right face
            verts_right = [[x_max, y_min, z_min], [x_max, y_max, z_min],
                          [x_max, y_max, z_max], [x_max, y_min, z_max]]
            ax_3d.add_collection3d(Poly3DCollection([verts_right], alpha=0.2,
                                                    facecolor='red', edgecolor='none'))
    
    ax_3d.set_xlabel('X [m]', fontsize=11)
    ax_3d.set_ylabel('Y [m]', fontsize=11)
    ax_3d.set_zlabel('Z [m]', fontsize=11)
    ax_3d.set_title('End-Effector 3D Trajectory', fontsize=14, fontweight='bold')
    ax_3d.legend(fontsize=10, loc='upper left')
    ax_3d.grid(True, alpha=0.3)
    
    # ============= Joint Positions =============
    ax_q = fig.add_subplot(gs[0, 2])
    for j in range(min(n_joints, 7)):
        label = joint_names[j] if joint_names and j < len(joint_names) else f"q{j}"
        ax_q.plot(history.time, history.joint_positions[:, j], 
                 label=label, linewidth=1.5)
        if history.q_target is not None:
            ax_q.axhline(y=history.q_target[j], color=f'C{j}', 
                        linestyle='--', alpha=0.4, linewidth=1)
    ax_q.set_xlabel('Time [s]', fontsize=9)
    ax_q.set_ylabel('Position [rad]', fontsize=9)
    ax_q.set_title('Joint Positions', fontsize=11, fontweight='bold')
    ax_q.grid(True, alpha=0.3)
    ax_q.legend(fontsize=7, ncol=2)
    
    # ============= Joint Velocities =============
    ax_qd = fig.add_subplot(gs[1, 2])
    for j in range(min(n_joints, 7)):
        label = joint_names[j] if joint_names and j < len(joint_names) else f"qd{j}"
        ax_qd.plot(history.time, history.joint_velocities[:, j],
                  label=label, linewidth=1.5)
    ax_qd.set_xlabel('Time [s]', fontsize=9)
    ax_qd.set_ylabel('Velocity [rad/s]', fontsize=9)
    ax_qd.set_title('Joint Velocities', fontsize=11, fontweight='bold')
    ax_qd.grid(True, alpha=0.3)
    ax_qd.legend(fontsize=7, ncol=2)
    
    # ============= EE Position vs Time =============
    ax_ee = fig.add_subplot(gs[2, 0])
    ax_ee.plot(history.time, history.ee_positions[:, 0], 'r-', label='x', linewidth=2)
    ax_ee.plot(history.time, history.ee_positions[:, 1], 'g-', label='y', linewidth=2)
    ax_ee.plot(history.time, history.ee_positions[:, 2], 'b-', label='z', linewidth=2)
    ax_ee.axhline(y=history.ee_target[0], color='r', linestyle='--', alpha=0.5)
    ax_ee.axhline(y=history.ee_target[1], color='g', linestyle='--', alpha=0.5)
    ax_ee.axhline(y=history.ee_target[2], color='b', linestyle='--', alpha=0.5)
    ax_ee.set_xlabel('Time [s]', fontsize=9)
    ax_ee.set_ylabel('EE Position [m]', fontsize=9)
    ax_ee.set_title('EE Position vs Time', fontsize=11, fontweight='bold')
    ax_ee.legend(fontsize=9)
    ax_ee.grid(True, alpha=0.3)
    
    # ============= EE Distance to Target =============
    ax_ee_dist = fig.add_subplot(gs[2, 1])
    ee_error = np.linalg.norm(history.ee_positions - history.ee_target, axis=1)
    ax_ee_dist.plot(history.time, ee_error * 1000, 'purple', linewidth=2.5)
    ax_ee_dist.axhline(y=20, color='red', linestyle='--', linewidth=2, 
                      label='20mm tolerance')
    ax_ee_dist.fill_between(history.time, 0, 20, alpha=0.2, color='green')
    ax_ee_dist.set_xlabel('Time [s]', fontsize=9)
    ax_ee_dist.set_ylabel('Distance [mm]', fontsize=9)
    ax_ee_dist.set_title('EE Distance to Target', fontsize=11, fontweight='bold')
    ax_ee_dist.legend(fontsize=8)
    ax_ee_dist.grid(True, alpha=0.3)
    ax_ee_dist.set_yscale('log')
    
    # ============= Joint Accelerations =============
    ax_u = fig.add_subplot(gs[2, 2])
    for j in range(min(n_joints, 7)):
        label = joint_names[j] if joint_names and j < len(joint_names) else f"u{j}"
        ax_u.plot(history.time, history.joint_accelerations[:, j],
                 label=label, linewidth=1.5, alpha=0.7)
    ax_u.set_xlabel('Time [s]', fontsize=9)
    ax_u.set_ylabel('Acceleration [rad/s²]', fontsize=9)
    ax_u.set_title('Joint Accelerations', fontsize=11, fontweight='bold')
    ax_u.grid(True, alpha=0.3)
    ax_u.legend(fontsize=7, ncol=2)
    
    # ============= Joint Error Norm =============
    ax_q_err = fig.add_subplot(gs[3, 0])
    if history.q_target is not None:
        q_error = np.linalg.norm(history.joint_positions - history.q_target, axis=1)
        ax_q_err.plot(history.time, q_error, 'b-', linewidth=2.5)
        ax_q_err.set_ylabel('||q - q*|| [rad]', fontsize=9)
    else:
        q_error = np.linalg.norm(history.joint_positions - history.joint_positions[-1], axis=1)
        ax_q_err.plot(history.time, q_error, 'b-', linewidth=2.5)
        ax_q_err.set_ylabel('||q - q_final|| [rad]', fontsize=9)
    ax_q_err.set_xlabel('Time [s]', fontsize=9)
    ax_q_err.set_title('Joint Configuration Error', fontsize=11, fontweight='bold')
    ax_q_err.grid(True, alpha=0.3)
    ax_q_err.set_yscale('log')
    
    # ============= Velocity Norm =============
    ax_vel_norm = fig.add_subplot(gs[3, 1])
    qd_norm = np.linalg.norm(history.joint_velocities, axis=1)
    ax_vel_norm.plot(history.time, qd_norm, 'g-', linewidth=2.5)
    ax_vel_norm.set_xlabel('Time [s]', fontsize=9)
    ax_vel_norm.set_ylabel('||qd|| [rad/s]', fontsize=9)
    ax_vel_norm.set_title('Joint Velocity Norm', fontsize=11, fontweight='bold')
    ax_vel_norm.grid(True, alpha=0.3)
    
    # ============= Acceleration Norm =============
    ax_acc_norm = fig.add_subplot(gs[3, 2])
    u_norm = np.linalg.norm(history.joint_accelerations, axis=1)
    ax_acc_norm.plot(history.time, u_norm, 'r-', linewidth=2.5)
    ax_acc_norm.set_xlabel('Time [s]', fontsize=9)
    ax_acc_norm.set_ylabel('||u|| [rad/s²]', fontsize=9)
    ax_acc_norm.set_title('Acceleration Norm', fontsize=11, fontweight='bold')
    ax_acc_norm.grid(True, alpha=0.3)
    
    # Overall title
    mode_str = "TRACKER" if history.mode == "tracker" else "PLANNER"
    fig.suptitle(f'Arm MPC Results ({mode_str} Mode)', 
                fontsize=16, fontweight='bold')
    
    # Summary
    final_ee_error = np.linalg.norm(history.ee_positions[-1] - history.ee_target)
    print("\n" + "="*70)
    print("ARM TRAJECTORY SUMMARY")
    print("="*70)
    print(f"Mode: {mode_str}")
    print(f"Duration: {history.time[-1]:.2f} s")
    print(f"Final EE error: {final_ee_error*1000:.2f} mm")
    print(f"Max EE error: {np.max(ee_error)*1000:.2f} mm")
    print(f"Mean EE error: {np.mean(ee_error)*1000:.2f} mm")
    if history.q_target is not None:
        final_q_error = np.linalg.norm(history.joint_positions[-1] - history.q_target)
        print(f"Final joint error: {final_q_error:.6f} rad ({np.rad2deg(final_q_error):.3f}°)")
    print(f"Max acceleration: {np.max(np.abs(history.joint_accelerations)):.2f} rad/s²")
    print(f"Trajectory length: {np.sum(np.linalg.norm(np.diff(history.ee_positions, axis=0), axis=1))*1000:.1f} mm")
    print("="*70 + "\n")
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Arm results saved to: {save_path}")
    plt.show()


# ============================================================================
# FULL TASK PLOTTING
# ============================================================================

def plot_full_task(task_history: TaskHistory, save_path: str = 'task_results.png'):
    """
    Plot complete mobile manipulation task (base + arm).
    
    Args:
        task_history: TaskHistory object
        save_path: Path to save figure
    """
    # Convert obstacles to proper format
    obstacles = convert_obstacles_to_plot_format(task_history.obstacles)
    
    fig = plt.figure(figsize=(20, 10))
    gs = fig.add_gridspec(2, 4, hspace=0.25, wspace=0.3)
    
    # ============= Combined 2D Trajectory (Base + Target) =============
    ax_combined = fig.add_subplot(gs[:, :2])
    
    # Obstacles (draw first for background)
    if obstacles:
        for i, obs in enumerate(obstacles):
            x_corner = obs['pos'][0] - obs['size'][0] / 2
            y_corner = obs['pos'][1] - obs['size'][1] / 2
            
            rect = Rectangle(
                (x_corner, y_corner),
                obs['size'][0], obs['size'][1],
                facecolor='red', alpha=0.4, edgecolor='darkred',
                linewidth=3, linestyle='--',
                label='Obstacles' if i == 0 else '',
                zorder=1
            )
            ax_combined.add_patch(rect)
    
    # Base trajectory (draw after obstacles)
    if task_history.base is not None:
        base_hist = task_history.base
        
        ax_combined.plot(base_hist.positions[:, 0], base_hist.positions[:, 1],
                        'b-', linewidth=2.5, label='Base trajectory', alpha=0.8, zorder=3)
        
        # Base start/end
        ax_combined.scatter(base_hist.positions[0, 0], base_hist.positions[0, 1],
                           color='green', s=200, marker='o', edgecolors='darkgreen',
                           linewidths=3, label='Base start', zorder=10)
        ax_combined.scatter(base_hist.positions[-1, 0], base_hist.positions[-1, 1],
                           color='blue', s=150, marker='s', edgecolors='darkblue',
                           linewidths=2, label='Base end', zorder=10)
    
    if task_history.arm is not None:
        arm_hist = task_history.arm
        
        # EE target in XY plane
        ax_combined.scatter(arm_hist.ee_target[0], arm_hist.ee_target[1],
                           color='red', s=300, marker='*', edgecolors='darkred',
                           linewidths=2, label='EE target (XY)', zorder=10)
    
    ax_combined.set_xlabel('X [m]', fontsize=12)
    ax_combined.set_ylabel('Y [m]', fontsize=12)
    ax_combined.set_title('Mobile Manipulation Task (Top View)', 
                         fontsize=14, fontweight='bold')
    ax_combined.legend(fontsize=10)
    ax_combined.grid(True, alpha=0.3)
    ax_combined.axis('equal')
    
    # ============= Base Position Error =============
    if task_history.base is not None:
        ax_base_err = fig.add_subplot(gs[0, 2])
        base_dist = np.linalg.norm(base_hist.positions[:, :2] - base_hist.goal[:2], axis=1)
        ax_base_err.plot(base_hist.time, base_dist, 'b-', linewidth=2.5)
        ax_base_err.axhline(y=0.1, color='red', linestyle='--', linewidth=2)
        ax_base_err.set_xlabel('Time [s]', fontsize=9)
        ax_base_err.set_ylabel('Distance [m]', fontsize=9)
        ax_base_err.set_title('Base Distance to Goal', fontsize=11, fontweight='bold')
        ax_base_err.grid(True, alpha=0.3)
        ax_base_err.set_yscale('log')
    
    # ============= Arm EE Error =============
    if task_history.arm is not None:
        ax_arm_err = fig.add_subplot(gs[0, 3])
        ee_error = np.linalg.norm(arm_hist.ee_positions - arm_hist.ee_target, axis=1)
        ax_arm_err.plot(arm_hist.time, ee_error * 1000, 'r-', linewidth=2.5)
        ax_arm_err.axhline(y=20, color='red', linestyle='--', linewidth=2)
        ax_arm_err.set_xlabel('Time [s]', fontsize=9)
        ax_arm_err.set_ylabel('Distance [mm]', fontsize=9)
        ax_arm_err.set_title('EE Distance to Target', fontsize=11, fontweight='bold')
        ax_arm_err.grid(True, alpha=0.3)
        ax_arm_err.set_yscale('log')
    
    # ============= Base Controls =============
    if task_history.base is not None:
        ax_base_ctrl = fig.add_subplot(gs[1, 2])
        ax_base_ctrl.plot(base_hist.time, base_hist.controls[:, 0], 
                         'r-', label='v', linewidth=2)
        ax_base_ctrl.plot(base_hist.time, base_hist.controls[:, 1],
                         'b-', label='ω', linewidth=2)
        ax_base_ctrl.set_xlabel('Time [s]', fontsize=9)
        ax_base_ctrl.set_ylabel('Control', fontsize=9)
        ax_base_ctrl.set_title('Base Control Inputs', fontsize=11, fontweight='bold')
        ax_base_ctrl.legend(fontsize=9)
        ax_base_ctrl.grid(True, alpha=0.3)
    
    # ============= Arm Joint Velocities =============
    if task_history.arm is not None:
        ax_arm_vel = fig.add_subplot(gs[1, 3])
        qd_norm = np.linalg.norm(arm_hist.joint_velocities, axis=1)
        ax_arm_vel.plot(arm_hist.time, qd_norm, 'g-', linewidth=2.5)
        ax_arm_vel.set_xlabel('Time [s]', fontsize=9)
        ax_arm_vel.set_ylabel('||qd|| [rad/s]', fontsize=9)
        ax_arm_vel.set_title('Joint Velocity Norm', fontsize=11, fontweight='bold')
        ax_arm_vel.grid(True, alpha=0.3)
    
    fig.suptitle('Mobile Manipulation Task - Complete Results', 
                fontsize=16, fontweight='bold')
    
    # Print summary
    print("\n" + "="*70)
    print("COMPLETE TASK SUMMARY")
    print("="*70)
    if task_history.base is not None:
        print(f"Base motion: {base_hist.time[-1]:.2f}s")
    if task_history.arm is not None:
        print(f"Arm motion: {arm_hist.time[-1]:.2f}s")
    print("="*70 + "\n")
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Task results saved to: {save_path}")
    plt.show()


# ============================================================================
# ROADMAP VISUALIZATION
# ============================================================================

def plot_roadmap(planner, path: List = None, start: Tuple = None, 
                goal: Tuple = None, obstacles: List = None,
                save_path: str = 'roadmap.png'):
    """
    Visualize PRM/RRT roadmap with path.
    
    Args:
        planner: Planner instance with samples and graph/tree
        path: Planned path (list of waypoints)
        start: Start position (x, y)
        goal: Goal position (x, y)
        obstacles: Obstacles (any format - will be auto-converted)
        save_path: Path to save figure
    """
    # Convert obstacles to proper format
    obstacles = convert_obstacles_to_plot_format(obstacles)
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Draw all nodes
    if hasattr(planner, 'samples') and planner.samples:
        samples = np.array(planner.samples)
        ax.scatter(samples[:, 0], samples[:, 1], c='lightblue', s=20, 
                  alpha=0.6, label='Sampled nodes')
    
    # Draw edges
    if hasattr(planner, 'graph') and planner.graph:
        for i, neighbors in planner.graph.items():
            for j in neighbors:
                if i < j:  # Avoid double drawing
                    ax.plot([planner.samples[i][0], planner.samples[j][0]],
                           [planner.samples[i][1], planner.samples[j][1]],
                           'b-', linewidth=0.5, alpha=0.3)
    
    # Draw RRT tree
    if hasattr(planner, 'tree') and hasattr(planner, 'parents'):
        for idx, parent_idx in planner.parents.items():
            if parent_idx is not None:
                ax.plot([planner.tree[idx][0], planner.tree[parent_idx][0]],
                       [planner.tree[idx][1], planner.tree[parent_idx][1]],
                       'b-', linewidth=0.5, alpha=0.3)
    
    # Draw path
    if path is not None:
        path_array = np.array(path)
        ax.plot(path_array[:, 0], path_array[:, 1], 'g-', 
               linewidth=3, label='Planned path', zorder=5)
    
    # Draw start and goal
    if start is not None:
        ax.scatter(*start, color='green', s=200, marker='o',
                  edgecolors='darkgreen', linewidths=3, label='Start', zorder=10)
    if goal is not None:
        ax.scatter(*goal, color='red', s=250, marker='*',
                  edgecolors='darkred', linewidths=2, label='Goal', zorder=10)
    
    # Draw obstacles
    if obstacles:
        for i, obs in enumerate(obstacles):
            x_corner = obs['pos'][0] - obs['size'][0] / 2
            y_corner = obs['pos'][1] - obs['size'][1] / 2
            
            rect = Rectangle(
                (x_corner, y_corner),
                obs['size'][0], obs['size'][1],
                facecolor='red', alpha=0.4, edgecolor='darkred',
                linewidth=2, label='Obstacles' if i == 0 else ''
            )
            ax.add_patch(rect)
    
    ax.set_xlabel('X [m]', fontsize=12)
    ax.set_ylabel('Y [m]', fontsize=12)
    ax.set_title('Global Planner Roadmap', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Roadmap saved to: {save_path}")
    plt.show()