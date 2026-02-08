"""
Global motion planners for manipulator arm in joint space.
Similar to global_planners.py but works in N-dimensional joint space.
"""
import numpy as np
import pybullet as p
from scipy.spatial import KDTree
import heapq
import random
from typing import List, Optional, Tuple


class ArmBasePlanner:
    """Base class for arm joint-space planners."""
    
    def __init__(self, robot, arm_pb_idxs: List[int], 
                 q_min: np.ndarray, q_max: np.ndarray,
                 step_size: float = 0.05,
                 use_direct: bool = True,
                 load_world_fn=None):
        """
        Args:
            robot: Robot instance
            arm_pb_idxs: PyBullet joint indices for arm
            q_min: Joint lower limits (n,)
            q_max: Joint upper limits (n,)
            step_size: Step size in joint space [rad]
            use_direct: If True, uses p.DIRECT client for collision checking
            load_world_fn: Optional function(client_id) to load obstacles in DIRECT
        """
        self.robot = robot
        self.arm_pb_idxs = arm_pb_idxs
        self.q_min = np.asarray(q_min)
        self.q_max = np.asarray(q_max)
        self.n_joints = len(arm_pb_idxs)
        self.step_size = step_size
        
        # p.DIRECT client for collision checking (separate from GUI)
        self._use_direct = use_direct
        self._cid = None  # DIRECT client ID
        self._robot_id = robot._robot  # Default: use GUI robot
        
        if use_direct:
            self._create_direct_client(load_world_fn)
    
    def _create_direct_client(self, load_world_fn):
        """Create separate p.DIRECT physics client for collision checking."""
        try:
            # Create DIRECT client (no rendering)
            self._cid = p.connect(p.DIRECT)
            
            # Load robot URDF in DIRECT
            urdf_path = getattr(self.robot, '_urdf_file', None)
            if urdf_path is None:
                print("⚠ No URDF found, disabling p.DIRECT")
                p.disconnect(self._cid)
                self._use_direct = False
                self._cid = None
                return
            
            self._robot_id = p.loadURDF(
                urdf_path,
                useFixedBase=True,
                flags=p.URDF_USE_SELF_COLLISION,
                physicsClientId=self._cid
            )
            
            # Load obstacles if provided
            if load_world_fn is not None:
                load_world_fn(self._cid)
            
            num_bodies = p.getNumBodies(physicsClientId=self._cid)
            print(f"✓ p.DIRECT client created for arm planning")
            print(f"  Client ID: {self._cid}")
            print(f"  Robot ID: {self._robot_id}")
            print(f"  Bodies: {num_bodies}")
            
        except Exception as e:
            print(f"⚠ Failed to create p.DIRECT client: {e}")
            if self._cid is not None:
                try:
                    p.disconnect(self._cid)
                except:
                    pass
            self._use_direct = False
            self._cid = None
            self._robot_id = self.robot._robot
    
    def close(self):
        """Disconnect p.DIRECT client."""
        if self._cid is not None:
            try:
                p.disconnect(self._cid)
                print("✓ p.DIRECT client disconnected")
            except:
                pass
            self._cid = None
    
    def __del__(self):
        """Cleanup on deletion."""
        self.close()
    
    def is_free(self, q: np.ndarray) -> bool:
        """
        Check if joint configuration is collision-free.
        
        Args:
            q: Joint configuration (n,)
        
        Returns:
            True if collision-free
        """
        # Check joint limits
        if np.any(q < self.q_min) or np.any(q > self.q_max):
            return False
        
        # Use DIRECT client if available, otherwise use GUI client
        cid = self._cid if self._cid is not None else 0
        rid = self._robot_id
        
        # Set robot to configuration
        for i, pb_idx in enumerate(self.arm_pb_idxs):
            p.resetJointState(rid, pb_idx, float(q[i]), physicsClientId=cid)
        
        # Check self-collision
        contact_points = p.getContactPoints(rid, rid, physicsClientId=cid)
        if len(contact_points) > 0:
            return False
        
        # Check environment collision
        contact_points = p.getContactPoints(bodyA=rid, physicsClientId=cid)
        for cp in contact_points:
            body_b = cp[2]
            # Skip self-collision (already checked)
            if body_b == rid:
                continue
            # Skip floor/plane
            try:
                body_name = p.getBodyInfo(body_b, physicsClientId=cid)[1].decode('utf-8').lower()
                if 'floor' in body_name or 'plane' in body_name:
                    continue
            except:
                pass
            # Found collision with obstacle
            return False
        
        return True
    
    def check_edge(self, q1: np.ndarray, q2: np.ndarray) -> bool:
        """Check if straight line in joint space is collision-free."""
        dist = np.linalg.norm(q2 - q1)
        n_steps = max(1, int(np.ceil(dist / self.step_size)))
        
        for i in range(n_steps + 1):
            alpha = i / n_steps
            q_interp = (1 - alpha) * q1 + alpha * q2
            
            if not self.is_free(q_interp):
                return False
        
        return True
    
    def sample_free(self) -> np.ndarray:
        """Sample random collision-free configuration."""
        max_attempts = 100
        for _ in range(max_attempts):
            q = np.random.uniform(self.q_min, self.q_max)
            if self.is_free(q):
                return q
        raise RuntimeError("Could not sample free configuration")
    
    def distance(self, q1: np.ndarray, q2: np.ndarray) -> float:
        """Euclidean distance in joint space."""
        return np.linalg.norm(q2 - q1)


class ArmPRMPlanner(ArmBasePlanner):
    """PRM planner for manipulator arm in joint space."""
    
    def __init__(self, robot, arm_pb_idxs, q_min, q_max, step_size=0.2,
                 use_direct=True, load_world_fn=None):
        super().__init__(robot, arm_pb_idxs, q_min, q_max, step_size, 
                        use_direct, load_world_fn)
        self.graph = {}
        self.samples = []
        self.kdtree = None
    
    def build_roadmap(self, n_samples=500, k_neighbors=10):
        """Build PRM roadmap in joint space."""
        print(f"\nArm PRM: Sampling {n_samples} joint configurations...")
        self.samples = []
        self.graph = {}
        
        # Sample collision-free configurations
        attempts = 0
        max_attempts = n_samples * 10
        
        while len(self.samples) < n_samples and attempts < max_attempts:
            attempts += 1
            try:
                q = self.sample_free()
                self.samples.append(q)
                self.graph[len(self.samples) - 1] = []
            except:
                pass
            
            if len(self.samples) % 50 == 0 and len(self.samples) > 0:
                print(f"  Sampled {len(self.samples)}/{n_samples}")
        
        if len(self.samples) < 2:
            raise RuntimeError("Failed to sample enough configurations")
        
        print(f"✓ Sampled {len(self.samples)} configurations")
        print(f"Arm PRM: Connecting nodes (k={k_neighbors})...")
        
        # Build KDTree for KNN
        sample_array = np.array(self.samples)
        self.kdtree = KDTree(sample_array)
        
        # Connect k-nearest neighbors
        for i, q_i in enumerate(sample_array):
            distances, indices = self.kdtree.query(q_i, k=k_neighbors + 1)
            
            for j, dist in zip(indices[1:], distances[1:]):
                if j <= i:
                    continue
                
                if self.check_edge(self.samples[i], self.samples[j]):
                    self.graph[i].append(j)
                    self.graph[j].append(i)
            
            if i % 50 == 0:
                print(f"  Connected {i}/{len(self.samples)}")
        
        print(f"✓ Arm roadmap built")
    
    def find_path(self, q_start: np.ndarray, q_goal: np.ndarray) -> Optional[np.ndarray]:
        """
        Find path from start to goal using A*.
        
        Returns:
            path: (T, n_joints) waypoints or None
        """
        if not self.is_free(q_start):
            print("⚠ Arm start configuration in collision!")
            return None
        
        if not self.is_free(q_goal):
            print("⚠ Arm goal configuration in collision!")
            return None
        
        # Add start/goal to graph temporarily
        start_idx = len(self.samples)
        self.samples.append(q_start.copy())
        self.graph[start_idx] = []
        
        goal_idx = len(self.samples)
        self.samples.append(q_goal.copy())
        self.graph[goal_idx] = []
        

        # ✅ FIX #1: Connect to nearest neighbors WITHOUT distance threshold
        print("\n=== Connecting start/goal to roadmap ===")
        for pt, idx in [(q_start, start_idx), (q_goal, goal_idx)]:
            distances, indices = self.kdtree.query(pt, k=15)
            
            connected_count = 0
            for j, d in zip(indices, distances):
                # Remove threshold - just check collision
                if self.check_edge(pt, self.samples[j]):
                    self.graph[idx].append(j)
                    self.graph[j].append(idx)
                    connected_count += 1
                    
                    # Stop after finding enough connections
                    if connected_count >= 5:
                        break
            
            node_name = "Start" if idx == start_idx else "Goal"
            print(f"  {node_name}: nearest_dist={distances[0]:.3f}, connected={connected_count}")
            
            if connected_count == 0:
                print(f"  ⚠️ {node_name} could not connect to roadmap!")
                return None
        
        # A* with proper g_score tracking
        def heuristic(i):
            return self.distance(self.samples[i], q_goal)
        
        # Track g_score efficiently
        g_score = {start_idx: 0.0}
        came_from = {}
        
        # Priority queue: (f_score, node)
        queue = [(heuristic(start_idx), start_idx)]
        visited = set()
        
        while queue:
            f_cost, curr = heapq.heappop(queue)

            
            if curr in visited:
                continue
            visited.add(curr)
            

            # Check if goal reached
            if curr == goal_idx:
                # Reconstruct path
                path = []
                node = goal_idx
                while node in came_from:
                    path.append(self.samples[node])
                    node = came_from[node]
                path.append(self.samples[start_idx])
                path = path[::-1]
                
                print(f"✓ Arm path found: {len(path)} waypoints")
                return np.array(path)
            
            # Explore neighbors
            for neighbor in self.graph.get(curr, []):
                if neighbor in visited:
                    continue
                
                edge_cost = self.distance(self.samples[curr], self.samples[neighbor])
                tentative_g_score = g_score[curr] + edge_cost
                
                # Only update if this path is better
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = curr
                    g_score[neighbor] = tentative_g_score
                    f_score = tentative_g_score + heuristic(neighbor)
                    heapq.heappush(queue, (f_score, neighbor))

        print("⚠ No arm path found")
        return None


class ArmRRTPlanner(ArmBasePlanner):
    """RRT planner for manipulator arm."""
    
    def __init__(self, robot, arm_pb_idxs, q_min, q_max, step_size=0.3,
                 use_direct=True, load_world_fn=None):
        super().__init__(robot, arm_pb_idxs, q_min, q_max, step_size,
                        use_direct, load_world_fn)
        self.tree = []
        self.parents = {}
    
    def nearest(self, q: np.ndarray) -> int:
        """Find nearest node in tree."""
        dists = [self.distance(q, node) for node in self.tree]
        return int(np.argmin(dists))
    
    def steer(self, q_from: np.ndarray, q_to: np.ndarray) -> np.ndarray:
        """Steer from q_from toward q_to."""
        direction = q_to - q_from
        dist = np.linalg.norm(direction)
        if dist < self.step_size:
            return q_to
        direction = direction / dist * self.step_size
        return q_from + direction
    
    def find_path(self, q_start: np.ndarray, q_goal: np.ndarray, 
                  max_iter=2000, goal_bias=0.1) -> Optional[np.ndarray]:
        """RRT planning in joint space."""
        if not self.is_free(q_start) or not self.is_free(q_goal):
            return None
        

        for i, pb_idx in enumerate(self.arm_pb_idxs):
            p.resetJointState(self.robot._robot, pb_idx, q_start[i])
        

        print(f"\nArm RRT: Planning with max_iter={max_iter}...")
        
        self.tree = [q_start.copy()]
        self.parents = {0: None}
        
        for it in range(max_iter):
            # Sample
            if random.random() < goal_bias:
                q_rand = q_goal
            else:
                try:
                    q_rand = self.sample_free()
                except:
                    continue
            
            # Nearest
            nearest_idx = self.nearest(q_rand)
            q_near = self.tree[nearest_idx]
            
            # Steer
            q_new = self.steer(q_near, q_rand)
            
            # Check collision
            if not self.check_edge(q_near, q_new):
                continue
            
            # Add to tree
            new_idx = len(self.tree)
            self.tree.append(q_new)
            self.parents[new_idx] = nearest_idx
            
            # Check goal
            if self.distance(q_new, q_goal) < self.step_size:
                # Reconstruct path
                path = []
                idx = new_idx
                while idx is not None:
                    path.append(self.tree[idx])
                    idx = self.parents[idx]
                
                path = path[::-1]
                print(f"✓ Arm path found: {len(path)} waypoints")
                return np.array(path)  # (T, n_joints)
            
            if it % 200 == 0:
                print(f"  Iteration {it}/{max_iter}, tree size: {len(self.tree)}")
        
        print("⚠ No arm path found")
        return None


def get_arm_planner(planner_type: str, robot, arm_pb_idxs, q_min, q_max,
                   use_direct: bool = True, load_world_fn=None):
    """
    Factory function for arm planners.
    
    Args:
        planner_type: 'prm', 'rrt', or 'rrt*'
        robot: Robot instance
        arm_pb_idxs: PyBullet joint indices
        q_min: Joint lower limits
        q_max: Joint upper limits
        use_direct: If True, uses p.DIRECT client for collision checking
        load_world_fn: Optional function(client_id) to load obstacles in DIRECT
    
    Returns:
        Arm planner instance
    """
    if planner_type.lower() == 'prm':
        return ArmPRMPlanner(robot, arm_pb_idxs, q_min, q_max, 
                            use_direct=use_direct, load_world_fn=load_world_fn)
    elif planner_type.lower() == 'rrt':
        return ArmRRTPlanner(robot, arm_pb_idxs, q_min, q_max,
                           use_direct=use_direct, load_world_fn=load_world_fn)
    elif planner_type.lower() == 'rrt*':
        # TODO: Implement RRT* for arm
        print("⚠ RRT* not yet implemented for arm, using RRT")
        return ArmRRTPlanner(robot, arm_pb_idxs, q_min, q_max,
                           use_direct=use_direct, load_world_fn=load_world_fn)
    else:
        raise ValueError(f"Unknown arm planner: {planner_type}")


# ============================================================================
# PATH PROCESSING FUNCTIONS FOR ARM
# ============================================================================

def shortcut_arm_path(path: np.ndarray, planner, max_iterations: int = 50) -> np.ndarray:
    """
    Remove unnecessary intermediate waypoints in joint space.
    
    Args:
        path: (N, n_joints) waypoints
        planner: Arm planner with check_edge() method
        max_iterations: Number of shortcut attempts
    
    Returns:
        Shortened path (N_short, n_joints)
    """
    if len(path) < 3:
        return path
    
    # Convert to list for easier manipulation
    shortened = list(path)
    
    for _ in range(max_iterations):
        if len(shortened) < 3:
            break
        
        # Pick two random waypoints
        i = random.randint(0, len(shortened) - 3)
        j = random.randint(i + 2, len(shortened) - 1)
        
        # Can we connect them directly in joint space?
        if planner.check_edge(shortened[i], shortened[j]):
            # Remove all intermediate waypoints
            shortened = shortened[:i+1] + shortened[j:]
    
    result = np.array(shortened)
    reduction = (1 - len(result) / len(path)) * 100
    print(f"  Arm path shortcut: {len(path)} → {len(result)} waypoints ({reduction:.1f}% reduction)")
    
    return result


def smooth_arm_path(path: np.ndarray, num_points: int = 100) -> np.ndarray:
    """
    Smooth joint-space path using cubic spline.
    
    Args:
        path: (N, n_joints) waypoints
        num_points: Number of interpolated points
    
    Returns:
        Smooth path (num_points, n_joints)
    """
    from scipy.interpolate import CubicSpline
    
    if len(path) < 4:
        print(f"  Arm path too short for smoothing ({len(path)} points), using as-is")
        return path
    
    n_joints = path.shape[1]
    
    # Parameterize by cumulative distance in joint space
    distances = np.cumsum([0] + [
        np.linalg.norm(path[i+1] - path[i])
        for i in range(len(path) - 1)
    ])
    
    # Create splines for each joint
    smooth_joints = []
    for j in range(n_joints):
        cs = CubicSpline(distances, path[:, j])
        s_smooth = np.linspace(0, distances[-1], num_points)
        smooth_joints.append(cs(s_smooth))
    
    smooth_path = np.column_stack(smooth_joints)
    
    print(f"  Arm path smoothed: {len(path)} → {num_points} interpolated points")
    
    return smooth_path