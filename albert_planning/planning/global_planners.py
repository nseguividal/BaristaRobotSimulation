"""
Global motion planners: PRM, RRT, RRT*, RRT-Connect.
"""
import numpy as np
import pybullet as p
from scipy.spatial import KDTree
import heapq
import random
from typing import List, Optional, Tuple


class BasePlanner:
    """Base class for sampling-based planners."""
    
    def __init__(self, bounds, robot_radius=0.35, step_size=0.1, robot_id=None):
        self.x_min, self.x_max, self.y_min, self.y_max = bounds
        self.robot_radius = robot_radius
        self.step_size = step_size
        self.robot_id = robot_id
    
    def is_free(self, pos: Tuple[float, float]) -> bool:
        """Check if position is collision-free using AABB."""
        x, y = pos
        r = self.robot_radius
        
        # Define a Box around the robot position
        # Z: Start at 0.1 (above floor) and go up to 1.5 (head height)
        aabb_min = [x - r, y - r, 0.1]
        aabb_max = [x + r, y + r, 1.5]
        
        # Ask PyBullet: "What objects are inside this box?"
        overlaps = p.getOverlappingObjects(aabb_min, aabb_max)
        if overlaps:
            for body_id, _ in overlaps:
                # Skip robot itself!
                if body_id == self.robot_id:
                    continue
                
                body_name = p.getBodyInfo(body_id)[1].decode('utf-8')
                print(f"DEBUG: Collision detected with {body_name} at {pos}")
                if "floor" not in body_name.lower() and "plane" not in body_name.lower():
                    return False  # Found a valid obstacle
        return True  # Box is empty
    
    def check_edge(self, p1: Tuple, p2: Tuple) -> bool:
        """Check if straight line is collision-free."""
        dist = np.linalg.norm(np.array(p2) - np.array(p1))
        steps = max(1, int(dist / self.step_size))
        
        for i in range(steps + 1):
            alpha = i / steps
            pt = (1 - alpha) * np.array(p1) + alpha * np.array(p2)
            if not self.is_free(tuple(pt)):
                return False
        return True
    
    def sample_free(self) -> Tuple[float, float]:
        """Sample random free configuration."""
        max_attempts = 1000
        for _ in range(max_attempts):
            x = random.uniform(self.x_min, self.x_max)
            y = random.uniform(self.y_min, self.y_max)
            if self.is_free((x, y)):
                return (x, y)
        raise RuntimeError("Could not sample free configuration")


class PRMPlanner(BasePlanner):
    """Probabilistic Roadmap planner."""
    
    def __init__(self, bounds, robot_radius=0.35, step_size=0.1, robot_id=None):
        super().__init__(bounds, robot_radius, step_size, robot_id)
        self.graph = {}
        self.samples = []
        self.kdtree = None
    
    def build_roadmap(self, n_samples=1000, k_neighbors=10):
        """Build PRM roadmap."""
        print(f"PRM: Sampling {n_samples} nodes...")
        self.samples = []
        self.graph = {}
        
        # 1. Random Sampling
        attempts = 0
        while len(self.samples) < n_samples and attempts < n_samples * 5:
            attempts += 1
            rand_x = random.uniform(self.x_min, self.x_max)
            rand_y = random.uniform(self.y_min, self.y_max)
            if self.is_free((rand_x, rand_y)):
                self.samples.append((rand_x, rand_y))
                self.graph[len(self.samples)-1] = []
        
        if len(self.samples) < 2:
            raise RuntimeError("Failed to sample enough nodes")
        
        print(f"PRM: Connecting {len(self.samples)} nodes (k={k_neighbors})...")
        # 2. Vectorized KNN Connections
        sample_array = np.array(self.samples)
        for i, point in enumerate(sample_array):
            dists = np.linalg.norm(sample_array - point, axis=1)
            closest_indices = np.argsort(dists)[1:k_neighbors + 1]
            
            for j in closest_indices:
                if j <= i:
                    continue
                if self.check_edge(self.samples[i], self.samples[j]):
                    self.graph[i].append(j)
                    self.graph[j].append(i)
        
        self.kdtree = KDTree(sample_array)
        print(f"PRM: Roadmap ready")
    
    def find_path(self, start: Tuple, goal: Tuple) -> List[Tuple]:
        """A* search on roadmap."""
        # Check Validity
        # if not self.is_free(start):
        #     print("Start is obstructed!")
        #     return []
        # if not self.is_free(goal):
        #     print("Goal is obstructed!")
        #     return []
        
        # Add start/goal to graph
        start_idx = len(self.samples)
        self.samples.append(start)
        self.graph[start_idx] = []
        
        goal_idx = len(self.samples)
        self.samples.append(goal)
        self.graph[goal_idx] = []
        
        # Connect to nearest neighbors
        for pt, idx in [(start, start_idx), (goal, goal_idx)]:
            dists, indices = self.kdtree.query(pt, k=15)
            for j, d in zip(indices, dists):
                if d < 5.0 and self.check_edge(pt, self.samples[j]):
                    self.graph[idx].append(j)
                    self.graph[j].append(idx)
        
        # A* search
        # Queue: (f_score, node, path, g_score)
        queue = [(0, start_idx, [], 0)]
        visited = set()
        
        while queue:
            # Extract f_score and g_score
            f_score, curr, path, g_score = heapq.heappop(queue)
            if curr in visited:
                continue
            visited.add(curr)
            
            path = path + [self.samples[curr]]
            if curr == goal_idx:
                return path
            
            for neighbor in self.graph.get(curr, []):
                if neighbor not in visited:
                    edge_cost = np.linalg.norm(np.array(self.samples[curr]) - np.array(self.samples[neighbor]))
                    # A*: g_score = actual cost from start
                    new_g_score = g_score + edge_cost
                    heuristic = np.linalg.norm(np.array(self.samples[neighbor]) - np.array(goal))
                    new_f_score = new_g_score + heuristic
                    # Push (f_score, node, path, g_score)
                    heapq.heappush(queue, (new_f_score, neighbor, path, new_g_score))
        
        return []

    def draw_roadmap(self):
        print(f"Drawing full roadmap with {len(self.samples)} nodes...")
        for i, neighbors in self.graph.items():
            p1 = self.samples[i]
            for j in neighbors:
                # Only draw if i < j to avoid drawing every line twice
                if i < j:
                    p2 = self.samples[j]
                    # Blue lines, thin
                    p.addUserDebugLine([p1[0], p1[1], 0.1], 
                                       [p2[0], p2[1], 0.1], 
                                       [0, 0, 1], lineWidth=1, lifeTime=0)
                    
    def draw_path(self, path):
        p.removeAllUserDebugItems()
        for i in range(len(path) - 1):
            p.addUserDebugLine([path[i][0], path[i][1], 0.1], 
                               [path[i+1][0], path[i+1][1], 0.1], 
                               [0, 1, 0], lineWidth=3)


class RRTPlanner(BasePlanner):
    """Rapidly-exploring Random Tree planner for 2D base motion."""

    def __init__(self, bounds, robot_radius=0.35, step_size=0.1, robot_id=None):
        super().__init__(bounds, robot_radius, step_size, robot_id)
        self.tree = []      # List of (x, y) configurations
        self.parents = {}   # parents[i] = index of parent node

    def nearest(self, pos: Tuple[float, float]) -> int:
        """Find index of nearest node in tree."""
        dists = [np.linalg.norm(np.array(pos) - np.array(node)) for node in self.tree]
        return int(np.argmin(dists))

    def steer(self, p_from: Tuple, p_to: Tuple) -> Tuple[float, float]:
        """Steer from p_from toward p_to with limited step size."""
        direction = np.array(p_to) - np.array(p_from)
        dist = np.linalg.norm(direction)
        if dist < self.step_size:
            return p_to
        direction = direction / dist * self.step_size
        new_pos = np.array(p_from) + direction
        return (float(new_pos[0]), float(new_pos[1]))

    def find_path(self, start: Tuple, goal: Tuple,
                  max_iter: int = 2000, goal_bias: float = 0.1) -> List[Tuple]:
        """
        Find path from start to goal using RRT.

        Args:
            start: Start position (x, y)
            goal: Goal position (x, y)
            max_iter: Maximum iterations
            goal_bias: Probability of sampling goal directly

        Returns:
            List of (x, y) waypoints or empty list if no path found
        """
        # if not self.is_free(start):
        #     print("RRT: Start position is obstructed!")
        #     return []
        # if not self.is_free(goal):
        #     print("RRT: Goal position is obstructed!")
        #     return []

        print(f"RRT: Planning with max_iter={max_iter}, goal_bias={goal_bias}...")

        # Initialize tree with start
        self.tree = [start]
        self.parents = {0: None}

        for it in range(max_iter):
            # Sample: with goal_bias probability, sample goal directly
            if random.random() < goal_bias:
                q_rand = goal
            else:
                try:
                    q_rand = self.sample_free()
                except RuntimeError:
                    continue

            # Find nearest node in tree
            nearest_idx = self.nearest(q_rand)
            q_near = self.tree[nearest_idx]

            # Steer toward sample
            q_new = self.steer(q_near, q_rand)

            # Check if edge is collision-free
            if not self.check_edge(q_near, q_new):
                continue

            # Add new node to tree
            new_idx = len(self.tree)
            self.tree.append(q_new)
            self.parents[new_idx] = nearest_idx

            # Check if goal is reached
            dist_to_goal = np.linalg.norm(np.array(q_new) - np.array(goal))
            if dist_to_goal < self.step_size:
                # Try to connect directly to goal
                if self.check_edge(q_new, goal):
                    goal_idx = len(self.tree)
                    self.tree.append(goal)
                    self.parents[goal_idx] = new_idx

                    # Reconstruct path
                    path = []
                    idx = goal_idx
                    while idx is not None:
                        path.append(self.tree[idx])
                        idx = self.parents[idx]
                    path = path[::-1]

                    print(f"RRT: Path found with {len(path)} waypoints after {it+1} iterations")
                    return path

            if it % 500 == 0 and it > 0:
                print(f"  RRT iteration {it}/{max_iter}, tree size: {len(self.tree)}")

        print("RRT: No path found")
        return []

    def draw_tree(self):
        """Visualize the RRT tree in PyBullet."""
        print(f"Drawing RRT tree with {len(self.tree)} nodes...")
        for i, parent_idx in self.parents.items():
            if parent_idx is not None:
                p1 = self.tree[parent_idx]
                p2 = self.tree[i]
                p.addUserDebugLine([p1[0], p1[1], 0.1],
                                   [p2[0], p2[1], 0.1],
                                   [0.5, 0.5, 1], lineWidth=1, lifeTime=0)

    def draw_path(self, path):
        """Visualize the path in PyBullet."""
        p.removeAllUserDebugItems()
        for i in range(len(path) - 1):
            p.addUserDebugLine([path[i][0], path[i][1], 0.1],
                               [path[i+1][0], path[i+1][1], 0.1],
                               [0, 1, 0], lineWidth=3)


class RRTStarPlanner(BasePlanner):
    """RRT* planner with asymptotic optimality for 2D base motion.

    Uses OptimalNode inner class for proper cost propagation during rewiring.
    """

    class OptimalNode:
        """Node class for RRT* with cost tracking and rewiring support."""

        def __init__(self, config, parent=None, cost=0.0):
            self.config = config
            self.parent = parent
            self.children = set()
            self.cost = cost
            if parent is not None:
                self.parent.children.add(self)

        def rewire(self, new_parent, new_cost):
            """Change parent and update costs recursively."""
            if self.parent is not None:
                self.parent.children.discard(self)
            self.parent = new_parent
            self.parent.children.add(self)
            self.cost = new_cost
            self._update_children_costs()

        def _update_children_costs(self):
            """Recursively update costs of all descendants."""
            for child in self.children:
                edge_cost = np.linalg.norm(
                    np.array(child.config) - np.array(self.config))
                child.cost = self.cost + edge_cost
                child._update_children_costs()

        def retrace(self) -> List[Tuple]:
            """Reconstruct path from start to this node."""
            path = []
            node = self
            while node is not None:
                path.append(node.config)
                node = node.parent
            return path[::-1]

    def __init__(self, bounds, robot_radius=0.35, step_size=0.1, robot_id=None):
        super().__init__(bounds, robot_radius, step_size, robot_id)
        self.nodes = []


    def nearest(self, pos: Tuple[float, float]) -> 'RRTStarPlanner.OptimalNode':
        """Find nearest node to position."""
        best_node = None
        best_dist = float('inf')
        for node in self.nodes:
            dist = np.linalg.norm(np.array(pos) - np.array(node.config))
            if dist < best_dist:
                best_dist = dist
                best_node = node
        return best_node

    def near(self, pos: Tuple[float, float], radius: float) -> List['RRTStarPlanner.OptimalNode']:
        """Find all nodes within radius."""
        neighbors = []
        for node in self.nodes:
            if np.linalg.norm(np.array(pos) - np.array(node.config)) < radius:
                neighbors.append(node)
        return neighbors

    def steer(self, p_from: Tuple, p_to: Tuple) -> Tuple[float, float]:
        """Steer from p_from toward p_to with limited step size."""
        direction = np.array(p_to) - np.array(p_from)
        dist = np.linalg.norm(direction)
        if dist < self.step_size:
            return p_to
        direction = direction / dist * self.step_size
        new_pos = np.array(p_from) + direction
        return (float(new_pos[0]), float(new_pos[1]))

    def get_radius(self, n: int) -> float:
        """Compute near radius for RRT*."""
        if n < 2:
            return self.step_size * 5
        gamma = 1.5
        radius = gamma * np.sqrt(np.log(n) / n)
        return min(radius * 2, self.step_size * 5)

    def find_path(self, start: Tuple, goal: Tuple,
                  max_iter: int = 2000, goal_bias: float = 0.1) -> List[Tuple]:
        """Find optimal path using RRT*."""

        if not self.is_free(start):
            print("RRT*: Start position is obstructed!")
            return []
        if not self.is_free(goal):
            print("RRT*: Goal position is obstructed!")
            return []


        print(f"RRT*: Planning with max_iter={max_iter}, goal_bias={goal_bias}...")

        # Initialize with start node
        start_node = self.OptimalNode(start, parent=None, cost=0.0)
        self.nodes = [start_node]

        goal_node = None  # Best node that reaches the goal
        EPSILON = self.step_size * 2  # Tolerance for goal reach

        for it in range(max_iter):
            # Sample (with goal bias)
            do_goal = (goal_node is None) and (random.random() < goal_bias)
            if do_goal:
                q_rand = goal
            else:
                try:
                    q_rand = self.sample_free()
                except RuntimeError:
                    continue

            # Find nearest and steer
            nearest_node = self.nearest(q_rand)
            q_new = self.steer(nearest_node.config, q_rand)

            # Check collision
            if not self.is_free(q_new):
                continue
            if not self.check_edge(nearest_node.config, q_new):
                continue

            # Compute cost via nearest
            edge_cost = np.linalg.norm(np.array(q_new) - np.array(nearest_node.config))
            new_cost = nearest_node.cost + edge_cost

            # Find neighbors for rewiring
            radius = self.get_radius(len(self.nodes))
            neighbors = self.near(q_new, radius)

            # Find best parent among neighbors
            best_parent = nearest_node
            best_cost = new_cost

            for neighbor in neighbors:
                edge_cost = np.linalg.norm(np.array(q_new) - np.array(neighbor.config))
                candidate_cost = neighbor.cost + edge_cost
                if candidate_cost < best_cost:
                    if self.check_edge(neighbor.config, q_new):
                        best_parent = neighbor
                        best_cost = candidate_cost

            # Create new node
            new_node = self.OptimalNode(q_new, parent=best_parent, cost=best_cost)
            self.nodes.append(new_node)

            # Rewire neighbors through new node if beneficial
            for neighbor in neighbors:
                if neighbor is best_parent or neighbor is start_node:
                    continue
                edge_cost = np.linalg.norm(np.array(neighbor.config) - np.array(q_new))
                candidate_cost = new_node.cost + edge_cost
                if candidate_cost < neighbor.cost:
                    if self.check_edge(q_new, neighbor.config):
                        neighbor.rewire(new_node, candidate_cost)

            # Check if goal is reached
            dist_to_goal = np.linalg.norm(np.array(q_new) - np.array(goal))
            if dist_to_goal < EPSILON:
                if self.check_edge(q_new, goal):
                    total_cost = new_node.cost + dist_to_goal
                    if goal_node is None or total_cost < goal_node.cost + np.linalg.norm(
                            np.array(goal_node.config) - np.array(goal)):
                        goal_node = new_node
                        print(f"  RRT* found path with cost {total_cost:.3f} at iteration {it+1}")

            if it % 500 == 0 and it > 0:
                print(f"  RRT* iteration {it}/{max_iter}, tree size: {len(self.nodes)}")


        # Return path
        if goal_node is not None:
            path = goal_node.retrace()
            # Add actual goal if not already at goal
            if np.linalg.norm(np.array(path[-1]) - np.array(goal)) > 1e-6:
                path.append(goal)
            total_cost = goal_node.cost + np.linalg.norm(np.array(goal_node.config) - np.array(goal))
            print(f"RRT*: Best path found with {len(path)} waypoints, cost: {total_cost:.3f}")
            return path

        print("RRT*: No path found")
        return []

    def draw_tree(self):
        """Visualize the RRT* tree in PyBullet."""
        print(f"Drawing RRT* tree with {len(self.nodes)} nodes...")
        for node in self.nodes:
            if node.parent is not None:
                p1 = node.parent.config
                p2 = node.config
                p.addUserDebugLine([p1[0], p1[1], 0.1],
                                   [p2[0], p2[1], 0.1],
                                   [0.5, 0.5, 1], lineWidth=1, lifeTime=0)

    def draw_path(self, path):
        """Visualize the path in PyBullet."""
        p.removeAllUserDebugItems()
        for i in range(len(path) - 1):
            p.addUserDebugLine([path[i][0], path[i][1], 0.1],
                               [path[i+1][0], path[i+1][1], 0.1],
                               [0, 1, 0], lineWidth=3)


def get_planner(planner_type: str, bounds, robot_radius=0.35, robot_id=None) -> BasePlanner:
    """Factory function for planners."""
    if planner_type.lower() == 'prm':
        return PRMPlanner(bounds, robot_radius, robot_id=robot_id)
    elif planner_type.lower() == 'rrt':
        return RRTPlanner(bounds, robot_radius, robot_id=robot_id)
    elif planner_type.lower() == 'rrt*':
        return RRTStarPlanner(bounds, robot_radius, robot_id=robot_id)
    else:
        raise ValueError(f"Unknown planner: {planner_type}")


# ============================================================================
# PATH PROCESSING FUNCTIONS
# ============================================================================

def shortcut_path(path: List[Tuple], planner, max_iterations: int = 50) -> List[Tuple]:
    """
    Remove unnecessary intermediate waypoints via random shortcutting.
    
    Args:
        path: List of (x, y) waypoints
        planner: Planner with check_edge() method
        max_iterations: Number of shortcut attempts
    
    Returns:
        Shortened path (typically 50-80% fewer waypoints)
    """
    if len(path) < 3:
        return path
    
    shortened = list(path)
    
    for _ in range(max_iterations):
        if len(shortened) < 3:
            break
        
        # Pick two random waypoints
        i = random.randint(0, len(shortened) - 3)
        j = random.randint(i + 2, len(shortened) - 1)
        
        # Can we connect them directly?
        if planner.check_edge(shortened[i], shortened[j]):
            # Remove all intermediate waypoints
            shortened = shortened[:i+1] + shortened[j:]
    
    reduction = (1 - len(shortened) / len(path)) * 100
    print(f"  Path shortcut: {len(path)} → {len(shortened)} waypoints ({reduction:.1f}% reduction)")
    
    return shortened


def smooth_path(path: List[Tuple], num_points: int = 100) -> List[Tuple]:
    """
    Smooth 2D path using cubic spline interpolation.
    
    Converts sharp corners into smooth curves.
    
    Args:
        path: List of (x, y) waypoints
        num_points: Number of points in smooth path
    
    Returns:
        Smooth path as list of (x, y) tuples
    """
    from scipy.interpolate import CubicSpline
    
    if len(path) < 4:
        print(f"  Path too short for smoothing ({len(path)} points), using as-is")
        return path
    
    path_array = np.array(path)
    
    # Parameterize by cumulative distance
    distances = np.cumsum([0] + [
        np.linalg.norm(path_array[i+1] - path_array[i])
        for i in range(len(path_array) - 1)
    ])
    
    # Create cubic splines for x and y
    cs_x = CubicSpline(distances, path_array[:, 0])
    cs_y = CubicSpline(distances, path_array[:, 1])
    
    # Interpolate at uniform spacing
    s_smooth = np.linspace(0, distances[-1], num_points)
    x_smooth = cs_x(s_smooth)
    y_smooth = cs_y(s_smooth)
    
    smooth_path_array = np.column_stack([x_smooth, y_smooth])
    
    # Convert back to list of tuples
    smooth_path_list = [(x, y) for x, y in smooth_path_array]
    
    print(f"  Path smoothed: {len(path)} → {num_points} interpolated points")
    
    return smooth_path_list