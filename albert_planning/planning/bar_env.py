import numpy as np
import pybullet as p
import os
import random
from typing import List, Optional, Tuple
from urdfenvs.urdf_common.generic_robot import GenericRobot
from urdfenvs.urdf_common.urdf_env import UrdfEnv
from mpscenes.obstacles.box_obstacle import BoxObstacle

BOTTLE_POSES = [
    [-1.55, 1.0, 0.75], # Middle table
    [-1.9, -2.55, 0.75], # Left top table
    [-1.65, 5.65, 0.75], # Right top table
    [2.55, 6.75, 0.75], # Right down table
    [2.55, -4.75, 0.75], # Left bottom table
    [1.8, -2.25, 0.8], # Right side of barstool
    [2.2, 2.25, 0.8],  # Left side of barstool
]


class DynamicObstacle:
    """
    A dynamic obstacle (cylinder) that moves at constant velocity.
    Spawns in a free position, moves in a random direction,
    and bounces off static obstacles/walls.
    """

    def __init__(self, radius: float = 0.25, height: float = 1.8, speed: float = 0.4):
        self.radius = radius
        self.height = height
        self.speed = speed
        self.body_id: Optional[int] = None
        self.position: np.ndarray = np.zeros(2)  # (x, y)
        self.velocity: np.ndarray = np.zeros(2)  # (vx, vy)

    # Orientation for upright barstool (90 deg rotation around x-axis)
    ORIENTATION = None  # Will be set on first use

    def spawn(self, position: Tuple[float, float], body_id: int,
              direction: Optional[Tuple[float, float]] = None):
        """Activate a pre-created obstacle body at position."""
        self.position = np.array(position)
        self.body_id = body_id

        # Initialize orientation once
        if DynamicObstacle.ORIENTATION is None:
            DynamicObstacle.ORIENTATION = p.getQuaternionFromEuler([1.57, 0, 0])

        if direction is None:
            angle = random.uniform(0, 2 * np.pi)
            direction = (np.cos(angle), np.sin(angle))

        dir_vec = np.array(direction)
        dir_vec = dir_vec / (np.linalg.norm(dir_vec) + 1e-6)
        self.velocity = dir_vec * self.speed

        # Move pre-created body to position with correct orientation
        p.resetBasePositionAndOrientation(
            self.body_id,
            [self.position[0], self.position[1], 0.0],
            DynamicObstacle.ORIENTATION
        )

    def update(self, dt: float, bounds: Tuple[float, float, float, float], check_collision_fn) -> None:
        """Update position, bounce off walls and static obstacles."""
        if self.body_id is None:
            return

        x_min, x_max, y_min, y_max = bounds
        new_pos = self.position + self.velocity * dt

        # Bounce on walls
        if new_pos[0] - self.radius < x_min:
            new_pos[0] = x_min + self.radius
            self.velocity[0] = abs(self.velocity[0])
        elif new_pos[0] + self.radius > x_max:
            new_pos[0] = x_max - self.radius
            self.velocity[0] = -abs(self.velocity[0])

        if new_pos[1] - self.radius < y_min:
            new_pos[1] = y_min + self.radius
            self.velocity[1] = abs(self.velocity[1])
        elif new_pos[1] + self.radius > y_max:
            new_pos[1] = y_max - self.radius
            self.velocity[1] = -abs(self.velocity[1])

        # Bounce on static obstacles
        if check_collision_fn(new_pos[0], new_pos[1], self.radius, exclude_body=self.body_id):
            self.velocity = -self.velocity
        else:
            self.position = new_pos

        p.resetBasePositionAndOrientation(
            self.body_id,
            [self.position[0], self.position[1], 0.0],
            DynamicObstacle.ORIENTATION
        )

    def remove(self):
        """Remove obstacle from simulation."""
        if self.body_id is not None:
            p.removeBody(self.body_id)
            self.body_id = None

    def get_state(self) -> Tuple[np.ndarray, np.ndarray, float]:
        """Return (position, velocity, radius)."""
        return self.position.copy(), self.velocity.copy(), self.radius



class BarEnvironment(UrdfEnv):
    """
    Bar environment with integrated scenario setup.
    Extends UrdfEnv to include bar-specific furniture and layout.
    
    This environment includes:
    - Room walls (as tracked obstacles)
    - Bar counter (as tracked obstacle)
    - Barstools, tables, chairs, cabinets (as PyBullet bodies)
    """
    
    # --- Room and Bar Constants ---
    BAR_POS = [2.0, 0.0, 0.4]
    BAR_SIZE = [0.6, 5, 0.8]

    ROOM_SIZE_x = 20.0 
    ROOM_SIZE_y = 10.0 

    WALL_HEIGHT = 1.0
    WALL_THICKNESS = 0.1
    
    def __init__(
        self,
        robots: List[type(GenericRobot)],
        render: bool = False,
        enforce_real_time: Optional[bool] = None,
        dt: float = 0.01,
        num_sub_steps: int = 20,
        observation_checking: bool = True,
        # Bar-specific parameters
        bar_cabinet_urdf: str = "urdfenvs/bar_cabinet/bar_cabinet.urdf",
        barstool_urdf: str = "urdfenvs/barstool/barstool.urdf",
        bottle_urdf: str = "urdfenvs/bottle/bottle.urdf",
        bottle_pose_index: Optional[int] = None,
        chair_urdf_prefix: str = "urdfenvs/chair/chair_table",
        table_urdf_prefix: str = "urdfenvs/round_table/round_table",
        auto_setup_scene: bool = True,
        furniture_as_obstacles: bool = True  # NEW: Add furniture as MPC obstacles
    ) -> None:
        """
        Initialize Bar Environment.
        
        Parameters
        ----------
        robots : List[GenericRobot]
            List of robots to simulate
        render : bool
            Whether to render the simulation
        dt : float
            Time step for physics engine
        num_sub_steps : int
            Number of physics sub-steps per dt
        observation_checking : bool
            Whether to validate observations
        bar_cabinet_urdf : str
            Path to bar cabinet URDF file
        barstool_urdf : str
            Path to barstool URDF file
        bottle_urdf : str
            Path to bottle URDF file
        bottle_pose_index : int, optional
            Index into BOTTLE_POSES for where to place the bottle
        chair_urdf_prefix : str
            Prefix for chair URDF files (will append _{counter}.urdf)
        table_urdf_prefix : str
            Prefix for table URDF files (will append _{counter}.urdf)
        furniture_as_obstacles : bool
            If True, add simplified furniture as BoxObstacle for MPC collision avoidance.
            If False, furniture is only visual (loaded as PyBullet bodies).
            Default: False (visual only)
        """
        # Initialize parent UrdfEnv
        super().__init__(
            robots=robots,
            render=render,
            enforce_real_time=enforce_real_time,
            dt=dt,
            num_sub_steps=num_sub_steps,
            observation_checking=observation_checking
        )

        # Store URDF paths
        from pathlib import Path

        THIS_FILE = Path(__file__).resolve()
        PROJECT_ROOT = THIS_FILE.parents[1]  # planning/ -> albert_planning/ (1 livello sopra)

        self.barstool_urdf = str((PROJECT_ROOT / "urdfenvs/barstool/barstool.urdf").resolve())
        self.bar_cabinet_urdf = str((PROJECT_ROOT / "urdfenvs/bar_cabinet/bar_cabinet.urdf").resolve())
        self.bottle_pose_index = 0 if bottle_pose_index is None else int(bottle_pose_index)
        self.chair_urdf_prefix = str((PROJECT_ROOT / "urdfenvs/chair/chair_table").resolve())
        self.table_urdf_prefix = str((PROJECT_ROOT / "urdfenvs/round_table/round_table").resolve())
        self.bottle_urdf = str((PROJECT_ROOT / "urdfenvs/bottle/bottle.urdf").resolve())

        self.furniture_as_obstacles = furniture_as_obstacles
        
        # Counters for unique URDFs
        self.chair_counter = 1
        self.table_counter = 1
        
        # Track furniture body IDs for potential removal/reset
        self.furniture_bodies = {
            'barstools': [],
            'cabinets': [],
            'tables': [],
            'chairs': [],
            'bottles': [],
        }
        
        # Setup scene if requested
        if auto_setup_scene:
            self.setup_bar_scene()
    
    def setup_bar_scene(self) -> None:
        """
        Setup the complete bar scene with walls, bar, and furniture.
        This method can be called manually if auto_setup_scene=False.
        Can also be used to reload the scene after modifications.
        """
        print("=" * 50)
        print("Setting up Bar Environment...")
        print("=" * 50)
        
        self._add_walls_and_bar()
        
        # Add furniture as obstacles if requested (for MPC)
        if self.furniture_as_obstacles:
            self._add_furniture_obstacles()
        
        # Always load visual furniture
        self._load_furniture()

        # Pre-create dynamic obstacle bodies (hidden underground)
        self._precreate_dynamic_obstacles()

        print("=" * 50)
        print("Bar Environment setup complete!")
        print("=" * 50)
    
    def _add_walls_and_bar(self) -> None:
        """
        Add walls and bar counter as tracked obstacles using mpscenes BoxObstacle.
        These will be registered in the environment's obstacle dictionary
        for collision detection and are suitable for MPC planning.
        """
        obstacles = [] # initializing obstacle list

        wall_offset_horiz = self.ROOM_SIZE_x / 2.0
        wall_offset_vert = self.ROOM_SIZE_y / 2.0
        wall_size_horiz = [self.ROOM_SIZE_y, self.WALL_THICKNESS, self.WALL_HEIGHT]
        wall_size_vert = [self.WALL_THICKNESS, self.ROOM_SIZE_x, self.WALL_HEIGHT]
        
        # Bar counter obstacle
        
        bar_dict = {
            'type': 'box',
            'geometry': {
                'position': self.BAR_POS,
                'length': self.BAR_SIZE[0],
                'width': self.BAR_SIZE[1],
                'height': self.BAR_SIZE[2],
            },
            'rgba': [0.6, 0.4, 0.2, 1.0],
            'movable': False,
        }
        obstacles.append(BoxObstacle(name=f"bar_counter", content_dict=bar_dict))
        """
        # ------- Bar counter obstacle - divided in small squares ----
        bar_x, bar_y, bar_z = self.BAR_POS
        bar_len_x, bar_len_y, bar_height = self.BAR_SIZE # [0.6, 5.0, 0.8]
        
        # # We want roughly 0.5m segments along the Y axis
        # segment_size_y = 0.5
        # num_segments = int(bar_len_y / segment_size_y) 
        
        # # Calculate start Y position (bottom of the bar)
        # start_y = bar_y - (bar_len_y / 2) + (segment_size_y / 2)
        
        # for i in range(num_segments):
        #     # Calculate Y center for this segment
        #     current_y = start_y + (i * segment_size_y)
            
            bar_counter_dict = {
                'type': 'box',
                'geometry': {
                    'position': [bar_x, current_y, bar_z],
                    'length': bar_len_x,      # Keep original width (0.6)
                    'width': segment_size_y,  # 1.0m length
                    'height': bar_height,
                },
                'rgba': [0.6, 0.4, 0.2, 1.0],
                'movable': False,
            }
            # Add segment as a unique obstacle
            obstacles.append(BoxObstacle(name=f"bar_segment_{i}", content_dict=bar_counter_dict))
        """
        
        
        # Wall obstacle dictionaries
        wall_dicts = [
            # Top wall
            {
                'type': 'box',
                'geometry': {
                    'position': [0, wall_offset_horiz, self.WALL_HEIGHT/2],
                    'length': wall_size_horiz[0],
                    'width': wall_size_horiz[1],
                    'height': wall_size_horiz[2],
                },
                'rgba': [0.5, 0.5, 0.5, 1.0],
                'movable': False,
            },
            # Bottom wall
            {
                'type': 'box',
                'geometry': {
                    'position': [0, -wall_offset_horiz, self.WALL_HEIGHT/2],
                    'length': wall_size_horiz[0],
                    'width': wall_size_horiz[1],
                    'height': wall_size_horiz[2],
                },
                'rgba': [0.5, 0.5, 0.5, 1.0],
                'movable': False,
            },
            # Right wall
            {
                'type': 'box',
                'geometry': {
                    'position': [wall_offset_vert, 0, self.WALL_HEIGHT/2],
                    'length': wall_size_vert[0],
                    'width': wall_size_vert[1],
                    'height': wall_size_vert[2],
                },
                'rgba': [0.5, 0.5, 0.5, 1.0],
                'movable': False,
            },
            # Left wall
            {
                'type': 'box',
                'geometry': {
                    'position': [-wall_offset_vert, 0, self.WALL_HEIGHT/2],
                    'length': wall_size_vert[0],
                    'width': wall_size_vert[1],
                    'height': wall_size_vert[2],
                },
                'rgba': [0.5, 0.5, 0.5, 1.0],
                'movable': False,
            },
        ]
        
        # Add walls to obstacle list
        for i, wall_dict in enumerate(wall_dicts):
            wall_names = ["wall_top", "wall_bottom", "wall_right", "wall_left"]
            obstacles.append(BoxObstacle(name=wall_names[i], content_dict=wall_dict))
        
        # Add all obstacles to environment
        for obstacle in obstacles:
            self.add_obstacle(obstacle)
        
        print(f"✓ Added {len(obstacles)} structural obstacles (walls + bar)")

    def _get_table_configs(self) -> List[dict]:
        """Shared table and chair layout config for visuals and MPC obstacles."""
        return [
            {'pos': [-2.0, 1.0, 0.0], 'skip_chairs': [0, 1, 2, 3]},
            {'pos': [-2.0, -3.0, 0.0], 'skip_chairs': [0, 2, 3]},
            {'pos': [-2.0, 6.0, 0.0], 'skip_chairs': []},
            {'pos': [3.0, 7.0, 0.0], 'skip_chairs': []},
            {'pos': [3.0, -5.0, 0.0], 'skip_chairs': []}
        ]
    
    def _add_furniture_obstacles(self) -> None:
        """
        Add simplified furniture as box obstacles for MPC collision avoidance.
        
        This creates bounding box approximations of furniture that the MPC
        controller can use for collision-free trajectory planning.
        
        Note: Visual furniture is still loaded separately for rendering.
        """
        from mpscenes.obstacles.cylinder_obstacle import CylinderObstacle
        
        furniture_obstacles = []
        
        # === BARSTOOLS (as small boxes) ===
        # barstool_positions = [0.0, 1.0, -1.0, 2.0, -2.0]
        barstool_positions = [-1.0, 0.0, 1.0]
        for y in barstool_positions:
            stool_dict = {
                'type': 'box',
                'geometry': {
                    'position': [1.3, y, 0.3],
                    'width': 0.4,
                    'height': 0.6,
                    'length': 0.4,
                },
                'rgba': [0.4, 0.3, 0.2, 0.3],  # Semi-transparent
                'movable': False,
            }
            furniture_obstacles.append(
                BoxObstacle(name=f"obstacle_barstool_{y}", content_dict=stool_dict)
            )
        
        # === TABLES (as cylinders for round tables) ===
        table_configs = self._get_table_configs()
        for i, config in enumerate(table_configs):
            pos = config['pos']

            # Use cylinder for round tables (more accurate)
            table_dict = {
                'type': 'cylinder',
                'geometry': {
                    'position': [pos[0], pos[1], 0.4],
                    'radius': 0.5,   # Table radius
                    'height': 0.8,   # Table height
                },
                'rgba': [0.5, 0.4, 0.3, 0.3],
                'movable': False,
            }
            furniture_obstacles.append(
                CylinderObstacle(name=f"obstacle_table_{i}", content_dict=table_dict)
            )
        
        # === CHAIRS (simplified as small boxes) ===
        # Define chairs around each table
        chair_offset = 0.7  # Distance from table center
        chair_configs = [
            ([0.0, chair_offset], 0.0),      # Top
            ([0.0, -chair_offset], 180.0),   # Bottom
            ([-chair_offset, 0.0], 90.0),    # Left
            ([chair_offset, 0.0], -90.0)     # Right
        ]
        
        chair_counter = 0
        for table_idx, config in enumerate(table_configs):
            table_pos = config['pos']
            skip_chairs = config.get('skip_chairs', [])
            for chair_idx, (offset, angle) in enumerate(chair_configs):
                if chair_idx in skip_chairs:
                    continue

                chair_dict = {
                    'type': 'box',
                    'geometry': {
                        'position': [table_pos[0] + offset[0], 
                                   table_pos[1] + offset[1], 
                                   0.25],
                        'width': 0.4,
                        'height': 0.5,
                        'length': 0.4,
                    },
                    'rgba': [0.3, 0.25, 0.2, 0.3],
                    'movable': False,
                }
                furniture_obstacles.append(
                    BoxObstacle(name=f"obstacle_chair_{chair_counter}", 
                              content_dict=chair_dict)
                )
                chair_counter += 1
        
        # === CABINETS ===
        # cabinet_positions = [0.0, 1.0, -1.0]
        cabinet_positions = [0.0]
        for y in cabinet_positions:
            cabinet_dict = {
                'type': 'box',
                'geometry': {
                    'position': [4.65, y, 0.5],
                    'width': 0.5,
                    'height': 1.0,
                    'length': 0.6,
                },
                'rgba': [0.3, 0.25, 0.2, 0.3],
                'movable': False,
            }
            furniture_obstacles.append(
                BoxObstacle(name=f"obstacle_cabinet_{y}", content_dict=cabinet_dict)
            )
        
        # Add all furniture obstacles to environment
        for obstacle in furniture_obstacles:
            self.add_obstacle(obstacle)
        
        print(f"✓ Added {len(furniture_obstacles)} furniture obstacles for MPC")
        print(f"  - {len(barstool_positions)} barstools")
        print(f"  - {len(table_configs)} tables")
        print(f"  - {chair_counter} chairs")
        print(f"  - {len(cabinet_positions)} cabinets")
    
    def _load_furniture(self) -> None:
        """
        Load all furniture (barstools, cabinets, tables, chairs).
        These are loaded directly as PyBullet bodies since they're
        purely visual/physics objects without need for obstacle tracking.
        """
        self._load_barstools()
        self._load_cabinets()
        self._load_table_groups()
        self._load_bottles()

    
    def _load_barstools(self) -> None:
        """Load barstools along the bar counter."""
        try:
            # stool_positions = [0.0, 1.0, -1.0, 2.0, -2.0]
            stool_positions = [-1.0, 0.0, 1.0]
            for y in stool_positions:
                body_id = p.loadURDF(
                    self.barstool_urdf, 
                    [1.3, y, 0.0], 
                    p.getQuaternionFromEuler([1.57, 0, 0]), 
                    useFixedBase=True
                )
                self.furniture_bodies['barstools'].append(body_id)
            
            print(f"✓ Loaded {len(stool_positions)} barstools")
        except Exception as e:
            print(f"✗ Error loading barstools: {e}")
    
    def _load_cabinets(self) -> None:
        """Load storage cabinets behind the bar."""
        try:
            # cabinet_positions = [0.0, 1.0, -1.0]
            cabinet_positions = [0.0]
            for y in cabinet_positions:
                body_id = p.loadURDF(
                    self.bar_cabinet_urdf, 
                    [4.65, y, 0.0], 
                    p.getQuaternionFromEuler([1.57, 0, -1.57]), 
                    useFixedBase=True
                )
                self.furniture_bodies['cabinets'].append(body_id)
            
            print(f"✓ Loaded {len(cabinet_positions)} cabinets")
        except Exception as e:
            print(f"✗ Error loading cabinets: {e}")
    
    def _load_table_groups(self) -> None:
        """
        Load all table groups with surrounding chairs.
        Each table has up to 4 chairs (top, bottom, left, right).
        """
        table_configs = self._get_table_configs()

        
        for config in table_configs:
            self._load_table_group(**config)
        
        print(f"✓ Loaded {len(table_configs)} table groups")
    
    def _load_table_group(self, pos: List[float], skip_chairs: List[int] = None) -> None:
        """
        Load a table with surrounding chairs.
        
        Parameters
        ----------
        pos : List[float]
            Table position [x, y, z]
        skip_chairs : List[int], optional
            Indices of chairs to skip loading:
            0 = top, 1 = bottom, 2 = left, 3 = right
        """
        if skip_chairs is None:
            skip_chairs = []
        
        try:
            # Load table
            table_urdf = f"{self.table_urdf_prefix}_{self.table_counter}.urdf"
            self.table_counter += 1
            
            table_id = p.loadURDF(
                table_urdf, 
                pos, 
                p.getQuaternionFromEuler([1.57, 0, 0]), 
                useFixedBase=True
            )
            self.furniture_bodies['tables'].append(table_id)
            
            # Chair configurations: (offset_from_table, euler_angles)
            chair_configs = [
                ([0.0, 0.5, 0.0], [1.57, 0, 0]),           # 0: Top
                ([0.0, -0.5, 0.0], [1.57, 0, 3.14159]),    # 1: Bottom
                ([-0.5, 0.0, 0.0], [1.57, 0, 1.57]),       # 2: Left
                ([0.5, 0.0, 0.0], [1.57, 0, -1.57])        # 3: Right
            ]
            
            tx, ty, tz = pos
            
            # Load chairs
            for i, (offset, euler) in enumerate(chair_configs):
                chair_urdf = f"{self.chair_urdf_prefix}_{self.chair_counter}.urdf"
                self.chair_counter += 1
                
                if i not in skip_chairs:
                    cx, cy, cz = offset
                    chair_id = p.loadURDF(
                        chair_urdf,
                        [tx + cx, ty + cy, tz + cz],
                        p.getQuaternionFromEuler(euler),
                        useFixedBase=True
                    )
                    self.furniture_bodies['chairs'].append(chair_id)
        
        except Exception as e:
            print(f"✗ Error loading table group at {pos}: {e}")


    def _load_bottles(self) -> None:
        """Load bottles on top of a table."""
        try:
            bottle_pos = BOTTLE_POSES[0]
            if 0 <= self.bottle_pose_index < len(BOTTLE_POSES):
                bottle_pos = BOTTLE_POSES[self.bottle_pose_index]
            else:
                print(f"⚠ Invalid bottle_pose_index {self.bottle_pose_index}, using 0")
            bottle_id = p.loadURDF(
                self.bottle_urdf,
                bottle_pos,
                p.getQuaternionFromEuler([0, 0, 0]),
                useFixedBase=False,
            )
            self.furniture_bodies['bottles'].append(bottle_id)
            self._disable_body_collisions_with_obstacles(bottle_id)
            print("✓ Loaded 1 bottle")
        except Exception as e:
            print(f"✗ Error loading bottle: {e}")
    
    def _disable_body_collisions_with_obstacles(self, body_id: int) -> None:
        """Disable collisions between a body and all tracked obstacles."""
        obstacles = self.get_obstacles()
        if not obstacles:
            return

        num_joints = p.getNumJoints(body_id)
        body_links = [-1] + list(range(num_joints))
        for obst_id in obstacles.keys():
            for link_idx in body_links:
                p.setCollisionFilterPair(body_id, obst_id, link_idx, -1, 0)


    def clear_furniture(self) -> None:
        """
        Remove all furniture from the scene.
        Useful for resetting or reconfiguring the environment.
        """
        for category, body_ids in self.furniture_bodies.items():
            for body_id in body_ids:
                try:
                    p.removeBody(body_id)
                except:
                    pass
            body_ids.clear()
        
        # Reset counters
        self.chair_counter = 1
        self.table_counter = 1
        
        print("✓ Cleared all furniture")
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
        pos: Optional[np.ndarray] = None,
        vel: Optional[np.ndarray] = None,
        mount_positions: Optional[np.ndarray] = None,
        mount_orientations: Optional[np.ndarray] = None,
        reload_scene: bool = False
    ) -> tuple:
        """
        Reset the environment.
        
        Parameters
        ----------
        seed : int, optional
            Random seed for reproducibility
        options : dict, optional
            Additional reset options
        pos : np.ndarray, optional
            Initial robot joint positions
        vel : np.ndarray, optional
            Initial robot joint velocities
        mount_positions : np.ndarray, optional
            Robot mounting positions
        mount_orientations : np.ndarray, optional
            Robot mounting orientations
        reload_scene : bool
            If True, clear and reload all furniture (useful if scene was modified)
        
        Returns
        -------
        tuple
            (observation, info) - Initial observation and info dict
        """
        # Reset parent environment
        result = super().reset(seed, options, pos, vel, mount_positions, mount_orientations)
        
        # Optionally reload the entire scene
        if reload_scene:
            self.clear_furniture()
            self.setup_bar_scene()
        
        return result
    
    def get_furniture_count(self) -> dict:
        """
        Get count of furniture items in the scene.

        Returns
        -------
        dict
            Dictionary with counts for each furniture category
        """
        return {

            category: len(body_ids)
            for category, body_ids in self.furniture_bodies.items()
        }

    # =========================================================================
    # DYNAMIC OBSTACLES
    # =========================================================================

    MAX_DYNAMIC_OBSTACLES = 5  # Pre-create this many obstacle bodies

    def _precreate_dynamic_obstacles(self) -> None:
        """Pre-create dynamic obstacle bodies (hidden underground) during init."""
        from pathlib import Path
        THIS_FILE = Path(__file__).resolve()
        PROJECT_ROOT = THIS_FILE.parents[1]
        # Use barstool URDF for now (known to render correctly)
        urdf_path = str((PROJECT_ROOT / "urdfenvs/barstool/barstool.urdf").resolve())

        self._precreated_obstacle_bodies: List[int] = []
        self._precreated_obstacles_used: List[bool] = []

        # Correct orientation for barstools (rotated 90 deg around x)
        orientation = p.getQuaternionFromEuler([1.57, 0, 0])

        for i in range(self.MAX_DYNAMIC_OBSTACLES):
            body_id = p.loadURDF(
                urdf_path,
                [0, 0, -100],  # Hidden underground
                orientation,
                useFixedBase=False
            )
            self._precreated_obstacle_bodies.append(body_id)
            self._precreated_obstacles_used.append(False)

        print(f"✓ Pre-created {self.MAX_DYNAMIC_OBSTACLES} dynamic obstacle bodies")

    def spawn_dynamic_obstacles(self, n_obstacles: int = 3, speed: float = 0.4,
                                 radius: float = 0.25, robot_id: Optional[int] = None) -> None:
        """
        Spawn dynamic obstacles in free positions.

        Args:
            n_obstacles: Number of obstacles to spawn
            speed: Movement speed [m/s]
            radius: Obstacle radius [m]
            robot_id: Robot body ID to avoid when spawning
        """
        self.dynamic_obstacles: List[DynamicObstacle] = []
        self._dynamic_obs_robot_id = robot_id

        # World bounds (with margin)
        margin = 1.0
        self._dynamic_obs_bounds = (
            -self.ROOM_SIZE_y / 2 + margin,
            self.ROOM_SIZE_y / 2 - margin,
            -self.ROOM_SIZE_x / 2 + margin,
            self.ROOM_SIZE_x / 2 - margin
        )

        n_obstacles = min(n_obstacles, self.MAX_DYNAMIC_OBSTACLES)
        print(f"\n=== Spawning {n_obstacles} dynamic obstacles ===")

        spawned = 0
        for i in range(n_obstacles):
            if spawned >= len(self._precreated_obstacle_bodies):
                print(f"  No more pre-created bodies available")
                break

            pos = self._find_free_position(radius, robot_id)
            if pos is not None:
                # Get a pre-created body
                body_id = self._precreated_obstacle_bodies[spawned]

                obs = DynamicObstacle(radius=radius, height=1.8, speed=speed)
                obs.spawn(pos, body_id)
                self.dynamic_obstacles.append(obs)
                spawned += 1

                print(f"  Obstacle {i+1}: pos=({pos[0]:.2f}, {pos[1]:.2f}), "
                      f"vel=({obs.velocity[0]:.2f}, {obs.velocity[1]:.2f})")
            else:
                print(f"  Obstacle {i+1}: Could not find free position, skipped")

        print(f"  Total spawned: {len(self.dynamic_obstacles)}")

    def _find_free_position(self, radius: float, robot_id: Optional[int],
                            max_attempts: int = 100) -> Optional[Tuple[float, float]]:
        """Find a free position for spawning."""
        x_min, x_max, y_min, y_max = self._dynamic_obs_bounds

        for _ in range(max_attempts):
            x = random.uniform(x_min, x_max)
            y = random.uniform(y_min, y_max)
            if not self._check_position_collision(x, y, radius, robot_id):
                return (x, y)
        return None

    def _check_position_collision(self, x: float, y: float, radius: float,
                                   exclude_body: Optional[int] = None) -> bool:
        """Check if position collides with static obstacles."""
        aabb_min = [x - radius, y - radius, 0.1]
        aabb_max = [x + radius, y + radius, 1.5]

        overlaps = p.getOverlappingObjects(aabb_min, aabb_max)
        if overlaps:
            for body_id, _ in overlaps:
                if exclude_body is not None and body_id == exclude_body:
                    continue
                if self._dynamic_obs_robot_id is not None and body_id == self._dynamic_obs_robot_id:
                    continue
                try:
                    body_name = p.getBodyInfo(body_id)[1].decode('utf-8')
                    if "floor" not in body_name.lower() and "plane" not in body_name.lower():
                        return True
                except:
                    return True
        return False

    def update_dynamic_obstacles(self) -> None:
        """Update all dynamic obstacles (call each simulation step)."""
        if not hasattr(self, 'dynamic_obstacles') or not self.dynamic_obstacles:
            return

        for obs in self.dynamic_obstacles:
            obs.update(self._dt, self._dynamic_obs_bounds, self._check_position_collision)

    def get_dynamic_obstacles_state(self) -> List[Tuple[np.ndarray, np.ndarray, float]]:
        """
        Get state of all dynamic obstacles for MPC.

        Returns:
            List of (position, velocity, radius) tuples
        """
        if not hasattr(self, 'dynamic_obstacles') or not self.dynamic_obstacles:
            return []
        return [obs.get_state() for obs in self.dynamic_obstacles]

    def remove_dynamic_obstacles(self) -> None:
        """Remove all dynamic obstacles."""
        if hasattr(self, 'dynamic_obstacles'):
            for obs in self.dynamic_obstacles:
                obs.remove()
            self.dynamic_obstacles = []
            print("Removed all dynamic obstacles")

    def step(self, action):
        """Override step to update dynamic obstacles automatically."""
        self.update_dynamic_obstacles()
        return super().step(action)