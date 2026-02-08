import numpy as np
import pybullet as p

# --- Constants ---
BAR_POS = [3.0, 0.0, 0.4]
BAR_SIZE = [0.6, 5, 0.8]
ROOM_SIZE_HORIZ = 20.0 
ROOM_SIZE_VERT = 10.0 
WALL_HEIGHT = 1.0
WALL_THICKNESS = 0.1
WALL_OFFSET_HORIZ = ROOM_SIZE_HORIZ / 2.0   
WALL_OFFSET_VERT = ROOM_SIZE_VERT / 2.0   
WALL_SIZE_HORIZ = [ROOM_SIZE_VERT, WALL_THICKNESS, WALL_HEIGHT]
WALL_SIZE_VERT = [WALL_THICKNESS, ROOM_SIZE_HORIZ, WALL_HEIGHT]

# --- BoxObstacle Class ---
class BoxObstacle:
    def __init__(self, name, pos, size, ori=None, rgba=None):
        self._name = name
        self._pos = np.array(pos)
        self._size = np.array(size)
        self._ori = np.array(ori) if ori is not None else np.array([0.0, 0.0, 0.0, 1.0])
        self._rgba = np.array(rgba) if rgba is not None else np.array([0.5, 0.5, 0.5, 1.0])

    def type(self): return "box"
    def position(self, t=None): return self._pos
    def velocity(self, t=None): return np.zeros(6)
    def orientation(self, t=None): return self._ori
    def size(self): return self._size   
    def rgba(self): return self._rgba
    def movable(self): return False 
    def urdf(self): return None 
    def name(self): return self._name

# --- Main Scenario Class ---
class BarClassScenario:
    def __init__(self):
        # Counters for unique URDFs
        self.chair_counter = 1
        self.table_counter = 1
        
        # Paths
        self.bar_cabinet_urdf = "urdfenvs/bar_cabinet/bar_cabinet.urdf"
        self.barstool_urdf = "urdfenvs/barstool/barstool.urdf"

    def get_next_chair_urdf(self):
        path = f"urdfenvs/chair/chair_table_{self.chair_counter}.urdf"
        self.chair_counter += 1
        return path

    def get_next_table_urdf(self):
        path = f"urdfenvs/round_table/round_table_{self.table_counter}.urdf"
        self.table_counter += 1
        return path

    def setup(self, env):
        """
        Called by UrdfEnv to load the scenario.
        :param env: The UrdfEnv instance
        """
        print("Setting up Bar Scenario...")
        # 1. Add Walls and Static Obstacles (Tracked by Env)
        self._add_static_obstacles(env)
        
        # 2. Load Furniture (Visual/Physics only, passed directly to PyBullet)
        self._load_furniture()

    def _add_static_obstacles(self, env):
        obstacles = []
        obstacles.append(BoxObstacle("bar_table", BAR_POS, BAR_SIZE, rgba=[0.6, 0.4, 0.2, 1.0]))
        obstacles.append(BoxObstacle("wall_top", [0, WALL_OFFSET_HORIZ, WALL_HEIGHT/2], WALL_SIZE_HORIZ, rgba=[0.5, 0.5, 0.5, 1.0]))
        obstacles.append(BoxObstacle("wall_bottom", [0, -WALL_OFFSET_HORIZ, WALL_HEIGHT/2], WALL_SIZE_HORIZ, rgba=[0.5, 0.5, 0.5, 1.0]))
        obstacles.append(BoxObstacle("right_wall", [WALL_OFFSET_VERT, 0, WALL_HEIGHT/2], WALL_SIZE_VERT, rgba=[0.5, 0.5, 0.5, 1.0]))
        obstacles.append(BoxObstacle("left_wall", [-WALL_OFFSET_VERT, 0, WALL_HEIGHT/2], WALL_SIZE_VERT, rgba=[0.5, 0.5, 0.5, 1.0]))

        for obst in obstacles:
            env.add_obstacle(obst)

    def _load_furniture(self):
        # --- Barstools ---
        try:
            for y in [0.0, 1.0, -1.0, 2.0, -2.0]:
                p.loadURDF(self.barstool_urdf, [2.3, y, 0.0], p.getQuaternionFromEuler([1.57, 0, 0]), useFixedBase=True)
            print("Barstools loaded.")
        except Exception as e: print(f"Error loading barstools: {e}")

        # --- Cabinets ---
        try:
            for y in [0.0, 1.0, -1.0]:
                p.loadURDF(self.bar_cabinet_urdf, [4.65, y, 0.0], p.getQuaternionFromEuler([1.57, 0, -1.57]), useFixedBase=True)
            print("Cabinets loaded.")
        except Exception as e: print(f"Error loading cabinets: {e}")

        # --- Tables and Chairs ---
        # Configuration: (x, y, z_offset, has_chairs)
        # We manually list them to match your specific layout
        
        # Table 1 (x,y,z)
        self._load_table_group([-2.0, 1.0, 0.0])
        # Table 2 
        # Example logic to skip specific chairs if needed use: self._load_table_group([-2.0, -3.0, 0.0], skip_chairs=[0, 1])
        self._load_table_group([-2.0, -3.0, 0.0]) 
        # Table 3
        self._load_table_group([-2.0, 6.0, 0.0])
        # Table 4
        self._load_table_group([3.0, 7.0, 0.0])
        # Table 5
        self._load_table_group([3.0, -5.0, 0.0])

    def _load_table_group(self, pos, skip_chairs=[]):
        """
        Helper to load a table and its 4 chairs
        pos: [x, y, z]
        skip_chairs: list of indices (0-3) to skip loading
        """
        try:
            # Load Table
            p.loadURDF(self.get_next_table_urdf(), pos, p.getQuaternionFromEuler([1.57, 0, 0]), useFixedBase=True)
            
            # Chair offsets relative to table center
            # 0: Top, 1: Bottom, 2: Left, 3: Right
            chair_defs = [
                ([0.0, 0.5, 0.0], [1.57, 0, 0]),       # Top
                ([0.0, -0.5, 0.0], [1.57, 0, 3.14159]),# Bottom
                ([-0.5, 0.0, 0.0], [1.57, 0, 1.57]),   # Left
                ([0.5, 0.0, 0.0], [1.57, 0, -1.57])    # Right
            ]

            tx, ty, tz = pos
            
            for i, (offset, euler) in enumerate(chair_defs):
                # Always increment counter to keep file sync, but only load if not skipped
                # (Or you can skip incrementing if you want to save file usage, 
                # but unique files usually map 1-to-1 with spawn attempts).
                # Here we just load if not in skip list.
                urdf_path = self.get_next_chair_urdf()
                
                if i not in skip_chairs:
                    cx, cy, cz = offset
                    p.loadURDF(urdf_path, [tx+cx, ty+cy, tz+cz], p.getQuaternionFromEuler(euler), useFixedBase=True)
            
        except Exception as e:
            print(f"Error loading table group at {pos}: {e}")