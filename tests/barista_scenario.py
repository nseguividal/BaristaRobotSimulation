import warnings
import gymnasium as gym
import numpy as np
import pybullet as p
import time

from urdfenvs.robots.generic_urdf.generic_diff_drive_robot import GenericDiffDriveRobot
from urdfenvs.urdf_common.urdf_env import UrdfEnv


# --- Constants for the Scenario ---

# 1. Robot Start Configuration
# Base: x=0.0, y=1.0, theta=0.0
# Arm: Standard "tucked" or "ready" position
ROBOT_START_POS = [0.0, 1.0, 0.0] 
ROBOT_START_ARM = [0.0, 0.0, 0.0, -1.5, 0.0, 1.8, 0.5] 

# 2. Map Layout Dimensions
# Note: PyBullet box positions are defined by their CENTER, not the corner.
# Height (z) has to be set to half the object's height so it sits on the floor (z=0).

# The Bar (Pickup Location)
BAR_POS = [3.0, 0.0, 0.4]    # Located at x=1.5, directly in front of the robot's start y=1.0m and z=0.4 means the table is 0.8m tall
BAR_SIZE = [0.6, 5, 0.8]   # Width (x), Depth (y), Height (z)

# The Customer (Dropoff Location)
TABLE_SIZE_SMALL = [0.8, 1.0, 0.8]
TABLE_SIZE_MEDIUM = [0.8, 1.5, 0.8]
TABLE_SIZE_LARGE = [0.8, 2.0, 0.8]

# The Walls
WALL_POS = [0.8, -0.5, 0.5]  # A 1m tall wall

# Room Constraints 
ROOM_SIZE_HORIZ = 20.0 # 20x20 meter room
ROOM_SIZE_VERT = 10.0 # 20x20 meter room
WALL_HEIGHT = 1.0
WALL_THICKNESS = 0.1
WALL_OFFSET_HORIZ = ROOM_SIZE_HORIZ / 2.0   # Positions: +/- 10 meters from center
WALL_OFFSET_VERT = ROOM_SIZE_VERT / 2.0   # Positions: +/- 5 meters from center


# Top/Bottom: Long in X, Thin in Y
WALL_SIZE_HORIZ = [ROOM_SIZE_VERT, WALL_THICKNESS, WALL_HEIGHT]

# Left/Right: Thin in X, Long in Y
WALL_SIZE_VERT = [WALL_THICKNESS, ROOM_SIZE_HORIZ, WALL_HEIGHT]








# --- BoxObstacle Class ---
# Define class
class BoxObstacle:
    def __init__(self, name, pos, size, ori=None, rgba=None):
        self._name = name
        # Convert inputs to NumPy arrays to satisfy urdf_env
        self._pos = np.array(pos)
        self._size = np.array(size)
        # Default orientation: Identity Quaternion [x, y, z, w] -> [0, 0, 0, 1]
        self._ori = np.array([0.0, 0.0, 0.0, 1.0])
        
        #ORIENTATION
        if ori is not None:
            self._ori = np.array(ori)
        else:
            self._ori = np.array([0.0, 0.0, 0.0, 1.0]) # [x, y, z, w]

        # COLOUR
        if rgba is not None:
            self._rgba = np.array(rgba)
        else:
            self._rgba = np.array([0.5, 0.5, 0.5, 1.0])

    def type(self): return "box"
    def position(self, t=None): return self._pos
    def velocity(self, t=None): return np.zeros(6)  # Returns 6D velocity (linear_x, _y, _z, angular_x, _y, _z)
    def orientation(self, t=None): return self._ori
    def size(self): return self._size   
    def rgba(self): return self._rgba
    def movable(self): return False    #Fixed = static obstacle 


def get_static_obstacles():
    """
    Returns a list of BoxObstacle instances.
    """
    obstacles = []
    
    # 1. The Bar Table (Brown)
    obstacles.append(BoxObstacle(
        name="bar_table",
        pos=BAR_POS,
        size=BAR_SIZE,
        rgba=[0.6, 0.4, 0.2, 1.0] 
    ))
    

    # 3.1. Top Wall (up)
    obstacles.append(BoxObstacle(
        name="wall_top",
        pos=[0, WALL_OFFSET_HORIZ, WALL_HEIGHT/2],
        size=WALL_SIZE_HORIZ, # Use horizontal size
        rgba=[0.5, 0.5, 0.5, 1.0]    # rgba=[0.8, 0.8, 0.8, 1.0]  (lighter gray)
    ))

    # 3.2. Bottom Wall (down)
    obstacles.append(BoxObstacle(
        name="wall_bottom",
        pos=[0, -WALL_OFFSET_HORIZ, WALL_HEIGHT/2],
        size=WALL_SIZE_HORIZ, # Use horizontal size
        rgba=[0.5, 0.5, 0.5, 1.0] 
    ))

    # 3.3. Right Wall (right)
    obstacles.append(BoxObstacle(
        name="right_wall",
        pos=[WALL_OFFSET_VERT, 0, WALL_HEIGHT/2],
        size=WALL_SIZE_VERT, # Use vertical size
        rgba=[0.5, 0.5, 0.5, 1.0] 
    ))

    # 3.4. Left Wall (left)
    obstacles.append(BoxObstacle(
        name="left_wall",
        pos=[-WALL_OFFSET_VERT, 0, WALL_HEIGHT/2],
        size=WALL_SIZE_VERT, # Use vertical size 
        rgba=[0.5, 0.5, 0.5, 1.0] 
    ))
    
    return obstacles



def run_barista_scenario(n_steps=1000000, render=True):
    # --- 1. Define the Robot ---
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
        ),
    ]

    # --- 2. Initialize Environment ---
    env: UrdfEnv = UrdfEnv(dt=0.01, robots=robots, render=render)
    
    # --- 3. Add Static Obstacles ---
    # We add them to the environment definition before resetting
    for obst in get_static_obstacles():
        env.add_obstacle(obst)

    # --- 4. Reset (Spawn everything) ---
    # Concatenate base pose and arm pose
    full_start_pose = np.array(ROBOT_START_POS + ROBOT_START_ARM)
    ob = env.reset(pos=full_start_pose)
    

    # --- 5. Load the Objects URDF manually to add then to the scene ---
    # Define paths to each object
    barstool_urdf_path = "urdfenvs/barstool/barstool.urdf"
    bar_cabinet_urdf_path = "urdfenvs/bar_cabinet/bar_cabinet.urdf"
    round_table_1_urdf_path = "urdfenvs/round_table/round_table_1.urdf"
    round_table_2_urdf_path = "urdfenvs/round_table/round_table_2.urdf"
    round_table_3_urdf_path = "urdfenvs/round_table/round_table_3.urdf"
    round_table_4_urdf_path = "urdfenvs/round_table/round_table_4.urdf"
    round_table_5_urdf_path = "urdfenvs/round_table/round_table_5.urdf"
    chair_table_1_urdf_path = "urdfenvs/chair/chair_table_1.urdf"
    chair_table_2_urdf_path = "urdfenvs/chair/chair_table_2.urdf"
    chair_table_3_urdf_path = "urdfenvs/chair/chair_table_3.urdf"
    chair_table_4_urdf_path = "urdfenvs/chair/chair_table_4.urdf"
    chair_table_5_urdf_path = "urdfenvs/chair/chair_table_5.urdf"

    try:
        # Barstool 1
        p.loadURDF(
            barstool_urdf_path, 
            basePosition=[2.3, 0.0, 0.0], # x, y, z (z=0 for floor)
            baseOrientation=p.getQuaternionFromEuler([1.57, 0, 0]), # Rotate 90 degrees
            useFixedBase=True # Make it static so the robot can crash into it without it flying away
        )
        # Barstool 2
        p.loadURDF(
            barstool_urdf_path, 
            basePosition=[2.3, 1.0, 0.0], # x, y, z (z=0 for floor)
            baseOrientation=p.getQuaternionFromEuler([1.57, 0, 0]), # Rotate 90 degrees
            useFixedBase=True # Make it static so the robot can crash into it without it flying away
        )
        # Barstool 3
        p.loadURDF(
            barstool_urdf_path, 
            basePosition=[2.3, -1.0, 0.0], # x, y, z (z=0 for floor)
            baseOrientation=p.getQuaternionFromEuler([1.57, 0, 0]), # Rotate 90 degrees
            useFixedBase=True # Make it static so the robot can crash into it without it flying away
        )
        # Barstool 4
        p.loadURDF(
            barstool_urdf_path, 
            basePosition=[2.3, 2.0, 0.0], # x, y, z (z=0 for floor)
            baseOrientation=p.getQuaternionFromEuler([1.57, 0, 0]), # Rotate 90 degrees
            useFixedBase=True # Make it static so the robot can crash into it without it flying away
        )
        # Barstool 5
        p.loadURDF(
            barstool_urdf_path, 
            basePosition=[2.3, -2.0, 0.0], # x, y, z (z=0 for floor)
            baseOrientation=p.getQuaternionFromEuler([1.57, 0, 0]), # Rotate 90 degrees
            useFixedBase=True # Make it static so the robot can crash into it without it flying away
        )
        print("Barstools loaded successfully.")
    except Exception as e:
        print(f"Could not load barstools. Check path: {barstool_urdf_path}. Error: {e}")
    
    try:
        # Bar Cabinet 1
        p.loadURDF(
            bar_cabinet_urdf_path, 
            basePosition=[4.65, 0.0, 0.0], # x, y, z (z=0 for floor)
            baseOrientation=p.getQuaternionFromEuler([1.57, 0, -1.57]), # Rotate 90 degrees
            useFixedBase=True # Make it static so the robot can crash into it without it flying away
        )
        # Bar Cabinet 2
        p.loadURDF(
            bar_cabinet_urdf_path, 
            basePosition=[4.65, 1.0, 0.0], # x, y, z (z=0 for floor)
            baseOrientation=p.getQuaternionFromEuler([1.57, 0, -1.57]), # Rotate 90 degrees
            useFixedBase=True # Make it static so the robot can crash into it without it flying away
        )
        # Bar Cabinet 3
        p.loadURDF(
            bar_cabinet_urdf_path, 
            basePosition=[4.65, -1.0, 0.0], # x, y, z (z=0 for floor)
            baseOrientation=p.getQuaternionFromEuler([1.57, 0, -1.57]), # Rotate 90 degrees
            useFixedBase=True # Make it static so the robot can crash into it without it flying away
        )
        print("Bar Cabinet loaded successfully.")
    except Exception as e:
        print(f"Could not load bar_cabinet. Check path: {bar_cabinet_urdf_path}. Error: {e}")

    try:
        # Cafe Table 1
        p.loadURDF(
            round_table_1_urdf_path, 
            basePosition=[-2.0, 1.0, 0.0], # x, y, z (z=0 for floor)
            baseOrientation=p.getQuaternionFromEuler([1.57, 0, 0]), # Rotate 90 degrees
            useFixedBase=True # Make it static so the robot can crash into it without it flying away
        )
        # Table Chair 1.1
        p.loadURDF(
            chair_table_1_urdf_path, 
            basePosition=[-2.0, 1.5, 0.0], # x, y, z (z=0 for floor)
            baseOrientation=p.getQuaternionFromEuler([1.57, 0, 0]), # Rotate 90 degrees
            useFixedBase=True # Make it static so the robot can crash into it without it flying away
        )
        # Table Chair 1.2
        p.loadURDF(
            chair_table_1_urdf_path, 
            basePosition=[-2.0, 0.5, 0.0], # x, y, z (z=0 for floor)
            baseOrientation=p.getQuaternionFromEuler([1.57, 0, 3.14159]), # Rotate 90 degrees
            useFixedBase=True # Make it static so the robot can crash into it without it flying away
        )
        # Table Chair 1.3
        p.loadURDF(
            chair_table_1_urdf_path, 
            basePosition=[-2.5, 1.0, 0.0], # x, y, z (z=0 for floor)
            baseOrientation=p.getQuaternionFromEuler([1.57, 0, 1.57]), # Rotate 90 degrees
            useFixedBase=True # Make it static so the robot can crash into it without it flying away
        )
        # Table Chair 1.4
        p.loadURDF(
            chair_table_1_urdf_path, 
            basePosition=[-1.5, 1.0, 0.0], # x, y, z (z=0 for floor)
            baseOrientation=p.getQuaternionFromEuler([1.57, 0, -1.57]), # Rotate 90 degrees
            useFixedBase=True # Make it static so the robot can crash into it without it flying away
        )        
        print("Cafe table 1 and chairs loaded successfully.")
    except Exception as e:
        print(f"Could not load cafe_table or chair_table")

    try:
        # Cafe Table 2
        p.loadURDF(
            round_table_2_urdf_path, 
            basePosition=[-2.0, -3.0, 0.0], # x, y, z (z=0 for floor)
            baseOrientation=p.getQuaternionFromEuler([1.57, 0, 0]), # Rotate 90 degrees
            useFixedBase=True # Make it static so the robot can crash into it without it flying away
        )
        """
        # Table Chair 2.1
        p.loadURDF(
            chair_table_2_urdf_path, 
            basePosition=[-2.0, -2.5, 0.0], # x, y, z (z=0 for floor)
            baseOrientation=p.getQuaternionFromEuler([1.57, 0, 0]), # Rotate 90 degrees
            useFixedBase=True # Make it static so the robot can crash into it without it flying away
        )
        # Table Chair 2.2
        p.loadURDF(
            chair_table_2_urdf_path, 
            basePosition=[-2.0, -3.5, 0.0], # x, y, z (z=0 for floor)
            baseOrientation=p.getQuaternionFromEuler([1.57, 0, 3.14159]), # Rotate 90 degrees
            useFixedBase=True # Make it static so the robot can crash into it without it flying away
        )
        """
        # Table Chair 2.3
        p.loadURDF(
            chair_table_2_urdf_path, 
            basePosition=[-2.5, -3.0, 0.0], # x, y, z (z=0 for floor)
            baseOrientation=p.getQuaternionFromEuler([1.57, 0, 1.57]), # Rotate 90 degrees
            useFixedBase=True # Make it static so the robot can crash into it without it flying away
        )
        # Table Chair 2.4
        p.loadURDF(
            chair_table_2_urdf_path, 
            basePosition=[-1.5, -3.0, 0.0], # x, y, z (z=0 for floor)
            baseOrientation=p.getQuaternionFromEuler([1.57, 0, -1.57]), # Rotate 90 degrees
            useFixedBase=True # Make it static so the robot can crash into it without it flying away
        ) 
        print("Cafe table 2 and chairs loaded successfully.")
    except Exception as e:
        print(f"Could not load cafe_table or chair_table")

    try:
        # Cafe Table 3
        p.loadURDF(
            round_table_3_urdf_path, 
            basePosition=[-2.0, 6.0, 0.0], # x, y, z (z=0 for floor)
            baseOrientation=p.getQuaternionFromEuler([1.57, 0, 0]), # Rotate 90 degrees
            useFixedBase=True # Make it static so the robot can crash into it without it flying away
        )
        # Table Chair 3.1
        p.loadURDF(
            chair_table_3_urdf_path, 
            basePosition=[-2.0, 6.5, 0.0], # x, y, z (z=0 for floor)
            baseOrientation=p.getQuaternionFromEuler([1.57, 0, 0]), # Rotate 90 degrees
            useFixedBase=True # Make it static so the robot can crash into it without it flying away
        )
        # Table Chair 3.2
        p.loadURDF(
            chair_table_3_urdf_path, 
            basePosition=[-2.0, 5.5, 0.0], # x, y, z (z=0 for floor)
            baseOrientation=p.getQuaternionFromEuler([1.57, 0, 3.14159]), # Rotate 90 degrees
            useFixedBase=True # Make it static so the robot can crash into it without it flying away
        )
        """
        # Table Chair 3.3
        p.loadURDF(
            chair_table_3_urdf_path, 
            basePosition=[-2.5, 6.0, 0.0], # x, y, z (z=0 for floor)
            baseOrientation=p.getQuaternionFromEuler([1.57, 0, 1.57]), # Rotate 90 degrees
            useFixedBase=True # Make it static so the robot can crash into it without it flying away
        )
        # Table Chair 3.4
        p.loadURDF(
            chair_table_3_urdf_path, 
            basePosition=[-1.5, 6.0, 0.0], # x, y, z (z=0 for floor)
            baseOrientation=p.getQuaternionFromEuler([1.57, 0, -1.57]), # Rotate 90 degrees
            useFixedBase=True # Make it static so the robot can crash into it without it flying away
        ) 
        """
        print("Cafe table 3 and chairs loaded successfully.")
    except Exception as e:
        print(f"Could not load cafe_table or chair_table. Check path: {round_table_urdf_path}. Error: {e}")


    try:
        # Cafe Table 4
        p.loadURDF(
            round_table_4_urdf_path, 
            basePosition=[3.0, 7.0, 0.0], # x, y, z (z=0 for floor)
            baseOrientation=p.getQuaternionFromEuler([1.57, 0, 0]), # Rotate 90 degrees
            useFixedBase=True # Make it static so the robot can crash into it without it flying away
        )
        # Table Chair 4.1
        p.loadURDF(
            chair_table_4_urdf_path, 
            basePosition=[3.0, 7.5, 0.0], # x, y, z (z=0 for floor)
            baseOrientation=p.getQuaternionFromEuler([1.57, 0, 0]), # Rotate 90 degrees
            useFixedBase=True # Make it static so the robot can crash into it without it flying away
        )
        
        # Table Chair 4.2
        p.loadURDF(
            chair_table_4_urdf_path, 
            basePosition=[3.0, 6.5, 0.0], # x, y, z (z=0 for floor)
            baseOrientation=p.getQuaternionFromEuler([1.57, 0, 3.14159]), # Rotate 90 degrees
            useFixedBase=True # Make it static so the robot can crash into it without it flying away
        )
        """
        # Table Chair 4.3
        p.loadURDF(
            chair_table_4_urdf_path, 
            basePosition=[2.5, 7.0, 0.0], # x, y, z (z=0 for floor)
            baseOrientation=p.getQuaternionFromEuler([1.57, 0, 1.57]), # Rotate 90 degrees
            useFixedBase=True # Make it static so the robot can crash into it without it flying away
        )
        # Table Chair 4.4
        p.loadURDF(
            chair_table_4_urdf_path, 
            basePosition=[3.5, 7.0, 0.0], # x, y, z (z=0 for floor)
            baseOrientation=p.getQuaternionFromEuler([1.57, 0, -1.57]), # Rotate 90 degrees
            useFixedBase=True # Make it static so the robot can crash into it without it flying away
        ) 
        """
        print("Cafe table 2 and chairs loaded successfully.")
    except Exception as e:
        print(f"Could not load cafe_table or chair_table")


    try:
        # Cafe Table 5
        p.loadURDF(
            round_table_5_urdf_path, 
            basePosition=[3.0, -5.0, 0.0], # x, y, z (z=0 for floor)
            baseOrientation=p.getQuaternionFromEuler([1.57, 0, 0]), # Rotate 90 degrees
            useFixedBase=True # Make it static so the robot can crash into it without it flying away
        )
        # Table Chair 5.1
        p.loadURDF(
            chair_table_5_urdf_path, 
            basePosition=[3.0, -4.5, 0.0], # x, y, z (z=0 for floor)
            baseOrientation=p.getQuaternionFromEuler([1.57, 0, 0]), # Rotate 90 degrees
            useFixedBase=True # Make it static so the robot can crash into it without it flying away
        )
        # Table Chair 5.2
        p.loadURDF(
            chair_table_5_urdf_path, 
            basePosition=[3.0, -5.5, 0.0], # x, y, z (z=0 for floor)
            baseOrientation=p.getQuaternionFromEuler([1.57, 0, 3.14159]), # Rotate 90 degrees
            useFixedBase=True # Make it static so the robot can crash into it without it flying away
        )
        """
        # Table Chair 5.3
        p.loadURDF(
            chair_table_5_urdf_path, 
            basePosition=[2.5, -5.0, 0.0], # x, y, z (z=0 for floor)
            baseOrientation=p.getQuaternionFromEuler([1.57, 0, 1.57]), # Rotate 90 degrees
            useFixedBase=True # Make it static so the robot can crash into it without it flying away
        )
        # Table Chair 5.4
        p.loadURDF(
            chair_table_5_urdf_path, 
            basePosition=[3.5, -5.0, 0.0], # x, y, z (z=0 for floor)
            baseOrientation=p.getQuaternionFromEuler([1.57, 0, -1.57]), # Rotate 90 degrees
            useFixedBase=True # Make it static so the robot can crash into it without it flying away
        ) 
        """
        print("Cafe table 5 and chairs loaded successfully.")
    except Exception as e:
        print(f"Could not load cafe_table or chair_table")





    print(f"Scenario Loaded. Robot at: {ROBOT_START_POS}")
    print("Simulation running... Press Ctrl+C to exit.")
    
    # --- 6. Simulation Loop ---
    # We send zero actions just to keep the window open and the robot steady.
    action = np.zeros(env.n())
    
    print("Simulation running... Press Ctrl+C in the terminal to close.")

    try:
        while True:
            ob, *_ = env.step(action)
            time.sleep(1./240.) 
            
    except KeyboardInterrupt:
        print("Closing...")
    
    env.close()


if __name__ == "__main__":
    show_warnings = False
    warning_flag = "default" if show_warnings else "ignore"
    with warnings.catch_warnings():
        warnings.filterwarnings(warning_flag)
        run_barista_scenario(render=True)



