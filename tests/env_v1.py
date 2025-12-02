import pybullet as p
import pybullet_data
import time

class SimpleHouseEnv:
    def __init__(self, render=True):
        """Creates a PyBullet environment with 4 walls and a table."""
        self.render = render

        # Connect to PyBullet
        if render:
            self.physics_client = p.connect(p.GUI)
        else:
            self.physics_client = p.connect(p.DIRECT)

        # Add PyBullet search paths
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        # Gravity
        p.setGravity(0, 0, -9.81)

        # Create environment
        self.create_floor()
        self.create_walls()
        self.create_table()

    # ----------------------------------------------------------
    # Basic elements
    # ----------------------------------------------------------
    def create_floor(self):
        self.plane_id = p.loadURDF("plane.urdf")

    def create_walls(self):
        """Creates a 4-wall room around the origin."""
        wall_thickness = 0.1
        wall_height = 1.0
        wall_length = 4.0

        # Collision/visual shape
        wall_collision = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=[wall_length / 2, wall_thickness / 2, wall_height / 2]
        )
        wall_visual = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[wall_length / 2, wall_thickness / 2, wall_height / 2],
            rgbaColor=[0.8, 0.8, 0.8, 1.0]
        )

        # Back wall
        p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=wall_collision,
            baseVisualShapeIndex=wall_visual,
            basePosition=[0, -wall_length / 2, wall_height / 2]
        )

        # Front wall
        p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=wall_collision,
            baseVisualShapeIndex=wall_visual,
            basePosition=[0, wall_length / 2, wall_height / 2]
        )

        # Left wall (rotated 90°)
        p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=wall_collision,
            baseVisualShapeIndex=wall_visual,
            basePosition=[-wall_length / 2, 0, wall_height / 2],
            baseOrientation=p.getQuaternionFromEuler([0, 0, 1.5708])
        )

        # Right wall
        p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=wall_collision,
            baseVisualShapeIndex=wall_visual,
            basePosition=[wall_length / 2, 0, wall_height / 2],
            baseOrientation=p.getQuaternionFromEuler([0, 0, 1.5708])
        )

    def create_table(self):
        """Loads a table from PyBullet's built-in assets."""
        try:
            p.setAdditionalSearchPath(pybullet_data.getDataPath())
            self.table_id = p.loadURDF("table/table.urdf", basePosition=[1.0, 0.0, 0.0])
        except:
            print("⚠️ Could not load table/table.urdf -- PyBullet version may lack this model")

    # ----------------------------------------------------------
    # Simulation loop
    # ----------------------------------------------------------
    def run(self, sim_time=10):
        """Runs the simulation for a given duration (seconds)."""
        steps = int(sim_time * 240)  # PyBullet default timestep
        for _ in range(steps):
            p.stepSimulation()
            if self.render:
                time.sleep(1.0 / 240.0)

    def close(self):
        p.disconnect()


if __name__ == "__main__":
    env = SimpleHouseEnv(render=True)
    env.run(sim_time=10)
    env.close()



