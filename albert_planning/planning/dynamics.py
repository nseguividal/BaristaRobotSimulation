"""
Robot dynamics models for mobile manipulator.
"""
import numpy as np
import casadi as cs
import pinocchio as pin


class DifferentialDriveDynamics:
    """Differential drive mobile base dynamics."""
    
    def __init__(self, dt, facing_direction='-y'):
        self.dt = dt
        self.facing_direction = facing_direction
        self.state_dim = 3  # [x, y, theta]
        self.input_dim = 2  # [v, omega]
        
        # State limits
        self.x_min = np.array([-5., -10., -2*np.pi]) #  assume room size of 10m x 20m
        self.x_max = np.array([5., 10., 2*np.pi])
        
        # Input limits
        self.u_min = np.array([-1.5, -1.5])
        self.u_max = np.array([ 1.5,  1.5])

    def continuous_dynamics(self, x, u):
        """Continuous-time dynamics dx/dt = f(x, u)."""
        if self.facing_direction == '-y':
            x_dot = u[0] * cs.sin(x[2])
            y_dot = -u[0] * cs.cos(x[2])
        elif self.facing_direction == 'y':
            x_dot = -u[0] * cs.sin(x[2])
            y_dot = u[0] * cs.cos(x[2])
        elif self.facing_direction == 'x':
            x_dot = u[0] * cs.cos(x[2])
            y_dot = u[0] * cs.sin(x[2])
        elif self.facing_direction == '-x':
            x_dot = -u[0] * cs.cos(x[2])
            y_dot = -u[0] * cs.sin(x[2])
        else:
            raise ValueError(f"Invalid facing_direction: {self.facing_direction}")
        
        theta_dot = u[1]
        return cs.vertcat(x_dot, y_dot, theta_dot)
    
    def discrete_dynamics(self, x, u):
        """Euler integration: x[k+1] = x[k] + dt * f(x[k], u[k])."""
        return x + self.dt * self.continuous_dynamics(x, u)


class ManipulatorDynamics:
    """Manipulator arm dynamics with Pinocchio inverse dynamics."""
    
    def __init__(self, robot, arm_indices, dt, urdf_path=None):
        self.robot = robot
        self.arm_indices = list(arm_indices)
        self.dt = float(dt)
        self.n = len(self.arm_indices)
        
        # Extract limits
        self.q_min, self.q_max = self._extract_limits("_limit_pos_j", offset=1, default=3.0)
        self.qd_min, self.qd_max = self._extract_limits("_limit_vel_j", offset=1, default=2.0)
        self.tau_min, self.tau_max = self._extract_limits("_limit_tor_j", offset=0, default=50.0)
        
        # Aggiungi questo print in dynamics.py dopo _extract_limits():
        print(f"Joint limits:")
        print(f"  q:   [{self.q_min}, {self.q_max}]")
        print(f"  qd:  [{self.qd_min}, {self.qd_max}]")
        print(f"  tau: [{self.tau_min}, {self.tau_max}]")
        
        try:
            acc = robot._limit_acc_j
            self.u_min = np.array([acc[0, idx] for idx in arm_indices])
            self.u_max = np.array([acc[1, idx] for idx in arm_indices])
        except:
            self.u_min = -np.ones(self.n) * 20.0
            self.u_max = np.ones(self.n) * 20.0
        
        # CasADi double integrator
        q = cs.SX.sym("q", self.n)
        qd = cs.SX.sym("qd", self.n)
        qdd = cs.SX.sym("qdd", self.n)
        state = cs.vertcat(q, qd)
        rhs = cs.vertcat(qd, qdd)
        self.f = cs.Function("f", [state, qdd], [rhs])
        
        # Build inverse dynamics
        self._build_inverse_dynamics(urdf_path)
    
    def _extract_limits(self, attr, offset, default):
        try:
            limits = getattr(self.robot, attr)
            lo = np.array([limits[0, idx + offset] for idx in self.arm_indices])
            hi = np.array([limits[1, idx + offset] for idx in self.arm_indices])
        except:
            lo = -np.ones(self.n) * float(default)
            hi = np.ones(self.n) * float(default)
        return lo, hi
    
    def _build_inverse_dynamics(self, urdf_path):
        """Build Pinocchio model for torque computation."""
        if urdf_path is None:
            urdf_path = getattr(self.robot, "_urdf_file", None)
        if urdf_path is None:
            raise RuntimeError("URDF path not provided")
        
        # Load and reduce model
        model = pin.buildModelFromUrdf(urdf_path)
        arm_joints = [f"mmrobot_joint{i}" for i in range(1, 8)]
        
        arm_joint_ids = [model.getJointId(jn) for jn in arm_joints if model.existJointName(jn)]
        joints_to_lock = [i for i in range(1, model.njoints) if i not in arm_joint_ids]
        
        model_red = pin.buildReducedModel(model, joints_to_lock, pin.neutral(model))
        data_red = model_red.createData()
        
        if model_red.nv != 7:
            raise RuntimeError(f"Expected 7 DOF, got {model_red.nv}")
        
        self._pin_model = model_red
        self._pin_data = data_red
    
    def compute_M_h(self, q, qd):
        """Compute mass matrix M(q) and bias h(q, qd) = C*qd + g."""
        q_np = np.array(q).flatten()
        qd_np = np.array(qd).flatten()
        
        M = pin.crba(self._pin_model, self._pin_data, q_np)
        h = pin.rnea(self._pin_model, self._pin_data, q_np, qd_np, np.zeros(7))
        
        return M, h
