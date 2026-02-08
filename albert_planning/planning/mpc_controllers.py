"""
MPC controllers for base and arm with collision avoidance.
"""
import numpy as np
import casadi as cs
import pybullet as p
from typing import Optional, List


class BaseMPC:

    """MPC for differential drive with obstacle avoidance and trajectory tracking.
    Supports both static obstacles and dynamic obstacles (velocity obstacles).
    Dynamic obstacles are predicted over the horizon based on their current velocity."""
    MAX_DYNAMIC_OBS = 5  # Maximum number of dynamic obstacles

    def __init__(self, dynamics, N, x_target=None, wx=50.0, wu=20.0, 
                 max_iter=30, warm_start=True, obstacles=None,
                 obs_radius=0.7, robot_radius=0.3,
                 safety_margin=0.05, use_soft_constraints=True, env=None, dynamic_obs_weight=700.0):

        self.dynamics = dynamics
        self.N = N
        
        # Support both goal tracking and trajectory tracking
        self.x_target = x_target if x_target is not None else np.zeros(3)
        self.x_ref_traj = None  # (3, N+1) reference trajectory for tracking
        
        self.wx = wx
        self.wu = wu
        self.max_iter = max_iter
        self.warm_start = warm_start
       
        # Obstacle avoidance
        self.env = env
        # self.obstacles = obstacles if obstacles is not None else []
        self.obs_radius = obs_radius
        self.safety_margin = safety_margin

        self.use_soft_constraints = use_soft_constraints
        self.robot_radius = robot_radius

        self.soft_constraint_weight = 8000.0  # Weight for soft constraint penalty
        self.constraint_every_n_steps = 3  # Apply obstacle constraints every N steps
        # Dynamic obstacle avoidance (velocity obstacles)
        self.dynamic_obs_weight = dynamic_obs_weight
        # Create optimization with velocity obstacle support
        (self.opti, self.X_var, self.U_var, self.p_x_init, self.p_x_ref,
         self.p_dyn_obs_pos, self.p_dyn_obs_vel, self.p_dyn_obs_rad) = self._create_optimization()

    
    def set_reference_trajectory(self, x_ref_traj):
        """
        Set reference trajectory for tracking mode.
        
        Args:
            x_ref_traj: (3, T) or (T, 3) reference trajectory [x, y, theta]
        """
        x_ref_traj = np.asarray(x_ref_traj)
        if x_ref_traj.shape[0] != 3:
            x_ref_traj = x_ref_traj.T
        self.x_ref_traj = x_ref_traj
    
    def _create_optimization(self):

        """Create CasADi optimization with trajectory tracking and velocity obstacle avoidance."""
        opti = cs.Opti()
        nx = self.dynamics.state_dim
        nu = self.dynamics.input_dim
        N = self.N
        dt = self.dynamics.dt

        X = [opti.variable(nx) for _ in range(N + 1)]
        U = [opti.variable(nu) for _ in range(N)]

        # Input constraints
        for k in range(N):
            opti.subject_to(opti.bounded(self.dynamics.u_min, U[k], self.dynamics.u_max))

        # Initial condition parameter
        p_x_init = opti.parameter(nx)
        opti.subject_to(X[0] == p_x_init)

        # Reference trajectory parameters
        p_x_ref = [opti.parameter(nx) for _ in range(N + 1)]

        # Dynamic obstacle parameters (velocity obstacles)
        p_dyn_obs_pos = [opti.parameter(2) for _ in range(self.MAX_DYNAMIC_OBS)]
        p_dyn_obs_vel = [opti.parameter(2) for _ in range(self.MAX_DYNAMIC_OBS)]
        p_dyn_obs_rad = [opti.parameter(1) for _ in range(self.MAX_DYNAMIC_OBS)]

        # Build cost function
        cost = 0
        robot_radius = 0.35

        for k in range(N):
            # Tracking cost
            pos_error = X[k][:2] - p_x_ref[k][:2]
            cost += self.wx * pos_error.T @ pos_error
            cost += self.wu * U[k].T @ U[k]

            # Dynamics constraint
            opti.subject_to(X[k + 1] == self.dynamics.discrete_dynamics(X[k], U[k]))

            # State bounds
            opti.subject_to(opti.bounded(self.dynamics.x_min[0], X[k][0], self.dynamics.x_max[0]))
            opti.subject_to(opti.bounded(self.dynamics.x_min[1], X[k][1], self.dynamics.x_max[1]))


            # Static obstacle avoidance
            # for obs in self.obstacles:
            #     obs_x, obs_y = obs[:2]
            #     dist_sq = (X[k][0] - obs_x)**2 + (X[k][1] - obs_y)**2
            #     cost += 20.0 / (dist_sq + 0.1) 

            # === VELOCITY OBSTACLES ===
            # Predict obstacle position at timestep k and add barrier cost
            t_k = k * dt
            for i in range(self.MAX_DYNAMIC_OBS):
                # Predicted position of obstacle i at time t_k
                pred_x = p_dyn_obs_pos[i][0] + p_dyn_obs_vel[i][0] * t_k
                pred_y = p_dyn_obs_pos[i][1] + p_dyn_obs_vel[i][1] * t_k

                # Distance from robot to predicted obstacle position
                dx = X[k][0] - pred_x
                dy = X[k][1] - pred_y
                dist_sq = dx**2 + dy**2

                # Barrier cost (weighted by obstacle radius - disabled obstacles have radius=0)
                # This naturally disables the cost for unused obstacle slots
                cost += self.dynamic_obs_weight * p_dyn_obs_rad[i] / (dist_sq + 0.1)

        # Terminal cost
        terminal_error = X[-1][:2] - p_x_ref[-1][:2]
        cost += 10.0 * terminal_error.T @ terminal_error

        # =====================================================================
        # COLLISION AVOIDANCE - Use obstacles from environment
        # =====================================================================
        
        self.obstacles = self.env.get_obstacles() if self.env is not None else []
        if len(self.obstacles) > 0:
            total_clearance = self.robot_radius + self.safety_margin
            
            for k in range(N + 1):
                # Apply constraints only every N steps to reduce computation
                if k % self.constraint_every_n_steps != 0:
                    continue
                
                for obst_id, obst in self.obstacles.items():
                    # Skip walls (handled by state bounds)
                    obst_name = str(obst.name()).lower()
                    if 'wall' in obst_name:
                        continue
                    
                    # Robot position at timestep k
                    robot_pos = X[k][:2]  # [x, y]
                    
                    # Obstacle position (from mpscenes obstacle)
                    obs_pos_full = obst.position()  # Returns [x, y, z]
                    obs_pos = cs.vertcat(obs_pos_full[0], obs_pos_full[1])  # Only [x, y]
                    
                    # Get obstacle size for minimum distance
                    obs_size = obst.size()  # Returns obstacle dimensions
                    
                    # Compute bounding radius based on obstacle type
                    if obst.type() == 'cylinder':
                        # For cylinder: radius is directly available
                        obs_radius = obs_size[0] if len(obs_size) > 0 else 0.5
                    elif obst.type() == 'box':
                        # For box: use half diagonal as bounding radius
                        # obs_size is [length, width, height]
                        obs_radius = cs.sqrt(obs_size[0]**2 + obs_size[1]**2) / 2
                    else:
                        # Default: assume sphere/circle
                        obs_radius = obs_size[0] if len(obs_size) > 0 else 0.5
                    
                    # Distance from robot to obstacle center
                    dx = robot_pos[0] - obs_pos[0]
                    dy = robot_pos[1] - obs_pos[1]
                    dist_squared = dx**2 + dy**2
                    
                    # Minimum distance required
                    min_dist = obs_radius + total_clearance
                    min_dist_squared = min_dist**2
                    
                    if self.use_soft_constraints:
                        # Soft constraint: barrier function in cost
                        print(f"Adding soft constraint for obstacle {obst_name} at step {k}")
                        epsilon = 0.01  # Numerical stability
                        penalty = self.soft_constraint_weight / (
                            dist_squared - min_dist_squared + epsilon
                        )
                        slack = cs.fmax(0, min_dist_squared - dist_squared)
                        cost += self.soft_constraint_weight * slack**2
                        # cost += penalty
                    else:
                        # Hard constraint: dist_squared >= min_dist_squared
                        opti.subject_to(dist_squared >= min_dist_squared)
        
        opti.minimize(cost)

        # Solver options
        opts = {
            "ipopt.print_level": 0,
            "print_time": 0,
            "ipopt.max_iter": 1000,
            "ipopt.tol": 1e-4,
            "ipopt.acceptable_tol": 1e-3,
            "ipopt.acceptable_iter": 5,
        }
        opti.solver("ipopt", opts)

        # Initialize parameters for first solve
        opti.set_value(p_x_init, np.zeros(nx))
        for k in range(N + 1):
            opti.set_value(p_x_ref[k], self.x_target)

        # Initialize dynamic obstacle parameters (all disabled)
        for i in range(self.MAX_DYNAMIC_OBS):
            opti.set_value(p_dyn_obs_pos[i], np.zeros(2))
            opti.set_value(p_dyn_obs_vel[i], np.zeros(2))
            opti.set_value(p_dyn_obs_rad[i], 0.0)

        opti.solve()

        # Update solver for real-time
        opts["ipopt.max_iter"] = self.max_iter
        opti.solver("ipopt", opts)

        return opti, X, U, p_x_init, p_x_ref, p_dyn_obs_pos, p_dyn_obs_vel, p_dyn_obs_rad
    
    def update_target(self, x_target):
        """Update target for waypoint tracking."""
        self.x_target = x_target
        # Recreate optimization with new target

        (self.opti, self.X_var, self.U_var, self.p_x_init, self.p_x_ref,
         self.p_dyn_obs_pos, self.p_dyn_obs_vel, self.p_dyn_obs_rad) = self._create_optimization()


    def _update_dynamic_obstacles(self):
        """Update dynamic obstacle parameters from environment."""
        if self.env is None:
            return

        try:
            dyn_obs_states = self.env.get_dynamic_obstacles_state()
        except AttributeError:
            return

        n_obs = min(len(dyn_obs_states), self.MAX_DYNAMIC_OBS)

        # Set active obstacles
        for i in range(n_obs):
            pos, vel, radius = dyn_obs_states[i]
            self.opti.set_value(self.p_dyn_obs_pos[i], pos)
            self.opti.set_value(self.p_dyn_obs_vel[i], vel)
            self.opti.set_value(self.p_dyn_obs_rad[i], radius)

        # Disable unused slots
        for i in range(n_obs, self.MAX_DYNAMIC_OBS):
            self.opti.set_value(self.p_dyn_obs_pos[i], np.zeros(2))
            self.opti.set_value(self.p_dyn_obs_vel[i], np.zeros(2))
            self.opti.set_value(self.p_dyn_obs_rad[i], 0.0)

    def solve(self, x_init):
        """
        Solve MPC for current state.
        Automatically uses trajectory tracking if reference is set via set_reference_trajectory(),
        otherwise uses fixed target (trajectory planning mode).
        """
        if x_init.shape[0] != 3:
            raise ValueError(f"Expected 3D state, got {x_init.shape[0]}")

        # Update dynamic obstacles from environment
        self._update_dynamic_obstacles()

        # Set reference trajectory or fixed target
        if self.x_ref_traj is not None:
            # Trajectory tracking mode
            if self.x_ref_traj.shape != (3, self.N + 1):
                raise ValueError(
                    f"Reference trajectory has wrong shape: {self.x_ref_traj.shape}, "
                    f"expected (3, {self.N + 1})"
                )
            for k in range(self.N + 1):
                self.opti.set_value(self.p_x_ref[k], self.x_ref_traj[:, k])
        else:
            # Trajectory planning mode (fixed target)
            for k in range(self.N + 1):
                self.opti.set_value(self.p_x_ref[k], self.x_target)
        
        try:
            self.opti.set_value(self.p_x_init, x_init)
            sol = self.opti.solve()
            
            # Warm start
            if self.warm_start:
                for k in range(self.N):
                    self.opti.set_initial(self.X_var[k], sol.value(self.X_var[k + 1]))
                for k in range(self.N - 1):
                    self.opti.set_initial(self.U_var[k], sol.value(self.U_var[k + 1]))
                self.opti.set_initial(self.X_var[-1], sol.value(self.X_var[-1]))
                self.opti.set_initial(self.U_var[-1], sol.value(self.U_var[-1]))
            
            u_opt = sol.value(self.U_var[0])
            x_next = sol.value(self.X_var[1])
            
            x_traj = np.zeros((3, self.N + 1))
            for k in range(self.N + 1):
                x_traj[:, k] = sol.value(self.X_var[k])
            
            return u_opt, x_next, x_traj
        
        except Exception as e:
            print(f"MPC solver failed: {e}")
            u_safe = np.zeros(2)
            x_traj = np.tile(x_init.reshape(-1, 1), (1, self.N + 1))
            return u_safe, x_init, x_traj


class ArmMPC:
    """MPC for manipulator arm with trajectory tracking."""
    def __init__(self, model, N, q_target=None, w_q=50.0, w_v=5.0, w_a=0.5,
                 w_final=None, max_iter=50, warm_start=True, use_torque_constraints=True,
                 w_bounds=1e3):
        self.model = model
        self.N = int(N)
        self.n = model.n
        self.dt = float(model.dt)
        
        self.q_target = np.zeros(self.n) if q_target is None else np.asarray(q_target).reshape(self.n)
        self.q_ref_traj = None  # For trajectory tracking
        
        self.w_q = float(w_q)
        self.w_v = float(w_v)
        self.w_a = float(w_a)
        self.w_final = float(w_final) if w_final is not None else 10.0 * float(w_q)
        
        self.max_iter = max_iter
        self.warm_start = warm_start
        self.use_torque_constraints = use_torque_constraints
        self.w_bounds = float(w_bounds)

        
        self.q_min = self.model.q_min
        self.q_max = self.model.q_max
        self.qd_min = self.model.qd_min
        self.qd_max = self.model.qd_max
        self.u_min = self.model.u_min
        self.u_max = self.model.u_max
        
        (self.opti, self.X_var, self.U_var, self.p_x_init, 
         self.p_q_ref, self.p_M, self.p_h) = self._create_optimization()
    
    def set_reference_trajectory(self, q_ref_traj):
        """Set reference trajectory for tracking."""
        q_ref_traj = np.asarray(q_ref_traj)
        if q_ref_traj.shape[0] != self.n:
            q_ref_traj = q_ref_traj.T
        self.q_ref_traj = q_ref_traj
    
    def _create_optimization(self):
        """Create optimization with trajectory tracking."""
        opti = cs.Opti()
        
        X = [opti.variable(2 * self.n) for _ in range(self.N + 1)]
        U = [opti.variable(self.n) for _ in range(self.N)]
        
        p_x_init = opti.parameter(2 * self.n)
        opti.subject_to(X[0] == p_x_init)
        
        # Reference trajectory parameters
        p_q_ref = [opti.parameter(self.n) for _ in range(self.N + 1)]
        
        # Torque constraint parameters
        p_M = opti.parameter(self.n, self.n)
        p_h = opti.parameter(self.n)
        
        cost = 0
        hard_slack = 0.05
        s_q = [opti.variable(self.n) for _ in range(self.N + 1)]
        s_qd = [opti.variable(self.n) for _ in range(self.N + 1)]
        for k in range(self.N + 1):
            opti.subject_to(s_q[k] >= 0)
            opti.subject_to(s_qd[k] >= 0)
        
        for k in range(self.N):
            qk = X[k][:self.n]
            qdk = X[k][self.n:]
            uk = U[k]
            
            # Tracking cost
            cost += self.w_q * cs.sumsqr(qk - p_q_ref[k])
            cost += self.w_v * cs.sumsqr(qdk)
            cost += self.w_a * cs.sumsqr(uk)
            cost += self.w_bounds * (cs.sumsqr(s_q[k]) + cs.sumsqr(s_qd[k]))
            
            # Constraints
            opti.subject_to(
                opti.bounded(
                    self.q_min - hard_slack - s_q[k],
                    qk,
                    self.q_max + hard_slack + s_q[k],
                )
            )
            opti.subject_to(
                opti.bounded(
                    self.qd_min - hard_slack - s_qd[k],
                    qdk,
                    self.qd_max + hard_slack + s_qd[k],
                )
            )
            opti.subject_to(opti.bounded(self.u_min, uk, self.u_max))
            opti.subject_to(X[k + 1] == X[k] + self.dt * self.model.f(X[k], uk))
            
            # Torque constraints
            if self.use_torque_constraints and False:
                tau_approx = cs.mtimes(p_M, uk) + p_h
                tau_slack = 2.0
                opti.subject_to(opti.bounded(
                    self.model.tau_min - tau_slack,
                    tau_approx,
                    self.model.tau_max + tau_slack
                ))
        
        # Terminal cost
        qT = X[-1][:self.n]
        qdT = X[-1][self.n:]
        cost += self.w_final * cs.sumsqr(qT - p_q_ref[-1])
        cost += self.w_bounds * (cs.sumsqr(s_q[-1]) + cs.sumsqr(s_qd[-1]))
        opti.subject_to(
            opti.bounded(
                self.q_min - hard_slack - s_q[-1],
                qT,
                self.q_max + hard_slack + s_q[-1],
            )
        )
        opti.subject_to(
            opti.bounded(
                self.qd_min - hard_slack - s_qd[-1],
                qdT,
                self.qd_max + hard_slack + s_qd[-1],
            )
        )
        
        opti.minimize(cost)
        
        opts = {
            "ipopt.print_level": 0,
            "print_time": 0,
            "ipopt.tol": 1e-3,
            "ipopt.acceptable_tol": 1e-2,
            "ipopt.acceptable_iter": 10,
            "ipopt.max_iter": 1000,
        }
        opti.solver("ipopt", opts)
        
        return opti, X, U, p_x_init, p_q_ref, p_M, p_h
    
    def solve(self, q_init, qd_init=None):
        """Solve MPC with current reference."""
        q0 = np.asarray(q_init).reshape(self.n)
        qd0 = np.zeros(self.n) if qd_init is None else np.asarray(qd_init).reshape(self.n)
        
        if q0.shape[0] != self.n or qd0.shape[0] != self.n:
            raise ValueError("Initial state size mismatch")
        
        # Set reference trajectory
        if self.q_ref_traj is not None:
            if self.q_ref_traj.shape != (self.n, self.N + 1):
                raise ValueError(f"Reference shape mismatch: {self.q_ref_traj.shape}")
            for k in range(self.N + 1):
                self.opti.set_value(self.p_q_ref[k], self.q_ref_traj[:, k])
        else:
            for k in range(self.N + 1):
                self.opti.set_value(self.p_q_ref[k], self.q_target)
        
        # Update M and h for torque constraints
        if self.use_torque_constraints and False:
            M, h = self.model.compute_M_h(q0, qd0)
            self.opti.set_value(self.p_M, M)
            self.opti.set_value(self.p_h, h)
        
        try:
            self.opti.set_value(self.p_x_init, np.concatenate([q0, qd0]))
            sol = self.opti.solve()
            
            if self.warm_start:
                for k in range(self.N):
                    self.opti.set_initial(self.X_var[k], sol.value(self.X_var[k + 1]))
                for k in range(self.N - 1):
                    self.opti.set_initial(self.U_var[k], sol.value(self.U_var[k + 1]))
                self.opti.set_initial(self.X_var[-1], sol.value(self.X_var[-1]))
                self.opti.set_initial(self.U_var[-1], sol.value(self.U_var[-1]))
            
            u_opt = np.array(sol.value(self.U_var[0])).reshape(self.n)
            x_next = np.array(sol.value(self.X_var[1])).reshape(2 * self.n)
            q_next = x_next[:self.n]
            qd_next = x_next[self.n:]
            
            x_traj = np.zeros((2 * self.n, self.N + 1))
            for k in range(self.N + 1):
                x_traj[:, k] = np.array(sol.value(self.X_var[k])).reshape(2 * self.n)
            
            return u_opt, q_next, qd_next, x_traj
        
        except Exception as e:
            print(f"Arm MPC solver failed: {e}")
            u_safe = np.zeros(self.n)
            x0 = np.concatenate([q0, qd0])
            x_traj = np.tile(x0.reshape(-1, 1), (1, self.N + 1))
            return u_safe, q0, qd0, x_traj
