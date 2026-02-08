import numpy as np
import time

class TrialMetrics:
    def __init__(self):
        self.success = False
        self.collision_static = 0
        self.collision_dynamic = 0
        self.robustness_fail = False

        # Path lengths
        self.global_path_length = 0.0  # Accumulates length of all global plans
        
        # Timers
        self.t_start = 0
        self.execution_time = 0.0
        
        # Physics Logs [time, v_lin, v_ang, pos_x, pos_y]
        self.motion_log = [] 

    def start_execution(self):
        self.t_start = time.time()

    def stop_execution(self):
        if self.t_start > 0:
            self.execution_time = time.time() - self.t_start

    def add_global_path(self, path):
        """
        Calculates and adds the length of a global planner path.
        Call this whenever a new path is generated (Pick phase, Place phase, etc).
        """
        if path is None or len(path) < 2:
            return
        
        # Convert to numpy array if list
        path_arr = np.array(path)
        
        # Use only X, Y columns
        points = path_arr[:, :2]
        
        # Calculate sum of Euclidean distances between segments
        segment_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
        length = np.sum(segment_lengths)
        
        self.global_path_length += length
        print(f"  [Metrics] Added global path segment: {length:.4f} m (Total: {self.global_path_length:.4f} m)")

    def log_step(self, v_lin, v_ang, pos_x, pos_y, dt):
        timestamp = len(self.motion_log) * dt
        self.motion_log.append([timestamp, v_lin, v_ang, pos_x, pos_y])

    def print_excel_report(self):
        path_length = 0.0
        acc_norm = 0.0
        jerk_norm = 0.0
        
        if len(self.motion_log) > 1:
            data = np.array(self.motion_log)
            # Path Length (Execution)
            positions = data[:, 3:5]
            dist_steps = np.linalg.norm(np.diff(positions, axis=0), axis=1)
            execution_path_length = np.sum(dist_steps)
            
            # Smoothness
            velocities = data[:, 1]
            dt = data[1,0] - data[0,0] if len(data) > 1 else 0.01
            acc = np.diff(velocities) / dt
            jerk = np.diff(acc) / dt
            acc_norm = np.mean(np.abs(acc))
            jerk_norm = np.mean(np.abs(jerk))

        print("\n" + "="*40)
        print("📋 METRICS REPORT (Copy to Excel)")
        print("="*40)
        print(f"Success (0/1):        {1 if self.success else 0}")
        print(f"Execution Time (s):   {self.execution_time:.4f}")
        print(f"Global Path Len (m):  {self.global_path_length:.4f}")  
        print(f"Exec Path Len (m):    {execution_path_length:.4f}")    
        print(f"Accel Norm (m/s²):    {acc_norm:.4f}")
        print(f"Jerk Norm (m/s³):     {jerk_norm:.4f}")
        print(f"Local Minima (0/1):   {1 if self.robustness_fail else 0}")
        print("="*40 + "\n")