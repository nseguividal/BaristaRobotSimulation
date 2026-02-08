"""
Pre-configured scenarios for common use cases.
Load with: python main.py --config fast_test
"""

CONFIGS = {
    # Basic configurations
    'default': {
        'description': 'Default settings - PLANNER mode for both base and arm',
        'base_mode': 'planner',
        'arm_mode': 'planner',
        'base_horizon': 80,
        'arm_horizon': 40,
        'base_wx': 50.0,
        'base_wu': 2.0,
        'arm_wq': 50.0,
        'arm_wv': 5.0,
        'arm_wa': 0.5,
    },
    
    # Base TRACKER configurations
    'tracker_prm': {
        'description': 'Base uses PRM + MPC tracking, Arm uses direct planning',
        'base_mode': 'tracker',
        'base_global_planner': 'prm',
        'prm_samples': 500,
        'prm_k_neighbors': 20,
        'arm_mode': 'planner',
        'base_horizon': 25,
        'arm_horizon': 20,
        'base_wx': 70.0,
        'base_wu': 25.0,
    },
    
    'tracker_rrt': {
        'description': 'Base uses RRT + MPC tracking',
        'base_mode': 'tracker',
        'base_global_planner': 'rrt',
        'rrt_max_iter': 3000,
        'arm_mode': 'planner',
        'base_horizon': 25,
        'arm_horizon': 20,
        'base_wx': 60.0,
        'base_wu': 20.0,
    },
    
    'tracker_rrt_star': {
        'description': 'Base uses RRT* + MPC tracking (optimal)',
        'base_mode': 'tracker',
        'base_global_planner': 'rrt*',
        'rrt_max_iter': 5000,
        'arm_mode': 'planner',
        'base_horizon': 30,
        'arm_horizon': 25,
        'base_wx': 80.0,
        'base_wu': 30.0,
    },
    
    # Arm TRACKER configurations
    'arm_tracker_prm': {
        'description': 'Arm uses PRM + MPC tracking in joint space',
        'base_mode': 'planner',
        'arm_mode': 'tracker',
        'arm_global_planner': 'prm',
        'arm_prm_samples': 500,
        'arm_prm_k_neighbors': 10,
        'base_horizon': 20,
        'arm_horizon': 25,
        'arm_wq': 70.0,
        'arm_wv': 8.0,
        'arm_wa': 0.8,
    },
    
    'arm_tracker_rrt': {
        'description': 'Arm uses RRT + MPC tracking in joint space',
        'base_mode': 'planner',
        'arm_mode': 'tracker',
        'arm_global_planner': 'rrt',
        'arm_rrt_max_iter': 2000,
        'base_horizon': 20,
        'arm_horizon': 25,
        'arm_wq': 60.0,
        'arm_wv': 7.0,
        'arm_wa': 0.7,
    },
    
    # Both TRACKER configurations
    'both_tracker_prm': {
        'description': 'Both base and arm use PRM + MPC tracking',
        'base_mode': 'tracker',
        'base_global_planner': 'prm',
        'prm_samples': 100,
        'prm_k_neighbors': 15,
        'arm_mode': 'tracker',
        'arm_global_planner': 'prm',
        'arm_prm_samples': 1000,
        'arm_prm_k_neighbors': 15,
        'base_horizon': 100,
        'arm_horizon': 50,
    },
    
    'both_tracker_prm': {
        'description': 'Both base and arm use PRM + MPC tracking',
        'base_mode': 'tracker',
        'base_global_planner': 'rrt*',
        'prm_samples': 100,
        'prm_k_neighbors': 15,
        'arm_mode': 'tracker',
        'arm_global_planner': 'rrt',
        'arm_rrt_max_iter': 2000,
        'base_horizon': 100,
        'arm_horizon': 50,
    },

    'both_tracker_mixed': {
        'description': 'Base uses RRT*, Arm uses PRM (mixed planners)',
        'base_mode': 'tracker',
        'base_global_planner': 'rrt*',
        'rrt_max_iter': 5000,
        'arm_mode': 'tracker',
        'arm_global_planner': 'prm',
        'arm_prm_samples': 600,
        'arm_prm_k_neighbors': 12,
        'base_horizon': 30,
        'arm_horizon': 28,
        'base_wx': 80.0,
        'base_wu': 30.0,
        'arm_wq': 80.0,
        'arm_wv': 9.0,
        'arm_wa': 0.9,
    },
    
}


def get_config(name):
    """
    Get configuration by name.
    
    Args:
        name: Configuration name (e.g., 'fast_test', 'production')
    
    Returns:
        dict: Configuration parameters
    
    Raises:
        KeyError: If configuration not found
    """
    if name not in CONFIGS:
        available = ', '.join(CONFIGS.keys())
        raise KeyError(f"Unknown config '{name}'. Available: {available}")
    
    return CONFIGS[name].copy()


def list_configs():
    """Print all available configurations."""
    print("\n" + "="*70)
    print("AVAILABLE CONFIGURATIONS")
    print("="*70)
    
    for name, config in CONFIGS.items():
        print(f"\n{name}:")
        print(f"  {config['description']}")
        print(f"  Mode: base={config.get('base_mode', 'N/A')}, arm={config.get('arm_mode', 'N/A')}")
        if config.get('base_mode') == 'tracker':
            print(f"  Global planner: {config.get('base_global_planner', 'N/A')}")
    
    print("\n" + "="*70)
    print("Usage: python main.py --config <name> [additional args]")
    print("Example: python main.py --config production --target_x 3.0")
    print("="*70 + "\n")


# Export for easy import
__all__ = ['CONFIGS', 'get_config', 'list_configs']
