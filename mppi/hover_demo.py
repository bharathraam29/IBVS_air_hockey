import torch
import numpy as np
import time
from env.panda_pushing_env import PandaHoverEnv
from controller.controller import PushingController, hover_cost_function
from model.state_dynamics_models import HoverKinematicModel

class TerminalColors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'


def hover_at_position(target_position, num_steps=5000, render=True, device="cpu"):
    """
    Make the robot arm hover at a specified target position.
    
    Args:
        target_position: numpy array [x, y, z] - target hover position in meters
        num_steps: number of control steps to take
        render: whether to render the simulation
        device: torch device ("cpu" or "cuda")
    
    Returns:
        final_state: final end-effector position
        reached_goal: whether the goal was reached
    """

    print(f"Target position: [{target_position[0]:.3f}, {target_position[1]:.3f}, {target_position[2]:.3f}]")
    print(f"Number of steps: {num_steps}")
    print()
    success = False
    hover_success = False
    # Create environment
    env = PandaHoverEnv(
        debug=render,
        render_non_push_motions=False,
        render_every_n_steps=1,
        camera_heigh=800,
        camera_width=800
    )

    # Set target position
    env.set_target_state(target_position)
    
    # Create simple kinematic model for hover
    dynamics_model = HoverKinematicModel()
    dynamics_model.eval()
    
    # Create controller
    controller = PushingController(
        env=env,
        model=dynamics_model,
        cost_function=hover_cost_function,
        num_samples=20,  # Number of MPPI samples
        horizon=10,       # Planning horizon
        device=device
    )
    # import pdb; pdb.set_trace()


    
    # Reset environment
    state = env.reset()
    controller.reset()
    puck_position = env.get_puck_position() 
    target_position[0] = 0.5
    target_position[1] = puck_position[1]
    target_position[2] = 0.02
    controller.set_target_state(target_position)
    controller.set_target_state(target_position)
    default_params = [0.01, 2.5, 2.5, 2.5]  # [lambda, sigma_x, sigma_y, sigma_z]
    controller.set_parameters(default_params)
    env.set_target_state(target_position)
    # Initialize controller with default parameters    
    initial_distance = np.linalg.norm(state - target_position)
    print(TerminalColors.OKGREEN + f"Initial end-effector position: [{state[0]:.3f}, {state[1]:.3f}, {state[2]:.3f}]" + TerminalColors.ENDC)
    print(TerminalColors.OKGREEN + f"Initial distance to target: {initial_distance:.4f} m" + TerminalColors.ENDC)
    print()
    
    # Control loop
    print(TerminalColors.BOLD + "Starting control loop..." + TerminalColors.ENDC)
    print("-" * 60)
    
    for step in range(num_steps):
        puck_position = env.get_puck_position() 
        print(abs(puck_position[1] - target_position[1]))
        if abs(puck_position[1] - target_position[1]) > 0.1:
            # state = env.reset()
            controller.reset()
            target_position[0] = 0.5
            target_position[1] = puck_position[1]
            target_position[2] = 0.02
            controller.set_target_state(target_position)
            controller.set_target_state(target_position)
            default_params = [0.01, 2.5, 2.5, 2.5]  # [lambda, sigma_x, sigma_y, sigma_z]
            controller.set_parameters(default_params)
            env.set_target_state(target_position)
            hover_success = False

        # Get action from controller
        action = controller.control(state)
        
        # Execute action
        if hover_success:
            action = np.array([0.0, 0.0, 0.0])
        state, reward, done, info = env.step(action)
        
        # Calculate distance to target
        distance = np.linalg.norm(state - target_position)
        
        # Print progress every 5 steps
        if step % 5 == 0 or step == num_steps - 1:
            print(f"Step {step:3d}: Position [{state[0]:.3f}, {state[1]:.3f}, {state[2]:.3f}], "
                  f"Distance: {distance:.4f} m")
        
        # Check if goal reached
        dist_xy = np.linalg.norm(state[:2] - puck_position[:2])
        if dist_xy < 0.1:
            success = True
            print("Made contact with puck ==> Success")
            break

        if done:
            if distance < 0.05:  # 5cm tolerance
                hover_success = True
                # break
        if puck_position[0] <0.2:
            success = False
            print("Puck in goal zone => Failed")
            break
        
    
    print("-" * 60)
    
    # Final results
    final_distance = np.linalg.norm(state - target_position)
    reached_goal = final_distance < 0.05
    
    print(TerminalColors.BOLD + "Final Results:" + TerminalColors.ENDC)
    print(f"  Final position: [{state[0]:.3f}, {state[1]:.3f}, {state[2]:.3f}]")
    print(f"  Target position: [{target_position[0]:.3f}, {target_position[1]:.3f}, {target_position[2]:.3f}]")
    print(f"  Final distance: {final_distance:.4f} m")
    print(f"  puck position: [{puck_position[0]:.3f}, {puck_position[1]:.3f}, {puck_position[2]:.3f}]")

    
    # Cleanup
    if render:
        print("\nPress Enter to close the visualization window...")
        input()
        env.disconnect()
    
    return state, reached_goal


if __name__ == "__main__":
    # Configuration
    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
    RENDER = True
    
    # Example 1: Hover at a specific position
    print(TerminalColors.BOLD + "\n" + "="*60 + TerminalColors.ENDC)
    print(TerminalColors.BOLD + "="*60 + TerminalColors.ENDC)
    
    target_pos_1 = np.array([0.5, 0.0, 0.02])  # [x, y, z] in meters
    hover_at_position(target_pos_1, render=RENDER, device=DEVICE)
