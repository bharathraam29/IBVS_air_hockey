import numpy as np
import pybullet as p

from env.panda_pushing_env import PandaHoverEnv
from controller.pushing_controller import PushingController, hover_cost_function
from model.state_dynamics_models import HoverKinematicModel

MAX_STEPS = 500  # run long enough to see multiple bounces

def main():
    render = True
    device = "cpu"

    # 1. Create environment
    env = PandaHoverEnv(
        debug=render,
        render_non_push_motions=True,
        render_every_n_steps=1,
        camera_heigh=800,
        camera_width=800
    )

    # 2. Create dynamics model and controller ONCE
    dynamics_model = HoverKinematicModel()
    dynamics_model.eval()

    controller = PushingController(
        env=env,
        model=dynamics_model,
        cost_function=hover_cost_function,
        num_samples=500,  # MPPI samples
        horizon=10,       # planning horizon
        device=device
    )

    # 3. Set default controller parameters ONCE
    default_params = [0.01, 2.5, 2.5, 2.5]  # [lambda, sigma_x, sigma_y, sigma_z]
    controller.set_parameters(default_params)

    # 4. Reset env and controller ONCE
    state = env.reset()   # state is EE position [x, y, z]
    controller.reset()

    # Blocking parameters
    z_block_height = 0.10   # hover ~10 cm above table
    block_radius = 0.06     # within 6 cm in XY means "blocked"

    num_blocks = 0
    # Cooldown so we don't bounce the puck every frame while they overlap
    block_cooldown_steps = 15
    last_block_step = -block_cooldown_steps

    for step in range(MAX_STEPS):
        # --- Get puck position from env ---
        puck_position = env.get_puck_position()   # [x, y, z]

        # --- Define EE target: hover above puck ---
        target_pos = puck_position.copy()
        target_pos[2] = z_block_height

        # Update target in both env (for its done condition) and controller
        env.set_target_state(target_pos)
        controller.set_target_state(target_pos)

        # --- Ask controller for an action ---
        action = controller.control(state)  # action = [dx, dy, dz]

        # OPTIONAL: make it more goalie-like (only y,z motion)
        # action[0] = 0.0

        # --- Step the environment ---
        state, reward, done, info = env.step(action)
        ee_pos = state  # EE position [x, y, z]

        # --- Compute distance in XY plane between EE and puck ---
        dist_xy = np.linalg.norm(ee_pos[:2] - puck_position[:2])

        if step % 10 == 0:
            print(
                f"Step {step:3d}: "
                f"EE [{ee_pos[0]:.3f}, {ee_pos[1]:.3f}, {ee_pos[2]:.3f}], "
                f"Puck [{puck_position[0]:.3f}, {puck_position[1]:.3f}, {puck_position[2]:.3f}], "
                f"XY dist: {dist_xy:.4f} m"
            )

        # --- Block detection + bounce ---
        if dist_xy < block_radius and (step - last_block_step) >= block_cooldown_steps:
            num_blocks += 1
            last_block_step = step
            print(f"✓ Block #{num_blocks} at step {step} (XY distance {dist_xy:.3f} m)")

            # Get current puck velocity
            v = env.get_puck_velocity()
            vx, vy, vz = v

            # If velocity is almost zero (puck "dead"), give it a default direction
            speed_xy = np.linalg.norm(v[:2])
            if speed_xy < 1e-3:
                # Send it back towards +x with some speed
                v_new = np.array([0.15, 0.0, 0.0], dtype=np.float32)
            else:
                # Simple "bounce": flip x-component, keep y, shrink a bit
                v_new = np.array([-vx, vy, 0.0], dtype=np.float32) * 0.9

            env.set_puck_velocity(v_new)

        # Don't break on block or done; we want continuous play

    print(f"\nSimulation finished. Total blocks: {num_blocks}")

    # Cleanup
    if render:
        input("\nPress Enter to close the simulation...")
    p.stopStateLogging(env.video_id)
    p.disconnect()

if __name__ == "__main__":
    main()
