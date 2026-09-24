"""Load the Microduck USD, initialize HOME, and hold it with simple PD.

This is an asset/control smoke test, not the PPO environment and not the BAM
actuator model used by the MuJoCo training repository.

Run from the repository root after activating the IsaacLab conda environment:

    python src/isaaclab_microduck/checks/home_preview.py
"""

from __future__ import annotations

from pathlib import Path
import sys

from isaaclab.app import AppLauncher

SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from isaaclab_microduck.assets.robot_walk_cfg import (  # noqa: E402
    ISAAC_DEVICE,
    PD_DAMPING,
    PD_EFFORT_LIMIT,
    PD_STIFFNESS,
    PHYSICS_DT,
    ROBOT_USD,
    make_microduck_walk_cfg,
)


def main() -> None:
    app_launcher = AppLauncher(headless=False, device=ISAAC_DEVICE)
    simulation_app = app_launcher.app

    try:
        import torch

        import isaaclab.sim as sim_utils
        from isaaclab.assets import Articulation

        sim_cfg = sim_utils.SimulationCfg(dt=PHYSICS_DT, device=ISAAC_DEVICE)
        sim = sim_utils.SimulationContext(sim_cfg)
        sim.set_camera_view(eye=(0.65, 0.65, 0.45), target=(0.0, 0.0, 0.10))

        ground_cfg = sim_utils.GroundPlaneCfg(size=(4.0, 4.0))
        ground_cfg.func("/World/Ground", ground_cfg)
        light_cfg = sim_utils.DomeLightCfg(intensity=1800.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        robot = Articulation(make_microduck_walk_cfg())

        sim.reset()

        # Write HOME explicitly so this test does not depend only on the
        # articulation config's reset callback.
        root_state = robot.data.default_root_state.clone()
        joint_pos = robot.data.default_joint_pos.clone()
        joint_vel = torch.zeros_like(joint_pos)
        robot.write_root_state_to_sim(root_state)
        robot.write_joint_state_to_sim(joint_pos, joint_vel)
        robot.reset()
        sim.play()

        print(f"[home-preview] USD: {ROBOT_USD}")
        print(f"[home-preview] root: {robot.num_joints} DOFs")
        print(f"[home-preview] joints: {robot.joint_names}")
        print(f"[home-preview] HOME: {joint_pos[0].detach().cpu().tolist()}")
        print(
            f"[home-preview] device={ISAAC_DEVICE}, dt={PHYSICS_DT}, "
            f"PD: stiffness={PD_STIFFNESS}, damping={PD_DAMPING}, "
            f"effort_limit={PD_EFFORT_LIMIT}"
        )

        target = joint_pos.clone()
        sim_dt = sim.get_physics_dt()
        step = 0
        while simulation_app.is_running():
            robot.set_joint_position_target(target)
            robot.write_data_to_sim()
            sim.step(render=True)
            robot.update(sim_dt)

            if step == 0 or step % max(1, int(1.0 / sim_dt)) == 0:
                actual = robot.data.joint_pos[0]
                error = torch.max(torch.abs(actual - target[0])).item()
                root_z = robot.data.root_link_pos_w[0, 2].item()
                print(
                    f"[home-preview] t={step * sim_dt:6.2f}s "
                    f"root_z={root_z: .4f} max_joint_error={error: .4f}",
                    flush=True,
                )
            step += 1
        print(f"[home-preview] closed after {step} simulation steps", flush=True)
    finally:
        # A separate Isaac/Replicator process may hold the shared Kit cache;
        # waiting for its shutdown can block a short standalone smoke test.
        simulation_app.close(wait_for_replicator=False, skip_cleanup=True)


if __name__ == "__main__":
    main()
