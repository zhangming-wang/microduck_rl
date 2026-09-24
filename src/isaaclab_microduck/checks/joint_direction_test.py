"""Preview and check Microduck's joint order and movement directions.

Run from the repository root in the IsaacLab conda environment:

    python src/isaaclab_microduck/checks/joint_direction_test.py

Each joint moves toward HOME +/-0.1 rad in turn. Close the window to exit.
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
    PHYSICS_DT,
    SERVO_JOINT_NAMES,
    make_microduck_walk_cfg,
)


TEST_OFFSET = 0.10
TRIAL_STEPS = 120
HOME_STEPS = 40


def main() -> None:
    app = AppLauncher(headless=False, device=ISAAC_DEVICE).app
    try:
        import torch

        import isaaclab.sim as sim_utils
        from isaaclab.assets import Articulation

        sim = sim_utils.SimulationContext(
            sim_utils.SimulationCfg(dt=PHYSICS_DT, device=ISAAC_DEVICE)
        )
        sim.set_camera_view(eye=(0.65, 0.65, 0.45), target=(0.0, 0.0, 0.10))
        ground_cfg = sim_utils.GroundPlaneCfg(size=(4.0, 4.0))
        ground_cfg.func("/World/Ground", ground_cfg)
        light_cfg = sim_utils.DomeLightCfg(intensity=1800.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)
        robot = Articulation(make_microduck_walk_cfg())
        sim.reset()
        sim.play()

        isaac_ids = {name: index for index, name in enumerate(robot.joint_names)}
        missing = [name for name in SERVO_JOINT_NAMES if name not in isaac_ids]
        if missing:
            raise RuntimeError(f"Missing Microduck joints in IsaacLab: {missing}")
        print("[joint-test] canonical -> IsaacLab index:")
        for canonical_index, name in enumerate(SERVO_JOINT_NAMES):
            print(f"  {canonical_index:2d} {name:16s} -> {isaac_ids[name]:2d}")

        root_state = robot.data.default_root_state.clone()
        home = robot.data.default_joint_pos.clone()
        zero_vel = torch.zeros_like(home)
        target = home.clone()
        sim_dt = sim.get_physics_dt()

        def step_target(steps: int) -> bool:
            for _ in range(steps):
                if not app.is_running():
                    return False
                robot.set_joint_position_target(target)
                robot.write_data_to_sim()
                sim.step(render=True)
                robot.update(sim_dt)
            return True

        def reset_home() -> bool:
            nonlocal target
            if not app.is_running():
                return False
            robot.write_root_state_to_sim(root_state)
            robot.write_joint_state_to_sim(home, zero_vel)
            robot.reset()
            target = home.clone()
            return step_target(HOME_STEPS)

        print("[joint-test] showing +/-0.10 rad target offsets, one joint at a time", flush=True)
        failures = []
        for canonical_index, name in enumerate(SERVO_JOINT_NAMES):
            joint_id = isaac_ids[name]
            deltas = []
            for offset in (TEST_OFFSET, -TEST_OFFSET):
                if not reset_home():
                    return
                target[:, joint_id] += offset
                print(f"[joint-test] {canonical_index:2d} {name:16s} target {offset:+.2f} rad", flush=True)
                if not step_target(TRIAL_STEPS):
                    return
                actual_delta = (robot.data.joint_pos[0, joint_id] - home[0, joint_id]).item()
                deltas.append(actual_delta)
            positive_ok = deltas[0] > 0.0
            negative_ok = deltas[1] < 0.0
            status = "PASS" if positive_ok and negative_ok else "FAIL"
            if status == "FAIL":
                failures.append(name)
            print(
                f"[joint-test] {status} {canonical_index:2d} {name:16s} "
                f"+target_delta={deltas[0]: .4f} -target_delta={deltas[1]: .4f}",
                flush=True,
            )
        if not reset_home():
            return
        print(
            f"[joint-test] done: {len(SERVO_JOINT_NAMES) - len(failures)} passed, "
            f"{len(failures)} failed; holding HOME until the window closes",
            flush=True,
        )
        while app.is_running():
            if not step_target(1):
                break
    finally:
        app.close(wait_for_replicator=False, skip_cleanup=True)


if __name__ == "__main__":
    main()
