"""IsaacLab configuration for the imported Microduck walking asset."""

from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
ROBOT_USD = REPO_ROOT / "src/isaaclab_microduck/assets/robot_walk/robot_walk_isaac.usd"

# The installed workstation uses the RTX 5090 as CUDA device 0.
ISAAC_DEVICE = "cuda:0"
PHYSICS_DT = 0.005

# Simple implicit PD used only for the asset smoke tests. This is not BAM.
PD_STIFFNESS = 20.0
PD_DAMPING = 0.5
PD_EFFORT_LIMIT = 0.96

# Policy/action order from the MuJoCo Microduck contract. IsaacLab's imported
# joint tensor uses a different order; callers must map by these names.
SERVO_JOINT_NAMES = (
    "left_hip_yaw",
    "left_hip_roll",
    "left_hip_pitch",
    "left_knee",
    "left_ankle",
    "neck_pitch",
    "head_pitch",
    "head_yaw",
    "head_roll",
    "right_hip_yaw",
    "right_hip_roll",
    "right_hip_pitch",
    "right_knee",
    "right_ankle",
)

HOME_JOINT_POS = {
    "left_hip_yaw": 0.0,
    "left_hip_roll": -0.0873,
    "left_hip_pitch": -0.4579,
    "left_knee": -0.0049,
    "left_ankle": 0.4530,
    "neck_pitch": 0.3491,
    "head_pitch": 0.3491,
    "head_yaw": 0.0,
    "head_roll": 0.0,
    "right_hip_yaw": 0.0,
    "right_hip_roll": 0.0873,
    "right_hip_pitch": 0.4579,
    "right_knee": 0.0049,
    "right_ankle": -0.4530,
}


def make_microduck_walk_cfg():
    """Build the walking articulation config after Isaac Sim has launched."""
    import isaaclab.sim as sim_utils
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.assets import ArticulationCfg

    if not ROBOT_USD.is_file():
        raise FileNotFoundError(f"Robot USD not found: {ROBOT_USD}")

    return ArticulationCfg(
        prim_path="/World/Microduck",
        # The USD also contains an empty worldBody articulation marker. Select
        # the real 14-DOF root explicitly instead of relying on auto-discovery.
        articulation_root_prim_path="/trunk_base/trunk_base",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(ROBOT_USD),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                enable_gyroscopic_forces=True,
                max_depenetration_velocity=5.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=1,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            # robot_walk.xml places trunk_base at z=0.12 m in the source model.
            pos=(0.0, 0.0, 0.12),
            joint_pos=HOME_JOINT_POS,
            joint_vel={".*": 0.0},
        ),
        soft_joint_pos_limit_factor=0.9,
        actuators={
            "microduck_pd": ImplicitActuatorCfg(
                joint_names_expr=[".*"],
                effort_limit_sim=PD_EFFORT_LIMIT,
                stiffness=PD_STIFFNESS,
                damping=PD_DAMPING,
            ),
        },
    )
