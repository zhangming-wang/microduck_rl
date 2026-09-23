# Microduck RL

This repository trains Microduck policies with mjlab/MuJoCo Warp and PPO. The
policies run at 50 Hz, are exported to ONNX, and are deployed by the runtime in
the `pollen-robotics/microduck` repository. Observation, actuator, and model
conventions are sim-to-real contracts; check them before changing a task.

## Setup and commands

- Use Python `>=3.12,<3.13` and `uv`; `uv sync` is the environment used by local
  runs and Hugging Face Jobs. Local-only packages or edits are not a valid
  verification of a training change.
- Local training needs a CUDA GPU. On Linux ARM machines, set
  `UV_HTTP_TIMEOUT=600` for the first `uv sync` because the CUDA wheels are
  large.
- `uv run list-envs` is the live task registry. Run a cheap smoke test before
  any long training run:

```bash
uv run train <TASK_ID> --env.scene.num-envs 64 --agent.max_iterations 5
uv run train <TASK_ID> --env.scene.num-envs 4096
uv run train <TASK_ID> --env.scene.num-envs 4096 --hf-jobs
uv run play <TASK_ID> --wandb-run-path <entity/project/run_id>
uv run scripts/export.py <TASK_ID> --wandb-run-path <entity/project/run_id> [--checkpoint N]
uv run scripts/infer_policy.py --walking output.onnx
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run --with pytest pytest tests/
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run --with pytest pytest tests/test_velstand_cfg.py -q
```

  `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` avoids an unrelated ROS
  `launch_testing` plugin present on some development hosts.

- Resume a run with `--agent.load-checkpoint model_N.pt --agent.resume True`.
  Use `--hf-jobs` for managed training; HF submission also supports
  `--namespace`, `--flavor`, `--timeout`, and `--dry-run`.
- `scripts/export.py` is the only checkpoint-to-ONNX path: it embeds the
  observation normalizer. `uv run publish` validates an existing ONNX or calls
  this exporter for a checkpoint, then writes a schema-2 manifest and performs
  an ONNX smoke check. Deployment-shaped publishing is, for example:

```bash
uv run publish --task <TASK_ID> --wandb-run-path <entity/project/run_id> \
  --checkpoint N --repo <user>/microduck-<name> \
  --kind episodic --duration-s 4.0 --dry-run
```

  Publishable policies must use constant commands; the graph must be `[1, 61] ->
  [1, 14]`. Use `scripts/infer_policy.py` for the CPU MuJoCo rehearsal; it uses
  the BAM M6 actuator model by default, while `--no-bam` selects XML PD.

## Code ownership

- Put custom rewards, observations, events, commands, and curricula in
  `src/mjlab_microduck/tasks/mdp.py`. It also owns the NaN-safe reward/advantage
  patches, passive-joint export filtering, and warm-start patch.
- Put one environment family per `src/mjlab_microduck/tasks/microduck_*_env_cfg.py`.
  Build from `make_microduck_velocity_env_cfg` when the task is locomotion so
  its DR, noise, delays, observations, and safety guards stay aligned.
- Register train/play configs and the runner in `src/mjlab_microduck/tasks/__init__.py`.
  Add a row to `_BACKLASH_TASKS` when a task has a backlash twin.
- `robot/microduck_constants.py` owns robot cfgs, HOME, actuator cfgs, and the
  authoritative MJCF mapping. The normal models are: velocity -> `robot_walk.xml`,
  VelStand -> `robot_allcollisions.xml`, stand/recovery and tricks ->
  `robot_groundcontact.xml`, and roller tasks -> `robot_groundcontact_rollers.xml`.
  Backlash variants use the matching `*_backlash.xml` model.
- `actuator/friction_dr_bam.py` owns BAM friction randomization and
  encoder-through-backlash behavior. `tests/` is CPU-oriented and contains cfg,
  MDP, ONNX-manifest, HF-hook, and architecture-invariant tests.

## Contracts and invariants

- The policy family has a shared 61-dimensional actor observation and 14 actions:
  48 proprioceptive values followed by `[twist(3), head_pose(4), body_pose(6)]`.
  Keep unused command slots in the layout and zero-pad them; do not delete a
  term. The isolated `testbench_env_cfg.py` is a one-DOF actuator test, not a
  policy-family template.
- The 14 servo order is left leg (hip_yaw, hip_roll, hip_pitch, knee, ankle),
  neck/head (neck_pitch, head_pitch, head_yaw, head_roll), then right leg in the
  same five-joint order. Roller and backlash models interleave unactuated joints:
  never use raw entity joint indices in MDP code. Use `_servo_joint_ids`,
  `_servo_joint_pos`, and related helpers in `mdp.py`.
- Unactuated joints start with `passive_` (wheels, jaw links, backlash hinges).
  Servo selectors normally use `^(?!passive_).*`; wheel selectors should be
  specific, such as `^passive_.*wheel`, so backlash hinges are not included.
- Normal robot tasks use the BAM voltage-controlled XL330 actuator. A standalone
  BAM cfg must register `expand_bam_friction_fields`; joint-friction DR must
  scale the actuator's `friction_scale`, because BAM zeroes MuJoCo's
  `dof_frictionloss`. The isolated testbench intentionally uses a different
  one-joint setup.
- Backlash adds one `passive_<servo>_backlash` hinge per servo with 2 degrees
  total play. Its actuator feedback and joint observations read the output-side
  encoder view (`servo + backlash`), while dimensions stay 14/61. A backlash task
  must use the same collision/model family as its base task.
- If an observation is remapped to a sensor or encoder view, tracking rewards on
  that quantity must use the same view. Otherwise the policy is penalized for
  correcting what it observes.
- Policy configs enable observation normalization. Never hand-convert a
  checkpoint or deploy an ONNX file that did not come from the exporter.
  Policies are unfiltered; adding an action low-pass requires a matched runtime
  behavior and transfer validation.

## Training pitfalls

- A cost function returning a non-negative value gets a negative reward weight.
  A Microduck self-negating `*_penalty`/`*_l1` returning a non-positive value
  gets a positive weight. Check that every weighted penalty logged as
  `Episode_Reward/<term>` is non-positive.
- Do not create per-step jackpots for reaching a pose or transition. Rate-limit
  or slew the target and use progress/potential shaping for recovery; do not make
  a positive recovery reward farmable merely by staying fallen or low.
- Keep motion-blocking regularizers light during dynamic skill discovery and add
  strong smoothness taxes only after the behavior exists. Compare total reward
  mass, not just copied coefficient values.
- Domain-randomization `dr.*` add/scale operations are reset-safe in mjlab 1.3.0;
  custom randomizers must restore defaults before applying a new sample.
- Curriculum steps are environment steps: one PPO iteration is 24 steps per
  environment. Use `microduck_mdp.reward_weight` for reward-weight schedules and
  separate parameter/event curricula for ranges.
- Managers deepcopy configs at initialization. During runtime curricula or eval
  overrides, mutate `env.event_manager.get_term_cfg(...)` or the corresponding
  reward manager term, not `env.cfg.events[...]`; the latter is a silent no-op.
- A normal resume restores the checkpoint's iteration and step counter. A
  cross-task warm start uses `MICRODUCK_WARM_START=1`, which resets those counters
  while keeping weights, normalizer, and optimizer. Collapse inherited curricula
  to their final stage with the existing `_collapse_curricula_to_final` pattern,
  and only warm-start tasks with compatible observation normalizers.

## Sim-to-real checks

- Before hardware testing, run the exported policy through `infer_policy.py` and
  verify the command encoding. In unified 13D command mode, SitStand uses
  `twist[0]` as a posture flag (`0 = stand`, `1 = sit`); an all-zero twist is
  the stand command.
- IMU orientation randomization is zero-centered and teaches tolerance to
  mounting-error magnitude; it cannot correct a fixed hardware mounting bias.
- For failures, inspect the actual checkpoint with headless rollouts and video,
  including the contact geom/axis, before changing rewards or curricula.
