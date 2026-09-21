# G1 EDU 29 DoF basketball student deployment

The deployment uses one independent YAML file, one ONNX policy, and the robot's existing MuJoCo XML. It does not load the training framework, its Python classes, or its JSON metadata. The basketball scene is assembled in memory from the original robot XML and [basketball.xml](../assets/objects/basketball.xml); no original model file is changed.

## Setup

Use the project's Python environment with `numpy`, `scipy`, `pyyaml`, `onnxruntime`, and `mujoco`. For hardware, install Unitree's `unitree_sdk2py` as described in [unitree_setup.md](unitree_setup.md). Edit [g1_shoot.yaml](../robojudo/deploy/g1_shoot.yaml):

- `onnx`: a path relative to this YAML, or supply `--onnx /path/to/new.onnx`. The example ONNX is copied into [assets/models/g1/shoot_student.onnx](../assets/models/g1/shoot_student.onnx), so deployment has no runtime dependency on the training repository.
- `first_frame_joint_position`: editable hardcoded first-frame joint angles, in the YAML's `joint_names` order.
- `initial_hoop_position_b`: fixed hoop-center position in the **initial pelvis frame**. Measure and update it for the real basket. The example is taken from training clip 0. `initial_ball_position_b` places the free basketball in sim.
- `phase_frames`: number of 100 Hz frames until phase reaches 1. The example is 210 frames. After reaching 1, the policy keeps running, with no automatic stop or reset.
- `nominal_joint_position`, safe limits, PD gains, and observation scales: values matching the exported student. When replacing the policy, update these values if its training contract differs.

The observation is 283 floats: phase (1), then 3 old-to-new frames each of projected gravity (9), pelvis local angular velocity (9), joint position minus nominal (87), joint velocity (87), **applied** action minus nominal (87), then initial pelvis-frame hoop position (3). The output is a 29-vector of radians relative to nominal. The absolute PD target is `clip(nominal + output, safe_lower, safe_upper)`. No reference trajectory, residual action, artificial sensor/action delay, noise, or domain randomization is used.

## Sim2sim

From the repository root:

```bash
python scripts/deploy_g1_shoot.py check
python scripts/deploy_g1_shoot.py sim
```

In the MuJoCo window, press **S** to set the first frame, **A** to start, and **Esc** to exit. To apply an external force while the policy is running, double-click a robot link or the ball to select it, then hold **Ctrl** and drag. **Ctrl + right drag** translates in the vertical plane; **Ctrl + Shift + right drag** translates in the horizontal plane. The perturbation arrow is shown in the viewer.

The basketball is a free 0.625 kg sphere of 0.246 m diameter. The orange ring indicates the configured hoop center; it is a visual target, not a full hoop collision model. The simulation starts directly at the editable first-frame root position and joint angles. The basketball remains at its initial pose until policy execution begins. This simple MuJoCo scene has no grasp constraint. With the example first-frame pose, the ball falls early and the robot later loses balance. The viewer exposes that sim2sim result; it does not establish a successful shot. Check contact and first-frame support before interpreting policy performance.

## Sim2real

The [official Unitree low-level sequence](https://github.com/unitreerobotics/unitree_rl_gym/blob/main/deploy/deploy_real/README.md) applies: suspend the robot, enter debug/damping mode with **L2+R2**, connect the SDK to the correct network interface, then run:

```bash
python scripts/deploy_g1_shoot.py real --net eth0 --onnx /path/to/student_policy.onnx
```

The G1 remote uses the same workflow as Unitree's deployment example: **Start** interpolates all 29 joints to `first_frame_joint_position` over 2 s; while holding that pose, place the ball; **A** starts the policy at phase 0; **Select** exits to damping. The control loop runs at 100 Hz with SDK2 `rt/lowstate` and `rt/lowcmd`, using the G1 pelvis IMU and [official motor indices](https://github.com/unitreerobotics/unitree_sdk2_python/blob/master/example/g1/low_level/g1_low_level_example.py) 0–28. A missing/stale low state or invalid policy/sensor value exits to damping. Interrupting with Ctrl+C also enters damping. The first-frame posture, physical ball support, and basket placement must be checked on the suspended robot before releasing support. The current sim2sim result shows loss of balance, so physical execution should wait for that mismatch to be understood.

The `real.motor_indices` mapping and `joint_names` must agree with the robot firmware and ONNX export. This implementation has been exercised in MuJoCo and against the local ONNX artifact; sending live commands to physical hardware requires the actual G1 connection and operator.
