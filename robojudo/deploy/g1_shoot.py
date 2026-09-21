"""One observation/action contract for G1 basketball in MuJoCo and on hardware."""

from __future__ import annotations

import math
import os
import struct
import time
import xml.etree.ElementTree as ET
from collections import deque
from pathlib import Path

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[2]
FIELDS = ("phase", "gravity", "angular_velocity", "joint_position", "joint_velocity", "applied_action", "hoop_position")
SIZES = (1, 9, 9, 87, 87, 87, 3)


def load_config(path: str | Path, onnx: str | None = None) -> dict:
    path = Path(path).expanduser().resolve()
    cfg = yaml.safe_load(path.read_text())
    if not isinstance(cfg, dict):
        raise ValueError("Deployment YAML must contain a mapping")
    model = Path(onnx).expanduser().resolve() if onnx else (path.parent / cfg["onnx"]).resolve()
    if not model.is_file():
        raise FileNotFoundError(f"ONNX policy missing: {model}")
    cfg["onnx"] = str(model)
    names = cfg["joint_names"]
    if len(names) != 29 or len(set(names)) != 29:
        raise ValueError("Expected 29 unique G1 joint names")
    for key in ("nominal_joint_position", "first_frame_joint_position", "safe_lower_joint_position",
                "safe_upper_joint_position", "pd_stiffness", "pd_damping"):
        value = np.asarray(cfg[key], dtype=np.float64)
        if value.shape != (29,) or not np.all(np.isfinite(value)):
            raise ValueError(f"{key} must have 29 finite values")
    lower = np.asarray(cfg["safe_lower_joint_position"])
    upper = np.asarray(cfg["safe_upper_joint_position"])
    first = np.asarray(cfg["first_frame_joint_position"])
    if np.any(lower >= upper) or np.any(first < lower) or np.any(first > upper):
        raise ValueError("Invalid safe limits or first frame outside limits")
    if cfg["policy_hz"] <= 0 or cfg["phase_frames"] < 2:
        raise ValueError("policy_hz must be positive and phase_frames at least 2")
    if set(cfg["observation_scales"]) != set(FIELDS):
        raise ValueError(f"observation_scales requires {FIELDS}")
    for key in ("initial_hoop_position_b", "initial_ball_position_b"):
        value = np.asarray(cfg[key], dtype=float)
        if value.shape != (3,) or not np.all(np.isfinite(value)):
            raise ValueError(f"{key} must contain three finite coordinates")
    return cfg


def gravity_body(quat_wxyz: np.ndarray) -> np.ndarray:
    """World down in the pelvis frame, for a unit wxyz orientation."""
    w, x, y, z = np.asarray(quat_wxyz, dtype=np.float64)
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm < 0.5 or not np.isfinite(norm):
        raise ValueError("Invalid pelvis IMU quaternion")
    w, x, y, z = (w / norm, x / norm, y / norm, z / norm)
    return np.array([-2 * (x * z - w * y), -2 * (y * z + w * x), -(1 - 2 * (x * x + y * y))], dtype=np.float32)


def rotate_world(quat_wxyz: np.ndarray, vector: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    return Rotation.from_quat(np.asarray(quat_wxyz)[[1, 2, 3, 0]]).apply(vector)


class ShootingPolicy:
    def __init__(self, cfg: dict):
        os.environ.setdefault("ORT_DISABLE_TELEMETRY", "1")
        import onnxruntime as ort

        self.cfg = cfg
        self.nominal = np.asarray(cfg["nominal_joint_position"], dtype=np.float32)
        self.first = np.asarray(cfg["first_frame_joint_position"], dtype=np.float32)
        self.lower = np.asarray(cfg["safe_lower_joint_position"], dtype=np.float32)
        self.upper = np.asarray(cfg["safe_upper_joint_position"], dtype=np.float32)
        self.scales = cfg["observation_scales"]
        self.hoop = np.asarray(cfg["initial_hoop_position_b"], dtype=np.float32)
        self.session = ort.InferenceSession(cfg["onnx"], providers=["CPUExecutionProvider"])
        inputs, outputs = self.session.get_inputs(), self.session.get_outputs()
        if len(inputs) != 1 or len(outputs) != 1 or inputs[0].shape[-1] != 283 or outputs[0].shape[-1] != 29:
            raise ValueError("ONNX contract must be one 283-D input and one 29-D output")
        self.input_name = inputs[0].name
        self.gravity = deque(maxlen=3)
        self.omega = deque(maxlen=3)
        self.q = deque(maxlen=3)
        self.dq = deque(maxlen=3)
        self.actions = deque(maxlen=3)
        self.frame = 0

    def reset(self, state: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]) -> None:
        self.frame = 0
        self.gravity.clear(); self.omega.clear(); self.q.clear(); self.dq.clear(); self.actions.clear()
        self._append_sensor(state)
        for history in (self.gravity, self.omega, self.q, self.dq):
            while len(history) < 3:
                history.append(history[0].copy())
        # The controller has already applied the first-frame target before A.
        initial_action = self.first - self.nominal
        for _ in range(3):
            self.actions.append(initial_action.copy())

    def _append_sensor(self, state: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]) -> None:
        q, dq, quat, omega = (np.asarray(x, dtype=np.float32) for x in state)
        if q.shape != (29,) or dq.shape != (29,) or omega.shape != (3,):
            raise ValueError("Invalid G1 sensor dimensions")
        if not all(np.all(np.isfinite(x)) for x in (q, dq, quat, omega)):
            raise ValueError("Nonfinite G1 sensor reading")
        self.gravity.append(gravity_body(quat))
        self.omega.append(omega.copy())
        self.q.append(q - self.nominal)
        self.dq.append(dq.copy())

    def observation(self) -> np.ndarray:
        s = self.scales
        phase = min(self.frame / (self.cfg["phase_frames"] - 1), 1.0)
        parts = (
            np.array([phase * s["phase"]], dtype=np.float32),
            np.concatenate(self.gravity) * s["gravity"],
            np.concatenate(self.omega) * s["angular_velocity"],
            np.concatenate(self.q) * s["joint_position"],
            np.concatenate(self.dq) * s["joint_velocity"],
            np.concatenate(self.actions) * s["applied_action"],
            self.hoop * s["hoop_position"],
        )
        obs = np.concatenate(parts).astype(np.float32)
        if obs.shape != (sum(SIZES),):
            raise RuntimeError("Observation size changed")
        return obs

    def step(self, state: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]) -> np.ndarray:
        self._append_sensor(state)
        action = np.asarray(self.session.run(None, {self.input_name: self.observation()[None]})[0][0], dtype=np.float32)
        if action.shape != (29,) or not np.all(np.isfinite(action)):
            raise ValueError("Policy returned invalid actions")
        target = np.clip(self.nominal + action, self.lower, self.upper)
        self.actions.append(target - self.nominal)  # actual applied offset, after clipping
        self.frame += 1
        return target


def make_scene(cfg: dict):
    """Compose a new scene in memory; the checked-in robot XML is untouched."""
    import mujoco

    robot_file = ROOT / "assets/robots/g1/g1_bb_sdf.xml"
    robot = ET.parse(robot_file).getroot()
    robot.find("compiler").set("meshdir", str(robot_file.parent / "meshes"))
    world = robot.find("worldbody")
    ball = ET.parse(ROOT / "assets/objects/basketball.xml").getroot().find("worldbody/body")
    world.append(ball)
    root_pos = np.asarray(cfg["sim"]["root_position"], dtype=float)
    root_quat = np.asarray(cfg["sim"]["root_quaternion_wxyz"], dtype=float)
    hoop = root_pos + rotate_world(root_quat, np.asarray(cfg["initial_hoop_position_b"]))
    for i in range(24):
        a, b = 2 * np.pi * np.array([i, i + 1]) / 24
        p0 = hoop + np.array([0.23 * np.cos(a), 0.23 * np.sin(a), 0])
        p1 = hoop + np.array([0.23 * np.cos(b), 0.23 * np.sin(b), 0])
        ET.SubElement(world, "geom", name=f"hoop_ring_{i}", type="capsule", fromto=" ".join(map(str, [*p0, *p1])), size="0.012", rgba="0.9 0.18 0.03 1", contype="0", conaffinity="0")
    model = mujoco.MjModel.from_xml_string(ET.tostring(robot, encoding="unicode"))
    model.opt.timestep = float(cfg["sim"]["physics_dt"])
    data = mujoco.MjData(model)
    return model, data


class ShootingSim:
    def __init__(self, cfg: dict):
        import mujoco

        self.mj = mujoco
        self.cfg = cfg
        self.model, self.data = make_scene(cfg)
        self.joints = [self.model.joint(name).id for name in cfg["joint_names"]]
        self.actuators = [self.model.actuator(name).id for name in cfg["joint_names"]]
        self.qadr = self.model.jnt_qposadr[self.joints]
        self.dadr = self.model.jnt_dofadr[self.joints]
        self.ball_adr = self.model.jnt_qposadr[self.model.joint("basketball_free").id]
        self.pelvis = self.model.body("pelvis").id
        self.kp = np.asarray(cfg["pd_stiffness"])
        self.kd = np.asarray(cfg["pd_damping"])
        self.substeps = round(1 / (cfg["policy_hz"] * self.model.opt.timestep))
        if not np.isclose(self.substeps * self.model.opt.timestep, 1 / cfg["policy_hz"]):
            raise ValueError("physics_dt must divide the policy period")
        self.set_first_pose()

    def set_first_pose(self) -> None:
        self.mj.mj_resetData(self.model, self.data)
        root_pos = np.asarray(self.cfg["sim"]["root_position"])
        root_quat = np.asarray(self.cfg["sim"]["root_quaternion_wxyz"])
        self.data.qpos[:7] = [*root_pos, *root_quat]
        self.data.qpos[self.qadr] = self.cfg["first_frame_joint_position"]
        ball_pos = root_pos + rotate_world(root_quat, np.asarray(self.cfg["initial_ball_position_b"]))
        self.data.qpos[self.ball_adr:self.ball_adr + 7] = [*ball_pos, 1, 0, 0, 0]
        self.mj.mj_forward(self.model, self.data)

    def read(self):
        velocity = np.zeros(6)
        self.mj.mj_objectVelocity(self.model, self.data, self.mj.mjtObj.mjOBJ_BODY, self.pelvis, velocity, 1)
        return (self.data.qpos[self.qadr].astype(np.float32).copy(),
                self.data.qvel[self.dadr].astype(np.float32).copy(),
                self.data.xquat[self.pelvis].astype(np.float32).copy(),
                velocity[:3].astype(np.float32).copy())

    def step(self, target, perturb=None):
        for _ in range(self.substeps):
            q = self.data.qpos[self.qadr]
            dq = self.data.qvel[self.dadr]
            self.data.ctrl[self.actuators] = self.kp * (target - q) - self.kd * dq
            # The passive viewer only updates MjvPerturb. User-driven physics
            # loops must explicitly turn that mouse displacement into force.
            # Clear first so a force cannot remain on a previously selected body.
            self.data.xfrc_applied[:] = 0.0
            if perturb is not None:
                self.mj.mjv_applyPerturbForce(self.model, self.data, perturb)
            self.mj.mj_step(self.model, self.data)


class ShootingReal:
    """Direct SDK2 low-level PD transport. The main thread is the sole command writer."""

    def __init__(self, cfg: dict):
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber
        from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_, unitree_hg_msg_dds__LowState_
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
        from unitree_sdk2py.utils.crc import CRC

        self.cfg = cfg
        self.state = None
        self.state_time = 0.0
        self.crc = CRC()
        self.indices = cfg["real"]["motor_indices"]
        if len(self.indices) != 29 or len(set(self.indices)) != 29 or any(i < 0 or i >= 29 for i in self.indices):
            raise ValueError("motor_indices must map uniquely to G1 motors 0..28")
        ChannelFactoryInitialize(0, cfg["real"]["network_interface"])
        self.cmd = unitree_hg_msg_dds__LowCmd_()
        self.cmd.mode_pr = 0
        for motor in self.cmd.motor_cmd:
            motor.mode = 1
        self.publisher = ChannelPublisher("rt/lowcmd", LowCmd_)
        self.publisher.Init()
        self.subscriber = ChannelSubscriber("rt/lowstate", LowState_)
        self.subscriber.Init(self._receive, 10)
        self.kp = np.asarray(cfg["pd_stiffness"])
        self.kd = np.asarray(cfg["pd_damping"])

    def _receive(self, state):
        self.state = state
        self.state_time = time.monotonic()

    def fresh_state(self):
        state = self.state
        if state is None or time.monotonic() - self.state_time > self.cfg["real"]["state_timeout_s"]:
            raise RuntimeError("G1 lowstate missing or stale")
        return state

    def buttons(self) -> int:
        return struct.unpack_from("<H", bytes(self.fresh_state().wireless_remote), 2)[0]

    def read(self):
        state = self.fresh_state()
        q = np.array([state.motor_state[i].q for i in self.indices], dtype=np.float32)
        dq = np.array([state.motor_state[i].dq for i in self.indices], dtype=np.float32)
        quat = np.asarray(state.imu_state.quaternion, dtype=np.float32)
        gyro = np.asarray(state.imu_state.gyroscope, dtype=np.float32)
        return q, dq, quat, gyro

    def _send(self, require_fresh: bool = True):
        state = self.fresh_state() if require_fresh else self.state
        if state is None:
            return
        self.cmd.mode_machine = state.mode_machine
        self.cmd.crc = self.crc.Crc(self.cmd)
        self.publisher.Write(self.cmd)

    def zero(self):
        for motor in self.cmd.motor_cmd:
            motor.q = motor.dq = motor.kp = motor.kd = motor.tau = 0.0
        self._send()

    def position(self, target):
        if not np.all(np.isfinite(target)):
            raise ValueError("Nonfinite PD target")
        for j, idx in enumerate(self.indices):
            motor = self.cmd.motor_cmd[idx]
            motor.q = float(target[j]); motor.dq = 0.0
            motor.kp = float(self.kp[j]); motor.kd = float(self.kd[j]); motor.tau = 0.0
        self._send()

    def damping(self):
        for motor in self.cmd.motor_cmd:
            motor.q = motor.dq = motor.kp = motor.tau = 0.0
            motor.kd = 8.0
        if self.state is not None:
            self._send(require_fresh=False)


def run_sim(cfg: dict, check_steps: int = 0) -> None:
    sim = ShootingSim(cfg)
    policy = ShootingPolicy(cfg)
    if check_steps:
        policy.reset(sim.read())
        for _ in range(check_steps):
            sim.step(policy.step(sim.read()))
        ball_height = float(sim.data.qpos[sim.ball_adr + 2])
        pelvis_height = float(sim.data.qpos[2])
        print(f"sim2sim ran {check_steps} steps; phase={policy.observation()[0]:.3f}; "
              f"ball_z={ball_height:.3f} m; pelvis_z={pelvis_height:.3f} m")
        if ball_height < 0.2 or pelvis_height < 0.4:
            print("Physical outcome: ball on floor or robot down; shot NOT validated")
        return

    import glfw
    import mujoco.viewer

    keys: deque[int] = deque()
    def on_key(key: int):
        keys.append(key)

    stage = "waiting"
    print("MuJoCo: S = first frame, A = start policy, Esc = exit")
    print("Mouse force: double-click a body, then Ctrl+drag it (Ctrl+right-drag translates)")
    with mujoco.viewer.launch_passive(sim.model, sim.data, key_callback=on_key) as viewer:
        viewer.cam.lookat[:] = sim.data.qpos[:3]
        viewer.cam.distance = 4.0
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_PERTFORCE] = 1
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_PERTOBJ] = 1
        next_tick = time.monotonic()
        while viewer.is_running():
            while keys:
                key = keys.popleft()
                if key == glfw.KEY_ESCAPE:
                    return
                if key == glfw.KEY_S and stage == "waiting":
                    sim.set_first_pose(); stage = "ready"
                    print("First frame set. Press A to run.")
                if key == glfw.KEY_A and stage == "ready":
                    policy.reset(sim.read()); stage = "running"
                    print("Policy running; phase advances to 1 and stays there.")
            if stage == "running":
                # Viewer input lives on its render thread. Hold its lock while
                # consuming the selected body and perturbation reference.
                with viewer.lock():
                    sim.step(policy.step(sim.read()), viewer.perturb)
            viewer.sync()
            next_tick += 1 / cfg["policy_hz"]
            time.sleep(max(0.0, next_tick - time.monotonic()))
            if time.monotonic() - next_tick > 0.1:
                next_tick = time.monotonic()


def run_real(cfg: dict) -> None:
    real = ShootingReal(cfg)
    policy = ShootingPolicy(cfg)
    period = 1 / cfg["policy_hz"]
    print("G1: debug mode (L2+R2) first; Start = first pose, A = policy, Select = damping/exit")
    try:
        while real.state is None:
            time.sleep(period)
        stage = "waiting"
        first = np.asarray(cfg["first_frame_joint_position"], dtype=np.float32)
        next_tick = time.monotonic()
        previous_buttons = 0
        while True:
            buttons = real.buttons()
            pressed = buttons & ~previous_buttons
            previous_buttons = buttons
            if pressed & (1 << 3):  # Select, official exit key
                break
            if stage == "waiting":
                real.zero()
                if pressed & (1 << 2):  # Start
                    initial = real.read()[0]
                    move_steps = max(1, round(cfg["real"]["move_to_first_pose_s"] * cfg["policy_hz"]))
                    move_tick = 0
                    stage = "moving"
                    print("Moving to first-frame joint pose")
            elif stage == "moving":
                move_tick += 1
                alpha = min(move_tick / move_steps, 1.0)
                real.position(initial * (1 - alpha) + first * alpha)
                if alpha >= 1.0:
                    stage = "ready"
                    print("First frame reached. Place the ball and press A.")
            elif stage == "ready":
                real.position(first)
                if pressed & (1 << 8):  # A
                    policy.reset(real.read())
                    stage = "running"
                    print("Policy running")
            else:
                real.position(policy.step(real.read()))
            next_tick += period
            time.sleep(max(0.0, next_tick - time.monotonic()))
            if time.monotonic() - next_tick > cfg["real"]["state_timeout_s"]:
                raise RuntimeError("Control loop missed state timeout")
    finally:
        for _ in range(10):
            real.damping()
            time.sleep(period)
        print("Damping mode")
