import os
import json
import math
import shutil
from pathlib import Path

os.environ["MUJOCO_GL"] = "egl"

import mujoco
import mujoco.viewer
import numpy as np
from PIL import Image


# ──────────────────────────────────────────────
# 언어 지시 템플릿
# ──────────────────────────────────────────────
INSTRUCTION_TEMPLATES = {
    "cylinder": [
        "lift the {color} cylinder",
        "raise the {color} cylinder",
        "pick up and hold the {color} cylinder",
        "elevate the {color} cylinder",
        "lift up the {color} cylinder",
    ],
    "sphere": [
        "lift the {color} sphere",
        "raise the {color} sphere",
        "pick up and hold the {color} sphere",
        "elevate the {color} sphere",
        "lift up the {color} sphere",
    ],
}

# ──────────────────────────────────────────────
# 오브젝트 설정
# ──────────────────────────
CYLINDER_BODY_MAP = {
    "red":    "target_object",
    "blue":   "target_object_blue",
    "yellow": "target_object_yellow",
    "green":  "target_object_green",
}

SPHERE_BODY_MAP = {
    "red":    "target_sphere_red",
    "blue":   "target_sphere_blue",
    "green":  "target_sphere_green",
    "yellow": "target_sphere_yellow",
}

COLORS = ("red", "blue", "green", "yellow")

DEFAULT_X_RANGE = (-0.10, 0.10)
DEFAULT_Y_RANGE = (0.15, 0.25)
DEFAULT_MIN_DIST = 0.035
LIFT_HEIGHT = 0.12  # 들어올릴 높이 (m)
LIFT_SUCCESS_THRESHOLD = 0.08  # 성공 판단 높이 (m)


class DatasetLogger:
    def __init__(self, root_dir="dataset_raw", keep_failed=False):
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.keep_failed = keep_failed
        self.episode_dir = None
        self.meta = None

    def start_episode(self, episode_id, instruction, goal_xy, box_init_xy,
                      box_init_yaw, task_type="lift", target_color=None,
                      target_body_name=None, all_object_init_poses=None):
        episode_name = f"episode_{episode_id:06d}"
        self.episode_dir = self.root_dir / episode_name
        if self.episode_dir.exists():
            shutil.rmtree(self.episode_dir, ignore_errors=True)
        self.episode_dir.mkdir(parents=True, exist_ok=True)
        self.meta = {
            "episode_id": int(episode_id),
            "instruction": str(instruction),
            "task_type": str(task_type),
            "goal_xy": [float(goal_xy[0]), float(goal_xy[1])],
            "box_init_xy": [float(box_init_xy[0]), float(box_init_xy[1])],
            "box_init_yaw": float(box_init_yaw),
            "success": False,
            "steps": []
        }
        if target_color is not None:
            self.meta["target_color"] = str(target_color)
        if target_body_name is not None:
            self.meta["target_body_name"] = str(target_body_name)
        if all_object_init_poses is not None:
            self.meta["all_object_init_poses"] = all_object_init_poses

    def log_step(self, step_idx, image_rgb, joint_angles, gripper_state,
                 object_pose, ee_pose, action, is_first=False, is_last=False):
        image_file = f"frame_{step_idx:06d}.png"
        Image.fromarray(image_rgb).save(self.episode_dir / image_file)
        self.meta["steps"].append({
            "t": int(step_idx),
            "image_file": image_file,
            "joint_angles": [float(x) for x in joint_angles],
            "gripper_state": float(gripper_state),
            "object_pose": [float(x) for x in object_pose],
            "ee_pose": [float(x) for x in ee_pose],
            "action": [float(x) for x in action],
            "is_first": bool(is_first),
            "is_last": bool(is_last),
        })

    def finalize_episode(self, success, exception_text=None):
        self.meta["success"] = bool(success)
        if exception_text is not None:
            self.meta["exception"] = str(exception_text)
        with open(self.episode_dir / "meta.json", "w", encoding="utf-8") as f:
            json.dump(self.meta, f, indent=2, ensure_ascii=False)
        if (not success) and (not self.keep_failed):
            shutil.rmtree(self.episode_dir, ignore_errors=True)

    def abort_episode(self):
        if self.episode_dir is not None and self.episode_dir.exists():
            shutil.rmtree(self.episode_dir, ignore_errors=True)


class LiftSyncSimRaccoon:
    MAX_SPEEDS = [2.2, 2.3, 2.3, 2.3]
    GRIPPER_SPEED = 15.0
    L1, L2, L3, L4 = 8.25, 10.0, 10.0, 8.0
    MODE_POSITION = 0
    GRIP_OPEN = 0.15701
    GRIP_CLOSE = -0.85
    GRIP_MODE_FREE = 0
    GRIP_MODE_HORZ = 1

    def __init__(self, xml_path, image_size=(256, 256), camera_name=None, use_viewer=False):
        if not os.path.exists(xml_path):
            raise FileNotFoundError(f"xml 파일을 찾을 수 없습니다: {xml_path}")
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, height=image_size[1], width=image_size[0])
        self.camera_name = camera_name
        self.use_viewer = use_viewer
        self.viewer = None
        if self.use_viewer:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        self.target_angles = [0.0] * 4
        self.current_setpoints = [0.0] * 5
        self.joint_velocities = [0.0] * 4
        self.joint_control_mode = [self.MODE_POSITION] * 4
        self.gripper_target = self.GRIP_OPEN
        self.gripper_mode = self.GRIP_MODE_FREE
        self.active_object_body_name = CYLINDER_BODY_MAP["red"]
        for i in range(4):
            self.joint_velocities[i] = self.MAX_SPEEDS[i] * 0.7

    def _calc_inv_kinematics(self, x, y, z):
        if (-28.0 <= x <= 28.0) and (-15 <= y <= 28.0) and (0 <= z <= 36.25):
            x, y = y, -x
            th1 = math.atan2(y, x)
            c1, s1 = math.cos(th1), math.sin(th1)
            x = x - self.L4 * c1
            y = y - self.L4 * s1
            zL1 = z - self.L1
            c3 = (x*x + y*y + zL1*zL1 - self.L2*self.L2 - self.L3*self.L3) / (2*self.L2*self.L3)
            c32 = min(c3*c3, 1.0)
            s3 = -math.sqrt(1 - c32)
            th3 = math.atan2(s3, c3)
            M1 = c3*self.L3 + self.L2
            M2 = z - self.L1
            M3 = s3*self.L3
            M4 = c1*x + s1*y
            th2 = math.atan2(-M2*M3-M1*M4, M1*M2-M3*M4)
            th1, th2, th3 = math.degrees(th1), math.degrees(th2), math.degrees(th3)
            th4 = -(th2 + th3) - 90
            if not (-120 <= th1 <= 120): return None
            if not (-90 <= th2 <= 30): return None
            if not (-150 <= th3 <= 0): return None
            return [th1, th2, th3, th4]
        return None

    def degree_to(self, joints, degrees, speed=70):
        j_list = joints if isinstance(joints, (list, tuple)) else [joints]
        d_list = degrees if isinstance(degrees, (list, tuple)) else [degrees]
        if len(d_list) == 1 and len(j_list) > 1:
            d_list = d_list * len(j_list)
        for j, deg in zip(j_list, d_list):
            idx = j - 1
            if 0 <= idx < 4:
                self.joint_control_mode[idx] = self.MODE_POSITION
                self.target_angles[idx] = np.radians(deg)
                self.joint_velocities[idx] = np.clip(speed, 0, 100) / 100.0 * self.MAX_SPEEDS[idx]

    def move_to(self, x_cm, y_cm, z_cm, speed=70):
        angles = self._calc_inv_kinematics(x_cm, y_cm, z_cm)
        if angles is None:
            raise ValueError(f"도달할 수 없는 좌표: ({x_cm:.2f}, {y_cm:.2f}, {z_cm:.2f}) cm")
        self.degree_to([1, 2, 3, 4], angles[:4], speed)

    def open_gripper(self): self.gripper_target = self.GRIP_OPEN
    def close_gripper(self): self.gripper_target = self.GRIP_CLOSE
    def lockh(self): self.gripper_mode = self.GRIP_MODE_HORZ

    def execute_action(self, action, speed=70):
        tx, ty, tz, gripper = action
        self.move_to(tx * 100.0, ty * 100.0, tz * 100.0, speed=speed)
        if gripper >= 0.5:
            self.close_gripper()
        else:
            self.open_gripper()

    def _apply_controls_once(self):
        dt = self.model.opt.timestep
        for i in range(4):
            if i == 3 and self.gripper_mode != self.GRIP_MODE_FREE:
                base_angle = -(self.current_setpoints[1] + self.current_setpoints[2])
                desired = base_angle - np.radians(90)
                error = desired - self.current_setpoints[i]
                step = np.clip(error, -self.MAX_SPEEDS[i]*dt, self.MAX_SPEEDS[i]*dt)
                self.current_setpoints[i] += step
            else:
                error = self.target_angles[i] - self.current_setpoints[i]
                if abs(error) > 1e-4:
                    step = np.clip(error, -abs(self.joint_velocities[i])*dt, abs(self.joint_velocities[i])*dt)
                    self.current_setpoints[i] += step
            joint_id = self.model.actuator_trnid[i, 0]
            rng = self.model.jnt_range[joint_id]
            self.current_setpoints[i] = np.clip(self.current_setpoints[i], rng[0], rng[1])
            self.data.ctrl[i] = self.current_setpoints[i]

        try:
            touch_L = self.data.sensor("sensor_L").data[0]
            touch_R = self.data.sensor("sensor_R").data[0]
            is_touched = (touch_L > 0.1) and (touch_R > 0.1)
        except Exception:
            is_touched = False

        if self.gripper_target == self.GRIP_CLOSE and is_touched:
            self.gripper_target = self.data.qpos[4] - 0.028

        g_err = self.gripper_target - self.current_setpoints[4]
        if abs(g_err) > 1e-4:
            g_move = np.clip(g_err, -self.GRIPPER_SPEED*dt, self.GRIPPER_SPEED*dt)
            self.current_setpoints[4] += g_move
        self.data.ctrl[4] = self.current_setpoints[4]

    def step_n(self, n_steps):
        for _ in range(int(n_steps)):
            self._apply_controls_once()
            mujoco.mj_step(self.model, self.data)
            if self.viewer is not None and self.viewer.is_running():
                self.viewer.sync()

    def steps_for_seconds(self, seconds):
        return max(1, int(round(seconds / self.model.opt.timestep)))

    def settle_steps(self, seconds=2.0):
        self.step_n(self.steps_for_seconds(seconds))

    def get_robot_state(self):
        return {
            "joint_angles": [float(self.data.qpos[i]) for i in range(4)],
            "gripper_state": float(self.data.qpos[4])
        }

    def get_object_pose(self, body_name):
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body_id == -1:
            raise ValueError(f"body not found: {body_name}")
        pos = self.data.xpos[body_id].copy()
        xmat = self.data.xmat[body_id].reshape(3, 3).copy()
        yaw = math.atan2(xmat[1, 0], xmat[0, 0])
        return np.array([pos[0], pos[1], pos[2], yaw], dtype=np.float32)

    def render_rgb(self):
        cam_id = self.camera_name if self.camera_name is not None else -1
        self.renderer.update_scene(self.data, camera=cam_id)
        return self.renderer.render().copy()

    def get_observation(self, object_body_name=None):
        if object_body_name is None:
            object_body_name = self.active_object_body_name
        rs = self.get_robot_state()
        obj = self.get_object_pose(object_body_name)
        img = self.render_rgb()
        link4_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "Link4")
        ee_pose_list = list(self.data.xpos[link4_id].copy()) if link4_id != -1 else [0.0, 0.0, 0.0]
        return {
            "image": img,
            "joint_angles": rs["joint_angles"],
            "gripper_state": rs["gripper_state"],
            "object_pose": obj,
            "ee_pose": ee_pose_list,
        }

    def reset_object_pose(self, body_name, x=0.15, y=0.15, z=0.02, yaw=0.0):
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body_id == -1:
            return
        jnt_adr = self.model.body_jntadr[body_id]
        if self.model.body_jntnum[body_id] < 1:
            return
        qpos_adr = self.model.jnt_qposadr[jnt_adr]
        qw, qz = math.cos(yaw/2.0), math.sin(yaw/2.0)
        self.data.qpos[qpos_adr:qpos_adr+7] = np.array([x, y, z, qw, 0.0, 0.0, qz])
        self.data.qvel[self.model.jnt_dofadr[jnt_adr]:self.model.jnt_dofadr[jnt_adr]+6] = 0.0

    def reset_episode(self, cylinder_specs, sphere_specs, target_body_name):
        home = np.radians([0.0, -10.0, -140.0, 60.0])
        for i in range(4):
            self.data.qpos[i] = home[i]
            self.data.ctrl[i] = home[i]
            self.current_setpoints[i] = home[i]
            self.target_angles[i] = home[i]
            self.joint_control_mode[i] = self.MODE_POSITION
        self.data.qvel[:] = 0.0
        self.data.qpos[4] = self.GRIP_OPEN
        self.data.ctrl[4] = self.GRIP_OPEN
        self.current_setpoints[4] = self.GRIP_OPEN
        self.gripper_target = self.GRIP_OPEN
        self.gripper_mode = self.GRIP_MODE_FREE
        self.active_object_body_name = target_body_name

        for color, spec in cylinder_specs.items():
            self.reset_object_pose(spec["body_name"], x=spec["x"], y=spec["y"], z=0.02, yaw=spec["yaw"])
        for color, spec in sphere_specs.items():
            self.reset_object_pose(spec["body_name"], x=spec["x"], y=spec["y"], z=0.02, yaw=0.0)

        mujoco.mj_forward(self.model, self.data)
        self.step_n(20)

    def is_lift_success(self, target_body_name, height_threshold=LIFT_SUCCESS_THRESHOLD):
        """오브젝트가 일정 높이 이상 들어올려졌는지 확인"""
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, target_body_name)
        if body_id == -1:
            return False
        obj_z = float(self.data.xpos[body_id][2])
        gripper_closing = float(self.data.qpos[4]) < (self.GRIP_OPEN - 0.01)
        return bool(obj_z >= height_threshold and gripper_closing)

    def make_lift_plan(self, box_x, box_y):
        """
        lift trajectory:
        1. 오브젝트 위로 이동 (그리퍼 열기)
        2. 내려가서 잡기
        3. 들어올리기
        """
        return [
            [box_x, box_y, 0.10, 0],          # 오브젝트 위로
            [box_x, box_y, 0.02, 0],          # 내려가기
            [box_x, box_y, 0.02, 1],          # 
            [box_x, box_y, LIFT_HEIGHT, 1],   # 들어올리기
        ]

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None


def sample_specs(rng, body_map, x_range, y_range, min_distance, placed_xy=None, max_tries=1000):
    specs = {}
    placed_xy = list(placed_xy or [])
    colors = list(body_map.keys())
    rng.shuffle(colors)
    for color in colors:
        for _ in range(max_tries):
            x = float(rng.uniform(x_range[0], x_range[1]))
            y = float(rng.uniform(y_range[0], y_range[1]))
            xy = np.array([x, y])
            if all(np.linalg.norm(xy - other) >= min_distance for other in placed_xy):
                specs[color] = {
                    "body_name": body_map[color],
                    "x": x, "y": y,
                    "yaw": float(rng.uniform(-np.pi/4, np.pi/4)),
                }
                placed_xy.append(xy)
                break
        else:
            raise RuntimeError(f"배치 실패: {color}")
    return specs, placed_xy


def run_episode(rc, logger, episode_id, instruction, cylinder_specs, sphere_specs,
                target_color, target_shape, speed=70, settle_secs=0.8,
                initial_settle=0.3, hz=10, height_threshold=LIFT_SUCCESS_THRESHOLD):

    if target_shape == "cylinder":
        target_body_name = CYLINDER_BODY_MAP[target_color]
        target_spec = cylinder_specs[target_color]
    else:
        target_body_name = SPHERE_BODY_MAP[target_color]
        target_spec = sphere_specs[target_color]

    target_x = float(target_spec["x"])
    target_y = float(target_spec["y"])
    target_yaw = float(target_spec["yaw"])

    rc.reset_episode(cylinder_specs, sphere_specs, target_body_name)
    rc.lockh()
    if initial_settle > 0:
        rc.settle_steps(seconds=initial_settle)

    all_poses = {}
    for c, s in cylinder_specs.items():
        all_poses[f"{c}_cylinder"] = {"body_name": s["body_name"], "xy": [s["x"], s["y"]], "yaw": s["yaw"]}
    for c, s in sphere_specs.items():
        all_poses[f"{c}_sphere"] = {"body_name": s["body_name"], "xy": [s["x"], s["y"]], "yaw": s["yaw"]}

    logger.start_episode(
        episode_id=episode_id, instruction=instruction, task_type="lift",
        goal_xy=[target_x, target_y], box_init_xy=[target_x, target_y],
        box_init_yaw=target_yaw, target_color=f"{target_color}_{target_shape}",
        target_body_name=target_body_name, all_object_init_poses=all_poses,
    )

    try:
        plan = rc.make_lift_plan(target_x, target_y)
        obs = rc.get_observation()
        step_counter = 0
        dt = 1.0 / hz

        for action in plan:
            rc.execute_action(action, speed=speed)
            for _ in range(int(settle_secs * hz)):
                logger.log_step(step_counter, obs["image"], obs["joint_angles"],
                                obs["gripper_state"], obs["object_pose"], obs["ee_pose"],
                                action, is_first=(step_counter == 0), is_last=False)
                rc.settle_steps(seconds=dt)
                obs = rc.get_observation()
                step_counter += 1

        logger.log_step(step_counter, obs["image"], obs["joint_angles"],
                        obs["gripper_state"], obs["object_pose"], obs["ee_pose"],
                        plan[-1], is_first=False, is_last=True)

        success = rc.is_lift_success(target_body_name, height_threshold)
        logger.finalize_episode(success=success)
        return success
    except Exception as e:
        logger.abort_episode()
        raise e


def collect_dataset(
    xml_path="Raccoon_extended_scene.xml",
    dataset_root="raccoon_lift",
    num_episodes=300,
    keep_failed=False,
    use_viewer=False,
    camera_name="front_view",
    speed=150,
    settle_secs=0.8,
    initial_settle=0.1,
    hz=10,
    height_threshold=LIFT_SUCCESS_THRESHOLD,
    seed=None,
    x_range=(-0.10, 0.10),
    y_range=(0.15, 0.25),
    min_dist=0.035,
):
    rng = np.random.default_rng(seed)
    max_attempts = max(num_episodes * 20, 500)

    task_keys = [(color, shape) for shape in ["cylinder", "sphere"] for color in COLORS]
    base = num_episodes // len(task_keys)
    remainder = num_episodes % len(task_keys)
    target_counts = {k: base + (1 if idx < remainder else 0) for idx, k in enumerate(task_keys)}
    success_counts = {k: 0 for k in task_keys}

    print(f"Target counts: {target_counts}")

    rc = LiftSyncSimRaccoon(xml_path=xml_path, image_size=(256, 256),
                             camera_name=camera_name, use_viewer=use_viewer)
    logger = DatasetLogger(root_dir=dataset_root, keep_failed=keep_failed)

    attempt_count = 0
    episode_id = 0

    try:
        while sum(success_counts.values()) < num_episodes and attempt_count < max_attempts:
            attempt_count += 1

            remaining = {k: target_counts[k] - success_counts[k] for k in task_keys
                        if target_counts[k] - success_counts[k] > 0}
            if not remaining:
                break

            weights = np.array(list(remaining.values()), dtype=np.float64)
            weights /= weights.sum()
            idx = int(rng.choice(len(remaining), p=weights))
            target_color, target_shape = list(remaining.keys())[idx]

            templates = INSTRUCTION_TEMPLATES[target_shape]
            instruction = str(rng.choice(templates)).format(color=target_color)

            try:
                cylinder_specs, placed = sample_specs(rng, CYLINDER_BODY_MAP, x_range, y_range, min_dist)
                sphere_specs, _ = sample_specs(rng, SPHERE_BODY_MAP, x_range, y_range, min_dist, placed_xy=placed)

                episode_id += 1
                success = run_episode(
                    rc, logger, episode_id, instruction,
                    cylinder_specs, sphere_specs, target_color, target_shape,
                    speed=speed, settle_secs=settle_secs, initial_settle=initial_settle,
                    hz=hz, height_threshold=height_threshold,
                )

                if success:
                    success_counts[(target_color, target_shape)] += 1

                print(
                    f"[Attempt {attempt_count:04d}] episode={episode_id:06d} | "
                    f"target='{target_color}_{target_shape}' | "
                    f"instruction='{instruction}' | success={success} | "
                    f"total={sum(success_counts.values())}/{num_episodes}"
                )

            except Exception as e:
                print(f"[Attempt {attempt_count:04d}] exception: {e}")

    finally:
        rc.close()

    print(f"\n완료! total={sum(success_counts.values())}/{num_episodes}, attempts={attempt_count}")
    print(f"오브젝트별 성공: {success_counts}")


if __name__ == "__main__":
    collect_dataset(
        xml_path="/data/Raccoonbot_Openvla/Mujoco/Raccoon_extended_scene.xml",
        dataset_root="/data/Raccoonbot_Openvla/Mujoco/raccoon_lift",
        num_episodes=300,
        keep_failed=False,
        use_viewer=False,
        camera_name="front_view",
        initial_settle=0.1,
        x_range=(-0.10, 0.10),
        y_range=(0.15, 0.25),
        min_dist=0.035,
    )
