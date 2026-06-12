import argparse
import base64
import io
import json
import math
import os
import re
import time
import csv
from contextlib import nullcontext
from datetime import datetime
from getpass import getpass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import mujoco
import numpy as np
import requests
from PIL import Image
from sshtunnel import SSHTunnelForwarder

from raccoon_env import SyncSimRaccoonEnv


# ──────────────────────────────────────────────
# 오브젝트 설정 (cylinder + sphere)
# ──────────────────────────────────────────────
CYLINDER_BODY_BY_COLOR = {
    "red": "target_object",
    "blue": "target_object_blue",
    "green": "target_object_green",
    "yellow": "target_object_yellow",
}

SPHERE_BODY_BY_COLOR = {
    "red": "target_sphere_red",
    "blue": "target_sphere_blue",
    "green": "target_sphere_green",
    "yellow": "target_sphere_yellow",
}

ALL_COLORS = ("red", "blue", "green", "yellow")
ALL_SHAPES = ("cylinder", "sphere")

DEFAULT_OBJECT_X_RANGE = (-0.10, 0.10)
DEFAULT_OBJECT_Y_RANGE = (0.16, 0.25)
DEFAULT_MIN_OBJECT_DISTANCE = 0.035
DEFAULT_YAW_RANGE = (-math.pi / 4, math.pi / 4)

# 언어 지시 템플릿 (각 태스크별 5종 — 데이터셋 생성 시 사용한 표현과 동일)
GRASP_VERBS = ["grasp", "pick up", "grab", "take", "get"]
LIFT_VERBS = ["lift", "raise", "pick up and hold", "elevate", "lift up"]

GRASP_TEMPLATES = {
    "cylinder": [f"{v} the {{color}} cylinder" for v in GRASP_VERBS],
    "sphere": [f"{v} the {{color}} sphere" for v in GRASP_VERBS],
}
LIFT_TEMPLATES = {
    "cylinder": [f"{v} the {{color}} cylinder" for v in LIFT_VERBS],
    "sphere": [f"{v} the {{color}} sphere" for v in LIFT_VERBS],
}


def build_instruction(task_type, target_shape, target_color,
                      instruction=None, variant=None, rng=None):
    """언어 지시문 생성.
    우선순위: 직접 입력(--instruction) > variant 선택 > 기본(대표 1종, index 0).
    variant: 0~4 정수 또는 'random'.
    """
    # 1) 사용자가 문장을 직접 준 경우 그대로 사용
    if instruction:
        return instruction

    templates = LIFT_TEMPLATES if task_type == "lift" else GRASP_TEMPLATES
    candidates = templates[target_shape]  # 5종 리스트

    # 2) variant 지정
    if variant is None:
        idx = 0  # 기본: 대표 표현 (하위 호환)
    elif str(variant).lower() == "random":
        r = rng if rng is not None else np.random.default_rng()
        idx = int(r.integers(0, len(candidates)))
    else:
        idx = int(variant) % len(candidates)

    return candidates[idx].format(color=target_color)


# ──────────────────────────────────────────────
# 타이밍 로거
# ──────────────────────────────────────────────
class TimingLogger:
    def __init__(self, log_dir: Path, episode_id: int):
        self.log_dir = log_dir
        self.episode_id = episode_id
        self.inference_times = []
        self.start_time = None
        self.end_time = None

        log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = log_dir / f"episode_{episode_id:06d}_timing.csv"
        self.summary_file = log_dir / f"episode_{episode_id:06d}_summary.json"

        with open(self.log_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "step", "inference_time_sec", "action_dx", "action_dy",
                "action_dz", "action_droll", "action_dpitch", "action_dyaw",
                "action_gripper", "timestamp"
            ])

    def start_episode(self):
        self.start_time = time.time()
        print(f"[TIMING] Episode {self.episode_id} 시작: {datetime.now().strftime('%H:%M:%S')}")

    def log_step(self, step_idx: int, inference_time: float, action: list):
        self.inference_times.append(inference_time)
        with open(self.log_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                step_idx, round(inference_time, 4),
                *[round(float(a), 6) for a in action],
                datetime.now().strftime("%H:%M:%S.%f")
            ])
        print(
            f"[TIMING] Step {step_idx:03d} | "
            f"추론시간: {inference_time:.3f}s | "
            f"action: [{', '.join(f'{float(a):.4f}' for a in action)}]"
        )

    def end_episode(self, total_steps: int):
        self.end_time = time.time()
        total_time = self.end_time - self.start_time
        avg = sum(self.inference_times) / len(self.inference_times) if self.inference_times else 0.0
        min_t = min(self.inference_times) if self.inference_times else 0.0
        max_t = max(self.inference_times) if self.inference_times else 0.0

        summary = {
            "episode_id": self.episode_id,
            "total_steps": total_steps,
            "total_time_sec": round(total_time, 2),
            "avg_inference_time_sec": round(avg, 4),
            "min_inference_time_sec": round(min_t, 4),
            "max_inference_time_sec": round(max_t, 4),
        }
        with open(self.summary_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        print("\n" + "="*60)
        print(f"[SUMMARY] Episode {self.episode_id} 완료")
        print(f"  총 스텝 수     : {total_steps}")
        print(f"  총 실행 시간   : {total_time:.2f}초")
        print(f"  평균 추론 시간 : {avg:.3f}초")
        print(f"  최소 추론 시간 : {min_t:.3f}초")
        print(f"  최대 추론 시간 : {max_t:.3f}초")
        print("="*60)


def image_to_b64(image_rgb: np.ndarray, quality: int = 50) -> str:
    buffer = io.BytesIO()
    Image.fromarray(image_rgb).save(buffer, format="JPEG", quality=quality)
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    png_buffer = io.BytesIO()
    Image.fromarray(image_rgb).save(png_buffer, format="PNG")
    print(f"[IMG] PNG: {len(png_buffer.getvalue())/1024:.1f}KB → JPEG: {len(buffer.getvalue())/1024:.1f}KB")
    return encoded


def request_action(
    server_url: str,
    instruction: str,
    image_rgb: np.ndarray,
    unnorm_key: Optional[str],
    timeout: float = 60.0,
    image_quality: int = 50,
) -> Tuple[Dict[str, Any], float]:
    payload = {
        "instruction": instruction,
        "image_b64": image_to_b64(image_rgb, quality=image_quality),
        "unnorm_key": unnorm_key,
        "do_sample": False,
    }
    t_start = time.time()
    response = requests.post(f"{server_url.rstrip('/')}/predict", json=payload, timeout=timeout)
    inference_time = time.time() - t_start
    if not response.ok:
        print(f"[SERVER ERROR] {response.status_code} | {response.text}")
        response.raise_for_status()
    return response.json(), inference_time


def resolve_ssh_password(args):
    if args.ssh_password:
        return args.ssh_password
    env_password = os.environ.get("OPENVLA_SSH_PASSWORD")
    if env_password:
        return env_password
    if args.use_ssh_tunnel and args.ssh_ask_password:
        return getpass("SSH password: ")
    return None


def open_ssh_tunnel(args):
    from sshtunnel import SSHTunnelForwarder
    ssh_password = resolve_ssh_password(args)
    tunnel = SSHTunnelForwarder(
        ssh_address_or_host=(args.ssh_host, args.ssh_port),
        ssh_username=args.ssh_user,
        ssh_password=ssh_password,
        remote_bind_address=(args.remote_server_host, args.remote_server_port),
        local_bind_address=(args.local_server_host, args.local_server_port),
    )
    tunnel.start()
    return tunnel


def build_server_url(args, tunnel):
    if tunnel is not None:
        return f"http://{args.local_server_host}:{tunnel.local_bind_port}"
    if not args.server_url:
        raise ValueError("--server_url is required.")
    return args.server_url


def maybe_tunnel_context(args):
    if args.use_ssh_tunnel:
        return open_ssh_tunnel(args)
    return nullcontext(None)


def print_success_log(step_idx, exec_info):
    final_delta_xyz = [round(float(v), 4) for v in exec_info["final_delta_xyz"]]
    move_xyz = [round(float(v), 4) for v in exec_info["actual_move_xyz"]]
    target_xyz = [round(float(v), 4) for v in exec_info["target_xyz"]]
    gripper = float(exec_info["gripper_cmd"])
    retries = int(exec_info["retry_count"])
    print(
        f"[{step_idx:03d}] OK | final_delta={final_delta_xyz} | "
        f"move={move_xyz} | target={target_xyz} | "
        f"gripper={gripper:.1f} | retries={retries}"
    )


def print_fail_log(step_idx, exc):
    print(f"[{step_idx:03d}] FAIL | {exc}")


def sample_object_specs(rng, x_range, y_range, min_distance, max_tries=1000):
    """cylinder + sphere 모두 랜덤 배치"""
    all_bodies = {}
    all_bodies.update({f"cylinder_{c}": CYLINDER_BODY_BY_COLOR[c] for c in ALL_COLORS})
    all_bodies.update({f"sphere_{c}": SPHERE_BODY_BY_COLOR[c] for c in ALL_COLORS})

    specs = {}
    placed_xy = []
    keys = list(all_bodies.keys())
    rng.shuffle(keys)

    for key in keys:
        for _ in range(max_tries):
            x = float(rng.uniform(x_range[0], x_range[1]))
            y = float(rng.uniform(y_range[0], y_range[1]))
            xy = np.array([x, y])
            if all(np.linalg.norm(xy - other) >= min_distance for other in placed_xy):
                specs[key] = {
                    "body_name": all_bodies[key],
                    "x": x, "y": y,
                    "yaw": float(rng.uniform(DEFAULT_YAW_RANGE[0], DEFAULT_YAW_RANGE[1])),
                }
                placed_xy.append(xy)
                break
        else:
            raise RuntimeError(f"배치 실패: {key}")
    return specs


def reset_extended_scene(env, object_specs, target_key):
    if target_key not in object_specs:
        raise ValueError(f"target_key={target_key} not in object_specs")

    target_spec = object_specs[target_key]
    env.reset_episode(float(target_spec["x"]), float(target_spec["y"]), float(target_spec["yaw"]))

    for key, spec in object_specs.items():
        body_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, spec["body_name"])
        if body_id == -1:
            continue
        jnt_adr = int(env.model.body_jntadr[body_id])
        qpos_adr = int(env.model.jnt_qposadr[jnt_adr])
        yaw = spec["yaw"]
        qw, qz = math.cos(yaw/2.0), math.sin(yaw/2.0)
        env.data.qpos[qpos_adr:qpos_adr+7] = np.array([spec["x"], spec["y"], 0.02, qw, 0.0, 0.0, qz])
        env.data.qvel[env.model.jnt_dofadr[jnt_adr]:env.model.jnt_dofadr[jnt_adr]+6] = 0.0

    target_body_name = target_spec["body_name"]
    if hasattr(env, "active_object_body_name"):
        env.active_object_body_name = target_body_name
    mujoco.mj_forward(env.model, env.data)


def clear_existing_images(out_dir):
    image_exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    deleted = 0
    for f in out_dir.iterdir():
        if f.is_file() and f.suffix.lower() in image_exts:
            f.unlink()
            deleted += 1
    print(f"[CLEANUP] removed {deleted} existing image files from {out_dir}")


def rollout(
    xml_path, server_url, target_color, target_shape, task_type,
    unnorm_key, output_dir, episode_id=1, max_steps=1000000,
    use_viewer=True, camera_name="front_view", speed=70,
    settle_seconds_per_action=0.5, initial_settle_seconds=0.3,
    delta_scale=1.0, request_timeout=60.0, max_delta_xyz=0.005,
    image_quality=50, seed=None,
    instruction=None, instruction_variant=None,
    object_x_range=DEFAULT_OBJECT_X_RANGE,
    object_y_range=DEFAULT_OBJECT_Y_RANGE,
    min_object_distance=DEFAULT_MIN_OBJECT_DISTANCE,
) -> None:

    out_dir = Path(output_dir) / f"episode_{episode_id:06d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    clear_existing_images(out_dir)

    log_dir = Path(output_dir) / "logs"
    timing_logger = TimingLogger(log_dir=log_dir, episode_id=episode_id)

    rng = np.random.default_rng(seed)

    # 타겟 키 생성
    target_key = f"{target_shape}_{target_color}"

    # 언어 지시 생성 (직접 입력 > variant 선택 > 기본 대표 1종)
    instruction = build_instruction(
        task_type=task_type, target_shape=target_shape, target_color=target_color,
        instruction=instruction, variant=instruction_variant, rng=rng,
    )

    # 오브젝트 배치
    object_specs = sample_object_specs(rng, object_x_range, object_y_range, min_object_distance)

    env = SyncSimRaccoonEnv(
        xml_path=xml_path, image_size=(256, 256),
        camera_name=camera_name, use_viewer=use_viewer,
    )

    try:
        reset_extended_scene(env=env, object_specs=object_specs, target_key=target_key)
        env.lockh()
        env.debug_check_current_ee_reachable()

        if initial_settle_seconds > 0:
            env.settle_steps(seconds=initial_settle_seconds)

        target_spec = object_specs[target_key]
        print(
            f"[SCENE] instruction={instruction!r} | target={target_key} | "
            f"target_xy=({target_spec['x']:.3f}, {target_spec['y']:.3f})"
        )

        timing_logger.start_episode()
        obs = env.get_observation()
        step_idx = 0

        while True:
            response, inference_time = request_action(
                server_url=server_url, instruction=instruction,
                image_rgb=obs["image"], unnorm_key=unnorm_key,
                timeout=request_timeout, image_quality=image_quality,
            )
            action = response["action"]
            timing_logger.log_step(step_idx, inference_time, action)

            try:
                exec_info = env.execute_delta_action7(
                    action=action, speed=speed, delta_scale=delta_scale,
                    max_delta_xyz=max_delta_xyz,
                )
                print_success_log(step_idx, exec_info)
                env.settle_steps(seconds=settle_seconds_per_action)
                obs = env.get_observation()
                Image.fromarray(obs["image"]).save(out_dir / f"frame_{step_idx:06d}.png")

            except Exception as exc:
                print_fail_log(step_idx, exc)
                obs = env.get_observation()
                Image.fromarray(obs["image"]).save(out_dir / f"frame_{step_idx:06d}_skipped.png")
                step_idx += 1
                if step_idx >= max_steps:
                    break
                continue

            step_idx += 1
            if step_idx >= max_steps:
                break

    except KeyboardInterrupt:
        print("\n[STOP] interrupted by user")
    finally:
        timing_logger.end_episode(total_steps=step_idx)
        env.close()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--xml_path", type=str, default="Raccoon_extended_scene.xml")
    parser.add_argument("--server_url", type=str, default=None)
    parser.add_argument("--target_color", type=str, default="red", choices=list(ALL_COLORS))
    parser.add_argument("--target_shape", type=str, default="cylinder", choices=["cylinder", "sphere"])
    parser.add_argument("--task_type", type=str, default="grasp", choices=["grasp", "lift"])
    parser.add_argument("--instruction", type=str, default=None,
                        help="언어 지시문을 직접 지정 (예: 'grab the red cylinder'). 주면 템플릿 무시.")
    parser.add_argument("--instruction_variant", type=str, default=None,
                        help="5종 표현 중 선택: 0~4 인덱스 또는 'random'. 미지정 시 대표 1종(0).")
    parser.add_argument("--unnorm_key", type=str, default="raccoon_pick_place")
    parser.add_argument("--output_dir", type=str, default="rollout_outputs")
    parser.add_argument("--episode_id", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=1000000)
    parser.add_argument("--speed", type=int, default=70)
    parser.add_argument("--settle_seconds_per_action", type=float, default=0.5)
    parser.add_argument("--initial_settle_seconds", type=float, default=0.3)
    parser.add_argument("--delta_scale", type=float, default=1.0)
    parser.add_argument("--max_delta_xyz", type=float, default=0.005)
    parser.add_argument("--request_timeout", type=float, default=60.0)
    parser.add_argument("--use_viewer", action="store_true")
    parser.add_argument("--camera_name", type=str, default="front_view")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--image_quality", type=int, default=50)
    parser.add_argument("--object_x_range", type=float, nargs=2, default=DEFAULT_OBJECT_X_RANGE)
    parser.add_argument("--object_y_range", type=float, nargs=2, default=DEFAULT_OBJECT_Y_RANGE)
    parser.add_argument("--min_object_distance", type=float, default=DEFAULT_MIN_OBJECT_DISTANCE)
    parser.add_argument("--use_ssh_tunnel", action="store_true")
    parser.add_argument("--ssh_host", type=str, default="qlak315.iptime.org")
    parser.add_argument("--ssh_port", type=int, default=24100)
    parser.add_argument("--ssh_user", type=str, default="root")
    parser.add_argument("--ssh_password", type=str, default=None)
    parser.add_argument("--ssh_ask_password", action="store_true")
    parser.add_argument("--remote_server_host", type=str, default="127.0.0.1")
    parser.add_argument("--remote_server_port", type=int, default=8000)
    parser.add_argument("--local_server_host", type=str, default="127.0.0.1")
    parser.add_argument("--local_server_port", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    with maybe_tunnel_context(args) as tunnel:
        server_url = build_server_url(args, tunnel)
        rollout(
            xml_path=args.xml_path,
            server_url=server_url,
            target_color=args.target_color,
            target_shape=args.target_shape,
            task_type=args.task_type,
            instruction=args.instruction,
            instruction_variant=args.instruction_variant,
            unnorm_key=args.unnorm_key,
            output_dir=args.output_dir,
            episode_id=args.episode_id,
            max_steps=args.max_steps,
            use_viewer=args.use_viewer,
            camera_name=args.camera_name,
            speed=args.speed,
            settle_seconds_per_action=args.settle_seconds_per_action,
            initial_settle_seconds=args.initial_settle_seconds,
            delta_scale=args.delta_scale,
            request_timeout=args.request_timeout,
            max_delta_xyz=args.max_delta_xyz,
            image_quality=args.image_quality,
            seed=args.seed,
            object_x_range=tuple(args.object_x_range),
            object_y_range=tuple(args.object_y_range),
            min_object_distance=args.min_object_distance,
        )


if __name__ == "__main__":
    main()