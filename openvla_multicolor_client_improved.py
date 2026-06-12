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


CYLINDER_BODY_BY_COLOR = {
    "red": "target_object",
    "blue": "target_object_blue",
    "green": "target_object_green",
    "yellow": "target_object_yellow",
}
CYLINDER_COLORS = tuple(CYLINDER_BODY_BY_COLOR.keys())

DEFAULT_OBJECT_X_RANGE = (-0.10, 0.10)
DEFAULT_OBJECT_Y_RANGE = (0.16, 0.25)
DEFAULT_MIN_OBJECT_DISTANCE = 0.035
DEFAULT_YAW_RANGE = (-math.pi / 4, math.pi / 4)
DEFAULT_INSTRUCTION_TEMPLATE = "grasp the {color} cylinder"


# ──────────────────────────────────────────────
# [IMPROVEMENT 1] 타이밍 로거 클래스 추가
# ──────────────────────────────────────────────
class TimingLogger:
    """추론 시간 및 액션 로그를 기록하는 클래스"""

    def __init__(self, log_dir: Path, episode_id: int):
        self.log_dir = log_dir
        self.episode_id = episode_id
        self.inference_times = []
        self.action_log = []
        self.start_time = None
        self.end_time = None

        log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = log_dir / f"episode_{episode_id:06d}_timing.csv"
        self.summary_file = log_dir / f"episode_{episode_id:06d}_summary.json"

        # CSV 헤더 작성
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
        self.action_log.append({
            "step": step_idx,
            "inference_time": inference_time,
            "action": action
        })

        # CSV에 기록
        with open(self.log_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                step_idx,
                round(inference_time, 4),
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

        if self.inference_times:
            avg_inference = sum(self.inference_times) / len(self.inference_times)
            min_inference = min(self.inference_times)
            max_inference = max(self.inference_times)
        else:
            avg_inference = min_inference = max_inference = 0.0

        summary = {
            "episode_id": self.episode_id,
            "total_steps": total_steps,
            "total_time_sec": round(total_time, 2),
            "avg_inference_time_sec": round(avg_inference, 4),
            "min_inference_time_sec": round(min_inference, 4),
            "max_inference_time_sec": round(max_inference, 4),
            "start_time": datetime.fromtimestamp(self.start_time).strftime("%H:%M:%S"),
            "end_time": datetime.fromtimestamp(self.end_time).strftime("%H:%M:%S"),
        }

        with open(self.summary_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        print("\n" + "="*60)
        print(f"[SUMMARY] Episode {self.episode_id} 완료")
        print(f"  총 스텝 수     : {total_steps}")
        print(f"  총 실행 시간   : {total_time:.2f}초")
        print(f"  평균 추론 시간 : {avg_inference:.3f}초")
        print(f"  최소 추론 시간 : {min_inference:.3f}초")
        print(f"  최대 추론 시간 : {max_inference:.3f}초")
        print("="*60)

        return summary


def image_to_b64(image_rgb: np.ndarray, quality: int = 50) -> str:
    """JPEG 압축으로 이미지 크기 줄여서 전송 속도 향상"""
    buffer = io.BytesIO()
    # PNG → JPEG 변환 (quality=50으로 용량 대폭 감소)
    Image.fromarray(image_rgb).save(buffer, format="JPEG", quality=quality)
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    
    # 크기 비교 로그
    png_buffer = io.BytesIO()
    Image.fromarray(image_rgb).save(png_buffer, format="PNG")
    print(f"[IMG] PNG: {len(png_buffer.getvalue())/1024:.1f}KB → JPEG: {len(buffer.getvalue())/1024:.1f}KB")
    
    return encoded
# ──────────────────────────────────────────────
# [IMPROVEMENT 2] 추론 시간 측정 추가
# ──────────────────────────────────────────────
def request_action(
    server_url: str,
    instruction: str,
    image_rgb: np.ndarray,
    unnorm_key: Optional[str],
    timeout: float = 60.0,
    image_quality: int = 50,  # [IMPROVEMENT] JPEG 품질 파라미터 추가
) -> Tuple[Dict[str, Any], float]:
    payload = {
        "instruction": instruction,
        "image_b64": image_to_b64(image_rgb, quality=image_quality),  # quality 전달
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

def resolve_ssh_password(args: argparse.Namespace) -> Optional[str]:
    if args.ssh_password:
        return args.ssh_password
    env_password = os.environ.get("OPENVLA_SSH_PASSWORD")
    if env_password:
        return env_password
    if args.use_ssh_tunnel and args.ssh_ask_password:
        return getpass("SSH password: ")
    return None


def open_ssh_tunnel(args: argparse.Namespace) -> SSHTunnelForwarder:
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


def build_server_url(args: argparse.Namespace, tunnel: Optional[SSHTunnelForwarder]) -> str:
    if tunnel is not None:
        return f"http://{args.local_server_host}:{tunnel.local_bind_port}"
    if not args.server_url:
        raise ValueError("--server_url is required when --use_ssh_tunnel is not enabled.")
    return args.server_url


def maybe_tunnel_context(args: argparse.Namespace):
    if args.use_ssh_tunnel:
        return open_ssh_tunnel(args)
    return nullcontext(None)


def print_success_log(step_idx: int, exec_info: Dict[str, Any]) -> None:
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


def print_fail_log(step_idx: int, exc: Exception) -> None:
    print(f"[{step_idx:03d}] FAIL | {exc}")


def infer_color_from_instruction(instruction: Optional[str]) -> Optional[str]:
    if not instruction:
        return None
    text = instruction.lower()
    matches = []
    for color in CYLINDER_COLORS:
        if re.search(rf"\b{re.escape(color)}\b", text):
            matches.append(color)
    if len(matches) > 1:
        raise ValueError(f"instruction에 여러 색상이 있습니다: {matches}")
    return matches[0] if matches else None


def resolve_target_color_and_instruction(
    instruction: Optional[str],
    target_color_arg: str,
    rng: np.random.Generator,
    instruction_template: str,
) -> Tuple[str, str]:
    instruction_color = infer_color_from_instruction(instruction)
    if instruction_color is not None:
        target_color = instruction_color
    elif target_color_arg in CYLINDER_COLORS:
        target_color = target_color_arg
    elif target_color_arg in ("auto", "random"):
        target_color = str(rng.choice(CYLINDER_COLORS))
    else:
        raise ValueError(f"지원하지 않는 --target_color: {target_color_arg}")
    if instruction is None or instruction.strip() == "":
        instruction = instruction_template.format(color=target_color)
    return target_color, instruction


def make_default_object_specs() -> Dict[str, Dict[str, float]]:
    x_values = np.linspace(
        DEFAULT_OBJECT_X_RANGE[0] * 0.75,
        DEFAULT_OBJECT_X_RANGE[1] * 0.75,
        len(CYLINDER_COLORS),
    )
    y_center = float(sum(DEFAULT_OBJECT_Y_RANGE) / 2.0)
    return {
        color: {
            "body_name": CYLINDER_BODY_BY_COLOR[color],
            "x": float(x_values[idx]),
            "y": y_center,
            "yaw": 0.0,
        }
        for idx, color in enumerate(CYLINDER_COLORS)
    }


def sample_object_specs(
    rng: np.random.Generator,
    x_range: Tuple[float, float] = DEFAULT_OBJECT_X_RANGE,
    y_range: Tuple[float, float] = DEFAULT_OBJECT_Y_RANGE,
    yaw_range: Tuple[float, float] = DEFAULT_YAW_RANGE,
    min_distance: float = DEFAULT_MIN_OBJECT_DISTANCE,
    max_tries: int = 1000,
) -> Dict[str, Dict[str, float]]:
    specs: Dict[str, Dict[str, float]] = {}
    placed_xy = []
    placement_order = list(CYLINDER_COLORS)
    rng.shuffle(placement_order)
    for color in placement_order:
        for _ in range(max_tries):
            x = float(rng.uniform(x_range[0], x_range[1]))
            y = float(rng.uniform(y_range[0], y_range[1]))
            xy = np.array([x, y], dtype=np.float64)
            if all(np.linalg.norm(xy - other_xy) >= min_distance for other_xy in placed_xy):
                specs[color] = {
                    "body_name": CYLINDER_BODY_BY_COLOR[color],
                    "x": x, "y": y,
                    "yaw": float(rng.uniform(yaw_range[0], yaw_range[1])),
                }
                placed_xy.append(xy)
                break
        else:
            raise RuntimeError(f"객체 배치 실패: {color}")
    return {color: specs[color] for color in CYLINDER_COLORS}


def reset_freejoint_body_pose(env, body_name, x, y, z, yaw):
    body_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if body_id == -1:
        raise ValueError(f"body not found: {body_name}")
    jnt_adr = int(env.model.body_jntadr[body_id])
    qpos_adr = int(env.model.jnt_qposadr[jnt_adr])
    qw = math.cos(yaw / 2.0)
    qz = math.sin(yaw / 2.0)
    env.data.qpos[qpos_adr:qpos_adr + 7] = np.array([x, y, z, qw, 0.0, 0.0, qz], dtype=np.float64)
    qvel_adr = int(env.model.jnt_dofadr[jnt_adr])
    env.data.qvel[qvel_adr:qvel_adr + 6] = 0.0


def reset_multicolor_scene(env, object_specs, target_color):
    if target_color not in object_specs:
        raise ValueError(f"target_color={target_color}가 object_specs에 없습니다.")
    target_spec = object_specs[target_color]
    env.reset_episode(float(target_spec["x"]), float(target_spec["y"]), float(target_spec["yaw"]))
    for color, spec in object_specs.items():
        reset_freejoint_body_pose(
            env=env, body_name=str(spec["body_name"]),
            x=float(spec["x"]), y=float(spec["y"]), z=0.02, yaw=float(spec["yaw"]),
        )
    target_body_name = str(target_spec["body_name"])
    if hasattr(env, "active_object_body_name"):
        env.active_object_body_name = target_body_name
    if hasattr(env, "target_body_name"):
        env.target_body_name = target_body_name
    mujoco.mj_forward(env.model, env.data)


def object_specs_to_meta(object_specs):
    return {
        color: {
            "body_name": str(spec["body_name"]),
            "xy": [float(spec["x"]), float(spec["y"])],
            "yaw": float(spec["yaw"]),
        }
        for color, spec in object_specs.items()
    }


def write_rollout_meta(out_dir, instruction, target_color, object_specs, args):
    meta = {
        "instruction": instruction,
        "target_color": target_color,
        "target_body_name": CYLINDER_BODY_BY_COLOR[target_color],
        "all_object_init_poses": object_specs_to_meta(object_specs),
        "args": args,
    }
    with open(out_dir / "rollout_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)


def rollout(
    xml_path, server_url, instruction, unnorm_key, output_dir,
    episode_id=1, max_steps=1000000, use_viewer=True, camera_name="front_view",
    speed=70, settle_seconds_per_action=0.5,  # 0.8 → 0.5
    initial_settle_seconds=0.3,
    delta_scale=1.0, randomize_objects=True, request_timeout=60.0,
    max_delta_xyz=0.005, target_color_arg="auto",
    instruction_template=DEFAULT_INSTRUCTION_TEMPLATE, seed=None,
    object_x_range=DEFAULT_OBJECT_X_RANGE, object_y_range=DEFAULT_OBJECT_Y_RANGE,
    min_object_distance=DEFAULT_MIN_OBJECT_DISTANCE,
    image_quality: int = 50,  # [IMPROVEMENT] 추가
) -> None:
    out_dir = Path(output_dir) / f"episode_{episode_id:06d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    clear_existing_images(out_dir)

    # [IMPROVEMENT] 타이밍 로거 초기화
    log_dir = Path(output_dir) / "logs"
    timing_logger = TimingLogger(log_dir=log_dir, episode_id=episode_id)

    rng = np.random.default_rng(seed)
    target_color, instruction = resolve_target_color_and_instruction(
        instruction=instruction, target_color_arg=target_color_arg,
        rng=rng, instruction_template=instruction_template,
    )

    if randomize_objects:
        object_specs = sample_object_specs(
            rng=rng, x_range=object_x_range, y_range=object_y_range,
            min_distance=min_object_distance,
        )
    else:
        object_specs = make_default_object_specs()

    env = SyncSimRaccoonEnv(
        xml_path=xml_path, image_size=(256, 256),
        camera_name=camera_name, use_viewer=use_viewer,
    )

    try:
        reset_multicolor_scene(env=env, object_specs=object_specs, target_color=target_color)
        env.lockh()
        env.debug_check_current_ee_reachable()

        if initial_settle_seconds > 0:
            env.settle_steps(seconds=initial_settle_seconds)

        write_rollout_meta(
            out_dir=out_dir, instruction=instruction, target_color=target_color,
            object_specs=object_specs,
            args={
                "xml_path": xml_path, "unnorm_key": unnorm_key,
                "camera_name": camera_name, "speed": speed,
                "settle_seconds_per_action": settle_seconds_per_action,
                "initial_settle_seconds": initial_settle_seconds,
                "delta_scale": delta_scale, "max_delta_xyz": max_delta_xyz,
                "seed": seed, "object_x_range": list(object_x_range),
                "object_y_range": list(object_y_range),
                "min_object_distance": min_object_distance,
            },
        )

        print(
            f"[SCENE] instruction={instruction!r} | target_color={target_color!r} | "
            f"target_xy=({object_specs[target_color]['x']:.3f}, {object_specs[target_color]['y']:.3f})"
        )

        # [IMPROVEMENT] 에피소드 시작 시간 기록
        timing_logger.start_episode()

        obs = env.get_observation()
        step_idx = 0

        while True:
            # [IMPROVEMENT] 추론 시간 측정
            response, inference_time = request_action(
                server_url=server_url, instruction=instruction,
                image_rgb=obs["image"], unnorm_key=unnorm_key, timeout=request_timeout,
                image_quality=image_quality,  # [IMPROVEMENT] 추가
            )
            action = response["action"]

            # [IMPROVEMENT] 스텝 로그 기록
            timing_logger.log_step(step_idx, inference_time, action)

            try:
                exec_info = env.execute_delta_action7(
                    action=action, speed=speed, delta_scale=delta_scale,
                    max_delta_xyz=max_delta_xyz,
                )
                print_success_log(step_idx, exec_info)
                env.settle_steps(seconds=settle_seconds_per_action)
                obs = env.get_observation()
                frame_name = f"frame_{step_idx:06d}.png"
                Image.fromarray(obs["image"]).save(out_dir / frame_name)

            except Exception as exc:
                print_fail_log(step_idx, exc)
                obs = env.get_observation()
                frame_name = f"frame_{step_idx:06d}_skipped.png"
                Image.fromarray(obs["image"]).save(out_dir / frame_name)
                step_idx += 1
                if step_idx >= max_steps:
                    print("[STOP] max_steps reached")
                    break
                continue

            step_idx += 1
            if step_idx >= max_steps:
                print("[STOP] max_steps reached")
                break

    except KeyboardInterrupt:
        print("\n[STOP] interrupted by user")

    finally:
        # [IMPROVEMENT] 에피소드 종료 요약 출력 및 저장
        timing_logger.end_episode(total_steps=step_idx)
        env.close()


def clear_existing_images(out_dir: Path) -> None:
    image_exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    deleted_count = 0
    for file_path in out_dir.iterdir():
        if file_path.is_file() and file_path.suffix.lower() in image_exts:
            file_path.unlink()
            deleted_count += 1
    print(f"[CLEANUP] removed {deleted_count} existing image files from {out_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--xml_path", type=str, default="Raccoon_colored_cylinder.xml")
    parser.add_argument("--server_url", type=str, default=None)
    parser.add_argument("--instruction", type=str, default=None)
    parser.add_argument("--target_color", type=str, default="auto", choices=["auto", "random", *CYLINDER_COLORS])
    parser.add_argument("--instruction_template", type=str, default=DEFAULT_INSTRUCTION_TEMPLATE)
    parser.add_argument("--unnorm_key", type=str, default="raccoon_pick_place")
    parser.add_argument("--output_dir", type=str, default="rollout_outputs")
    parser.add_argument("--episode_id", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=1000000)
    parser.add_argument("--speed", type=int, default=70)
    parser.add_argument("--initial_settle_seconds", type=float, default=0.3)
    parser.add_argument("--delta_scale", type=float, default=1.0)
    parser.add_argument("--max_delta_xyz", type=float, default=0.005)
    parser.add_argument("--request_timeout", type=float, default=60.0)
    parser.add_argument("--use_viewer", action="store_true")
    parser.add_argument("--camera_name", type=str, default="front_view")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--object_x_range", type=float, nargs=2, default=DEFAULT_OBJECT_X_RANGE)
    parser.add_argument("--object_y_range", type=float, nargs=2, default=DEFAULT_OBJECT_Y_RANGE)
    parser.add_argument("--min_object_distance", type=float, default=DEFAULT_MIN_OBJECT_DISTANCE)
    parser.add_argument("--no_randomize_box", action="store_true")
    parser.add_argument("--no_randomize_objects", action="store_true")
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
    parser.add_argument("--image_quality", type=int, default=50, help="JPEG 압축 품질 (1-95, 낮을수록 빠름)")
    parser.add_argument("--settle_seconds_per_action", type=float, default=0.5)  # 0.8 → 0.5
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with maybe_tunnel_context(args) as tunnel:
        server_url = build_server_url(args, tunnel)
        if tunnel is not None:
            print(f"[SSH] {args.local_server_host}:{tunnel.local_bind_port} -> {args.remote_server_host}:{args.remote_server_port}")
        rollout(
            xml_path=args.xml_path, server_url=server_url,
            instruction=args.instruction, unnorm_key=args.unnorm_key,
            output_dir=args.output_dir, episode_id=args.episode_id,
            max_steps=args.max_steps, use_viewer=args.use_viewer,
            camera_name=args.camera_name, speed=args.speed,
            settle_seconds_per_action=args.settle_seconds_per_action,
            initial_settle_seconds=args.initial_settle_seconds,
            delta_scale=args.delta_scale,
            randomize_objects=not (args.no_randomize_box or args.no_randomize_objects),
            request_timeout=args.request_timeout, max_delta_xyz=args.max_delta_xyz,
            target_color_arg=args.target_color,
            instruction_template=args.instruction_template,
            seed=args.seed,
            object_x_range=tuple(args.object_x_range),
            object_y_range=tuple(args.object_y_range),
            min_object_distance=args.min_object_distance,
            image_quality=args.image_quality,
        )


if __name__ == "__main__":
    main()

