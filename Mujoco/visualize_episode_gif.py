
import json

from PIL import Image

from pathlib import Path

import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--episode_idx", type=int, default=0, help="몇 번째 에피소드 볼지")

parser.add_argument("--root", type=str, default="/data/Raccoonbot_Openvla/Mujoco/raccoon_grasp_extended")

parser.add_argument("--output", type=str, default="/data/Raccoonbot_Openvla/Mujoco/episode_animation.gif")

args = parser.parse_args()

root = Path(args.root)

episodes = sorted([p for p in root.iterdir() if p.is_dir() and p.name.startswith("episode_")])

ep = episodes[args.episode_idx]

frames = sorted(ep.glob("frame_*.png"))

with open(ep / "meta.json", "r") as f:

    meta = json.load(f)

instruction = meta.get("instruction", "N/A")

target = meta.get("target_color", "N/A")

success = meta.get("success", "N/A")

print(f"Episode: {ep.name}")

print(f"Instruction: {instruction}")

print(f"Target: {target}, Success: {success}")

images = [Image.open(f) for f in frames]

images[0].save(

    args.output,

    save_all=True,

    append_images=images[1:],

    duration=100,

    loop=0

)

print(f"GIF 저장완료: {args.output}")

