import argparse
from PIL import Image
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--episode_dir", type=str, required=True)
parser.add_argument("--output", type=str, default="output.gif")
args = parser.parse_args()

episode_dir = Path(args.episode_dir)
frames = sorted(episode_dir.glob("frame_*.png"))
print(f"프레임 수: {len(frames)}")

images = [Image.open(f) for f in frames]
images[0].save(args.output, save_all=True, append_images=images[1:], duration=100, loop=0)
print(f"저장완료: {args.output}")