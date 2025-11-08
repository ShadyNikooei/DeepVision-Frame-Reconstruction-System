"""
CLI for video interpolation.
Works out-of-the-box with the dummy interpolator; swap in a real RIFE implementation via --impl and --ckpt.
"""
import argparse
from deepvision.utils.io import read_video_frames, write_video_frames, get_fps
from deepvision.utils.seed import set_seed
from deepvision.models.rife_wrapper import RIFEInterpolator
from deepvision.pipelines.interpolate import interpolate_video_frames

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', required=True, help='Path to input video')
    p.add_argument('--output', required=True, help='Path to output MP4')
    p.add_argument('--num-intermediate', type=int, default=1, help='In-betweens per pair (e.g., 1 -> doubles FPS)')
    p.add_argument('--impl', type=str, default=None, help='Real RIFE impl as module:Class (e.g., some_repo.rife:RIFE)')
    p.add_argument('--ckpt', type=str, default=None, help='Optional checkpoint path for the chosen impl')
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()

    set_seed(args.seed)

    fps = get_fps(args.input)
    frames = read_video_frames(args.input)

    interpolator = RIFEInterpolator().load_pretrained(impl=args.impl, ckpt_path=args.ckpt)

    out_frames = interpolate_video_frames(frames, interpolator, num_intermediate=args.num_intermediate)

    new_fps = fps * (args.num_intermediate + 1)
    write_video_frames(args.output, out_frames, new_fps)
    print(f"Saved: {args.output} @ {new_fps:.2f} FPS")

if __name__ == '__main__':
    main()
