import argparse
from deepvision.utils.io import read_video_frames, write_video_frames, get_fps
from deepvision.utils.seed import set_seed
# Model Wrappers
from deepvision.models.rife_wrapper import RIFEInterpolator
from deepvision.models.adaptive_interpolator import AdaptiveInterpolator 
from deepvision.pipelines.interpolate import interpolate_video_frames
from deepvision.models.optical_flow_wrapper import OpticalFlowInterpolator 

def main():
    p = argparse.ArgumentParser(description='Video Frame Interpolation using RIFE, Optical Flow, or the Adaptive Switcher.')
    p.add_argument('--input', required=True, help='Path to input video (e.g., incomplete video).')
    p.add_argument('--output', required=True, help='Path to output MP4.')
    p.add_argument('--num-intermediate', type=int, default=1, help='Number of in-between frames to generate per pair.')
    
    # Deep Learning Model arguments
    p.add_argument('--impl', type=str, default=None, help='RIFE impl as module:Class for dynamic loading (e.g., some_repo.rife:RIFE).')
    p.add_argument('--ckpt', type=str, default=None, help='Optional checkpoint path for the chosen impl.')
    
    # Innovation arguments (Flow / Adaptive Switch)
    p.add_argument('--adaptive', action='store_true', help='Use the Adaptive Interpolator (Flow vs. RIFE).') 
    p.add_argument('--use-flow', action='store_true', help='Use only the classic Optical Flow Interpolator.') 
    
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()

    set_seed(args.seed)

    fps = get_fps(args.input)
    frames = read_video_frames(args.input)

    # ------------------ Interpolator Selection Logic ------------------
    if args.use_flow:
        print("Using pure Classic Optical Flow Interpolator (your method)...")
        # Use user's pure Optical Flow method
        interpolator = OpticalFlowInterpolator().load_pretrained()
    elif args.adaptive:
        print("Using the Adaptive Switching System (Flow / RIFE)...")
        # Use the smart adaptive system
        interpolator = AdaptiveInterpolator(
            rife_impl=args.impl, 
            rife_ckpt=args.ckpt, 
            motion_threshold=5.0 # Customizable motion threshold
        )
    else:
        # Default mode: Use RIFE (or Dummy Interpolator if no implementation is specified)
        print("Using RIFE (Deep Learning) or Dummy Interpolator...")
        interpolator = RIFEInterpolator().load_pretrained(impl=args.impl, ckpt_path=args.ckpt)
    # ---------------------------------------------------------------------

    out_frames = interpolate_video_frames(frames, interpolator, num_intermediate=args.num_intermediate)

    # Calculate new FPS: FPS * (num_intermediate + 1)
    new_fps = fps * (args.num_intermediate + 1)
    write_video_frames(args.output, out_frames, new_fps)
    print(f"Saved: {args.output} at {new_fps:.2f} FPS")

if __name__ == '__main__':
    main()