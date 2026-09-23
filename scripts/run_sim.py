"""CLI entry: run the InHand-GPT decision loop in simulation.

Examples:
  python scripts/run_sim.py --policy baseline --seed 0 --episodes 1 \
      --max-cycles 25 --video runs/seed0.mp4 --export runs/seed0.json
  python scripts/run_sim.py --policy gpt --dry-run --seed 0 --max-cycles 1
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import gymnasium as gym
import gymnasium_robotics  # noqa: F401

from inhand_gpt.loop import run_episode
from inhand_gpt.policy import GPTPolicy, PolicyError, RuleBaselinePolicy
from inhand_gpt.record import Recorder

ENV_ID = "HandManipulateBlockRotateZ-v1"


class VideoWriter:
    """MP4 via imageio-ffmpeg; falls back to GIF with a warning."""

    def __init__(self, path: str, fps: int = 25):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.writer = None
        self.fps = fps
        self._init()

    def _init(self):
        import imageio.v2 as imageio
        try:
            self.writer = imageio.get_writer(
                str(self.path), fps=self.fps, codec="libx264",
                macro_block_size=8, ffmpeg_log_level="error")
        except Exception as e:
            gif = self.path.with_suffix(".gif")
            print(f"warning: mp4 encoder unavailable ({e}); writing GIF to {gif}")
            self.path = gif
            self.writer = imageio.get_writer(str(gif), duration=1.0 / self.fps, loop=0)

    def append(self, frame):
        self.writer.append_data(frame)

    def close(self):
        if self.writer is not None:
            self.writer.close()


def build_env(seed: int, need_video: bool):
    return gym.make(
        ENV_ID,
        render_mode="rgb_array" if need_video else None,
        max_episode_steps=2000,  # 25 cycles x ~30 steps must not hit the TimeLimit
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="InHand-GPT sim loop")
    ap.add_argument("--policy", choices=["baseline", "gpt"], default="baseline")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--episodes", type=int, default=1)
    ap.add_argument("--max-cycles", type=int, default=25)
    ap.add_argument("--deadband", type=float, default=5.0,
                    help="deg, baseline finish deadband (success threshold is 5.73deg)")
    ap.add_argument("--video", type=str, default=None, help="mp4 path (episode 1 only)")
    ap.add_argument("--export", type=str, default=None, help="json export path")
    ap.add_argument("--dry-run", action="store_true",
                    help="gpt policy: print the full prompt, send nothing")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    try:
        if args.policy == "baseline":
            policy = RuleBaselinePolicy(deadband_deg=args.deadband)
        else:
            policy = GPTPolicy(dry_run=args.dry_run)
    except PolicyError as e:
        print(f"error: {e}")
        return 2

    results = []
    for ep in range(args.episodes):
        seed = args.seed + ep
        suffix = f"seed{seed}" + (f"_ep{ep}" if args.episodes > 1 else "")
        video_path = args.video if (args.video and ep == 0) else None
        env = build_env(seed, need_video=video_path is not None)

        vw = VideoWriter(video_path) if video_path else None

        def frame_fn():
            frame = env.render()
            if frame is not None:
                vw.append(frame)

        recorder = Recorder(policy_name=policy.name, seed=seed, max_cycles=args.max_cycles)
        try:
            summary = run_episode(
                env, policy, max_cycles=args.max_cycles, seed=seed,
                recorder=recorder,
                frame_fn=frame_fn if vw else None,
                verbose=not args.quiet,
            )
        finally:
            if vw:
                vw.close()
                if not args.quiet:
                    print(f"video written: {vw.path}")
            env.close()

        if args.export:
            exp = Path(args.export)
            if args.episodes > 1:
                exp = exp.with_name(f"{exp.stem}_{suffix}{exp.suffix}")
            out = recorder.export(exp)
            if not args.quiet:
                print(f"json exported: {out}")
        results.append((seed, summary))

    print("\n=== summary ===")
    n_success = 0
    for seed, s in results:
        n_success += s["outcome"] == "success"
        print(f"seed {seed}: {s['outcome']:20s} final_err={s['final_err_deg']:+.2f}deg "
              f"cycles={s['n_cycles']} wall={s['wall_time_s']:.1f}s")
    print(f"success rate: {n_success}/{len(results)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
