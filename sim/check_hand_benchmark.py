"""Sanity check: Gymnasium-Robotics Shadow Hand in-hand rotation benchmark on this machine.

Verifies:
1. HandManipulateBlockRotateZ-v1 (rotate block about palm-normal axis to target orientation)
   installs, resets and steps on Windows + Python 3.12.
2. Observation structure (achieved_goal / desired_goal = quaternions).
3. Offscreen RGB rendering (for vision-input experiments).
"""
import gymnasium as gym
import gymnasium_robotics  # noqa: F401  (registers the environments)


def main():
    ids = sorted(k for k in gym.registry.keys() if "HandManipulate" in k)
    print("available HandManipulate envs:")
    for env_id in ids:
        print("  ", env_id)

    env = gym.make("HandManipulateBlockRotateZ-v1", render_mode="rgb_array")
    obs, info = env.reset(seed=0)
    print("\nobs keys:", list(obs.keys()))
    for key, value in obs.items():
        print(f"  {key}: shape={getattr(value, 'shape', None)}")

    total_r = 0.0
    for step in range(10):
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        total_r += reward
    print(f"\n10 random steps ok: reward_sum={total_r:.2f}, is_success={info.get('is_success')}")

    frame = env.render()
    print("render frame:", None if frame is None else frame.shape)
    env.close()
    print("BENCHMARK_CHECK_OK")


if __name__ == "__main__":
    main()
