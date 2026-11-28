# Progress reward policy · model 2150

[Download `model_2150.pt`](model_2150.pt) · [Iteration 3 trial](../../docs/sim-to-real/trials.md#iteration-3--adjusting-the-progress-reward)

This RSL-RL PPO checkpoint comes from run `2026-11-25_23-58-06` at iteration 2150. It uses a squashed Gaussian actor with a ±0.6 m/s limit per axis. This run doubled the progress reward weight to 2.0 and preceded the directional obstacle-clearance penalty. At the checkpoint, the 100-iteration trailing averages were 82.1% goal success and 1.84 mean episode reward.

From the repository root, after [installation](../../README.md#install), play it in the current simulation:

```sh
uv run --extra video isaaclab play --rl_library rsl_rl --task MentorpiIsaaclab-Balance-Mentorpi --checkpoint checkpoints/progress-reward/model_2150.pt --num_envs 1 --video viz:newton --video_length 1200
```
