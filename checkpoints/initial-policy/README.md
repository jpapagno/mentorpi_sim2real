# Initial navigation policy · model 900

[Download `model_900.pt`](model_900.pt) · [Iteration 1 trial](../../docs/sim-to-real/trials.md#iteration-1--training-the-initial-policy)

This RSL-RL PPO checkpoint comes from run `2026-11-25_02-02-04` at iteration 900. It uses the original Gaussian actor and a progress reward weight of 1.0. The Raspberry Pi deployment bundle `isaaclab-motion-900` was exported from this checkpoint. The hardware trials exposed a gap between the simulated and physical robot's motion.

From the repository root, after [installation](../../README.md#install), select the original Gaussian agent configuration for playback:

```sh
uv run --extra video isaaclab play --rl_library rsl_rl --task MentorpiIsaaclab-Balance-Mentorpi --agent rsl_rl_deployment1_cfg_entry_point --checkpoint checkpoints/initial-policy/model_900.pt --num_envs 1 --video viz:newton --video_length 1200
```

Playback uses the current simulation, including its planar velocity clamp. This is an Isaac Lab training checkpoint. For hardware export and operation, see the [deployment guide](../../deployment/README.md).
