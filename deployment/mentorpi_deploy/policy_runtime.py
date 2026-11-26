"""Load, check and run a TorchScript policy through its task adapter."""

import json
import time

import numpy as np
import torch

from . import isaac_motion
from .command_guard import BODY_LIMITS

ADAPTERS = {isaac_motion.ARCHITECTURE: isaac_motion}


class PolicyRuntime:
    def __init__(self, model_dir):
        contract = json.loads((model_dir / "contract.json").read_text())
        architecture = contract.get("architecture")
        self.adapter = ADAPTERS.get(architecture)
        if self.adapter is None:
            raise ValueError(f"Unsupported deployment architecture: {architecture}")
        expected_contract = (
            self.adapter.BOUNDED_CONTRACT if "action_limit_mps" in contract else self.adapter.CONTRACT
        )
        if any(contract.get(key) != value for key, value in expected_contract.items()):
            raise ValueError(f"Incompatible deployment contract: {architecture}")
        self.action_limit = expected_contract.get("action_limit_mps", 1.0)
        self.ray_height_delta = expected_contract.get("ray_height_delta_m", self.adapter.RAY_HEIGHT_DELTA_M)
        self.period = contract["dt"]
        self.input_size = contract["policy_shape"][1]
        self.action_size = contract["action_shape"][1]
        self.actor = torch.jit.load(str(model_dir / "policy.ts"), map_location="cpu").eval()
        self.parity_path = model_dir / "parity_cases.npz"

    def verify(self):
        with np.load(self.parity_path, allow_pickle=False) as cases, torch.inference_mode():
            inputs = cases["policy"]
            expected = cases["action"]
            if (
                inputs.ndim != 2
                or not len(inputs)
                or inputs.shape[1] != self.input_size
                or expected.shape != (len(inputs), self.action_size)
                or not np.isfinite(inputs).all()
                or not np.isfinite(expected).all()
            ):
                raise ValueError("Invalid deployment parity fixtures")
            actual = self.actor(torch.from_numpy(inputs)).numpy()
            np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6, equal_nan=False)

        # Match the singleton-batch NumPy strides used by the live adapter.
        sample = inputs[0][None, :]
        with torch.inference_mode():
            tensor = torch.from_numpy(sample)
            for _ in range(50):
                self.actor(tensor)
        times = [self.predict(sample)[1] for _ in range(10)]
        return {"parity_cases": len(inputs), "inference_ms": {"median": float(np.median(times)), "max": max(times)}}

    def predict(self, state):
        if state.shape != (1, self.input_size) or not np.isfinite(state).all():
            raise RuntimeError("Invalid policy observation")
        started = time.perf_counter()
        with torch.inference_mode():
            output = self.actor(torch.from_numpy(state)).numpy()
        elapsed_ms = (time.perf_counter() - started) * 1000
        if elapsed_ms > 100:
            raise RuntimeError("Inference exceeded 100 ms; no new motor command sent")
        if output.shape != (1, self.action_size) or not np.isfinite(output).all():
            raise RuntimeError("Invalid policy action")
        return output[0], elapsed_ms

    def step(self, base, start, body_velocity, goal, scan_points):
        state = self.adapter.observation(base, start, body_velocity, goal, scan_points, self.ray_height_delta)
        raw_action, inference_ms = self.predict(state)
        action = np.clip(raw_action, -self.action_limit, self.action_limit)
        command = self.adapter.body_command(action, base[2] - start[2])
        limits = np.asarray(BODY_LIMITS)
        scale = max(1.0, np.max(np.abs(command) / limits))
        return state[0], raw_action, action, command / scale, inference_ms
