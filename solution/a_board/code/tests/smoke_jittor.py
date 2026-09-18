#!/usr/bin/env python3
"""Small Jittor model/operator smoke test; no external data or weights required."""

import sys
from pathlib import Path

import numpy as np
import jittor as jt

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))
from plr3d.models import DenoiseNet
from plr3d.ops.selective_scan import selective_scan, selective_scan_reference


def scan_inputs(seed=7):
    rng = np.random.RandomState(seed)
    shapes = [
        (2, 3, 7),
        (2, 3, 7),
        (3, 4),
        (2, 4, 7),
        (2, 4, 7),
        (3,),
        (2, 3, 7),
        (3,),
    ]
    arrays = [rng.randn(*shape).astype(np.float32) * 0.2 for shape in shapes]
    arrays[2] = -np.exp(arrays[2]).astype(np.float32)
    return arrays


def run_scan(arrays, use_cuda, custom):
    jt.flags.use_cuda = int(use_cuda)
    values = [jt.array(array).float32() for array in arrays]
    output = selective_scan(*values) if custom else selective_scan_reference(*values)
    gradients = jt.grad((output ** 2).sum(), values)
    return output.numpy(), [gradient.numpy() for gradient in gradients]


def main():
    jt.flags.use_cuda = 0
    model = DenoiseNet()
    state = model.state_dict()
    assert len(state) == 1280, len(state)
    assert sum(value.numel() for value in state.values()) == 16278412

    arrays = scan_inputs()
    reference, reference_grads = run_scan(arrays, False, False)
    if jt.compiler.has_cuda:
        candidate, candidate_grads = run_scan(arrays, True, True)
        np.testing.assert_allclose(candidate, reference, rtol=2e-5, atol=2e-6)
        for candidate_grad, reference_grad in zip(candidate_grads, reference_grads):
            np.testing.assert_allclose(
                candidate_grad, reference_grad, rtol=2e-4, atol=2e-5
            )
    print("JITTOR_SMOKE_OK", len(state), sum(value.numel() for value in state.values()))


if __name__ == "__main__":
    main()
