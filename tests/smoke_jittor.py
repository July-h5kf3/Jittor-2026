#!/usr/bin/env python3
"""Small Jittor model/operator smoke test; no external data or weights required."""

import argparse
import sys
from pathlib import Path

import numpy as np

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))
from plr3d.backend import add_device_argument, configure_device


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    add_device_argument(parser)
    return parser.parse_args(argv)


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


def scan_validation_plan(device):
    if device == "cpu":
        return (("cpu", False),)
    if device in ("cuda", "acl"):
        return (("cpu", False), (device, True))
    raise ValueError("unsupported smoke device: {}".format(device))


def run_scan(arrays, device, custom, jt_module, custom_scan, reference_scan):
    configure_device(device, jt_module)
    values = [jt_module.array(array).float32() for array in arrays]
    output = custom_scan(*values) if custom else reference_scan(*values)
    gradients = jt_module.grad((output ** 2).sum(), values)
    return output.numpy(), [gradient.numpy() for gradient in gradients]


def main(argv=None):
    args = parse_args(argv)
    import jittor as jt

    from plr3d.models import DenoiseNet
    from plr3d.ops.selective_scan import selective_scan, selective_scan_reference

    configure_device(args.device, jt)
    model = DenoiseNet()
    state = model.state_dict()
    assert len(state) == 1280, len(state)
    assert sum(value.numel() for value in state.values()) == 16278412

    arrays = scan_inputs()
    plan = scan_validation_plan(args.device)
    reference_device, reference_custom = plan[0]
    reference, reference_grads = run_scan(
        arrays,
        reference_device,
        reference_custom,
        jt,
        selective_scan,
        selective_scan_reference,
    )
    assert len(reference_grads) == 8, "expected 8 reference gradients"
    for candidate_device, candidate_custom in plan[1:]:
        candidate, candidate_grads = run_scan(
            arrays,
            candidate_device,
            candidate_custom,
            jt,
            selective_scan,
            selective_scan_reference,
        )
        assert len(candidate_grads) == 8, "expected 8 candidate gradients"
        np.testing.assert_allclose(candidate, reference, rtol=2e-5, atol=2e-6)
        for candidate_grad, reference_grad in zip(candidate_grads, reference_grads):
            np.testing.assert_allclose(
                candidate_grad, reference_grad, rtol=2e-4, atol=2e-5
            )
    if args.device == "acl":
        # In the pinned Jittor source, compiler.py aliases use_acl to use_cuda.
        # use_cuda is therefore an ACL internal device flag, not CUDA routing.
        assert bool(jt.flags.use_acl), "ACL smoke must leave ACL enabled"
    print("JITTOR_SMOKE_OK", len(state), sum(value.numel() for value in state.values()))


if __name__ == "__main__":
    main()
