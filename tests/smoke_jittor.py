#!/usr/bin/env python3
"""Small Jittor model/operator smoke test; no external data or weights required."""

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))
from plr3d.backend import add_device_argument, configure_device, device_status


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


def model_inputs(seed=20260810):
    """Return a deterministic cloud whose KNN=4 neighbourhood is complete."""
    rng = np.random.RandomState(seed)
    return rng.normal(loc=0.0, scale=0.1, size=(1, 5, 3)).astype(np.float32)


def reduced_model_gradient_targets(inputs, model):
    """Request only differentiable model state, never BatchNorm running buffers."""
    return [inputs, model.feature_nets[0].linear3.weight]


def unexecuted_model_acceptance_report():
    """Return the stable report shape when model acceptance is not requested."""
    return {
        "model_acceptance_executed": False,
        "input_shape": [],
        "output_shape": [],
        "parameter_count": 0,
        "optimizer_update": {"changed": False, "parameter_names": []},
        "checkpoint_roundtrip": False,
        "checkpoint_max_abs_diff": None,
    }


def serialize_smoke_report(device_report, elapsed_seconds, model_report):
    """Serialize the device-independent smoke report schema."""
    report = {
        "device_status": device_report,
        "elapsed_seconds": elapsed_seconds,
        **model_report,
    }
    return json.dumps(report, sort_keys=True)


def run_acl_model_acceptance(model, model_type, jt_module):
    """Exercise one real ACL training update and a model checkpoint round trip."""
    from jittor import optim

    if int(jt_module.flags.use_acl) != 1:
        raise AssertionError("ACL model smoke must start with ACL enabled")
    inputs_np = model_inputs()
    inputs = jt_module.array(inputs_np).float32()
    model.train()
    output = model(inputs)
    if tuple(output.shape) != tuple(inputs_np.shape):
        raise AssertionError(
            "reduced model changed shape {} -> {}".format(
                tuple(inputs_np.shape), tuple(output.shape)
            )
        )
    output_np = output.numpy()
    if not np.isfinite(output_np).all():
        raise AssertionError("reduced model output is not finite")

    loss = (output ** 2).mean()
    gradient_targets = reduced_model_gradient_targets(inputs, model)
    gradients = jt_module.grad(loss, gradient_targets)
    if len(gradients) != len(gradient_targets):
        raise AssertionError("reduced model returned an incomplete gradient list")
    gradient_arrays = [gradient.numpy() for gradient in gradients]
    input_gradient, weight_gradient = gradient_arrays
    if not np.isfinite(input_gradient).all():
        raise AssertionError("reduced model input gradient is not finite")
    if not np.isfinite(weight_gradient).all():
        raise AssertionError("reduced model training-weight gradient is not finite")
    if not np.any(np.abs(input_gradient) > 0.0):
        raise AssertionError("reduced model input gradient is zero")
    if not np.any(np.abs(weight_gradient) > 0.0):
        raise AssertionError("reduced model training-weight gradient is zero")

    before = {
        name: value.numpy().copy() for name, value in model.state_dict().items()
    }
    optimizer = optim.SGD(gradient_targets[1:], lr=1e-3)
    optimizer.zero_grad()
    optimizer.backward(loss)
    optimizer.step()
    jt_module.sync_all()
    after = {name: value.numpy() for name, value in model.state_dict().items()}
    if not all(np.isfinite(value).all() for value in after.values()):
        raise AssertionError("reduced model parameters are not finite after SGD")
    changed = [
        name for name, value in after.items() if not np.array_equal(before[name], value)
    ]
    if not changed:
        raise AssertionError("SGD did not update any reduced model parameter")

    model.eval()
    pre_save = model(inputs)
    pre_save_np = pre_save.numpy()
    with tempfile.TemporaryDirectory() as directory:
        checkpoint_path = str(Path(directory) / "denoise_acl_smoke.pkl")
        jt_module.save({"state_dict": model.state_dict()}, checkpoint_path)
        restored = model_type(
            frame_knn=model.frame_knn,
            num_modules=model.num_modules,
            noise_decay=model.noise_decay,
        )
        restored.load_parameters(jt_module.load(checkpoint_path)["state_dict"])
        jt_module.sync_all()
        restored.eval()
        post_load_np = restored(inputs).numpy()
    checkpoint_max_abs_diff = float(np.max(np.abs(post_load_np - pre_save_np)))
    # With all K=4 neighbours present, the measured ACL float32 drift is <=3.03e-5.
    np.testing.assert_allclose(post_load_np, pre_save_np, rtol=1e-3, atol=5e-5)
    if int(jt_module.flags.use_acl) != 1:
        raise AssertionError("ACL model smoke must leave ACL enabled")
    return {
        "model_acceptance_executed": True,
        "input_shape": list(inputs_np.shape),
        "output_shape": list(output.shape),
        "parameter_count": int(
            sum(value.numel() for value in model.state_dict().values())
        ),
        "optimizer_update": {"changed": True, "parameter_names": changed},
        "checkpoint_roundtrip": True,
        "checkpoint_max_abs_diff": checkpoint_max_abs_diff,
    }


def main(argv=None):
    args = parse_args(argv)
    import jittor as jt

    from plr3d.models import DenoiseNet
    from plr3d.ops.selective_scan import selective_scan, selective_scan_reference

    started = time.perf_counter()
    configure_device(args.device, jt)
    jt.set_global_seed(20260810)
    np.random.seed(20260810)
    model = DenoiseNet(frame_knn=4, num_modules=1)
    state = model.state_dict()
    parameter_count = int(sum(value.numel() for value in state.values()))

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
        model_result = run_acl_model_acceptance(model, DenoiseNet, jt)
    else:
        model_result = unexecuted_model_acceptance_report()
    print(
        serialize_smoke_report(
            device_status(args.device, jt),
            round(time.perf_counter() - started, 6),
            model_result,
        )
    )
    print("JITTOR_SMOKE_OK", len(state), parameter_count)


if __name__ == "__main__":
    main()
