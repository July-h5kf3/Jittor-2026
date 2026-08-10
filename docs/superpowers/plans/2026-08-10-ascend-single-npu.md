# Ascend Single-NPU Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the PLR-003 baseline installable and functionally verifiable on one Ascend 910 using Jittor ACL at commit `06f5d3d271555682c95aa3505518f47eeab2bd9c`.

**Architecture:** Add a shared `cpu/cuda/acl` backend boundary, preserve the CUDA implementation, and route ACL selective scan through the differentiable Jittor reference path for the first correctness milestone. Prepare the remote environment entirely from locally downloaded sources and ARM64 wheels, then validate ACL primitives, scan gradients, model forward/backward, optimizer update, and checkpoint reload.

**Tech Stack:** Python 3.11, Jittor ACL, CANN 9.1, Ascend 910, unittest, Bash, Docker, SSH, rsync

---

## Scope

This plan implements Milestone A only. Selective-scan/KNN performance kernels and 2/4/8-card HCCL are separate follow-up plans after single-NPU correctness passes.

## File Map

- `plr3d/backend.py`: shared backend configuration and status.
- `requirements-ascend.txt`: offline Python dependencies.
- `scripts/ascend_env.sh`: pinned remote activation.
- `scripts/setup_ascend_env.sh`: offline remote installer.
- `scripts/prepare_ascend_bundle.sh`: local Jittor/wheel preparation.
- `scripts/sync_ascend.sh`: local-to-remote synchronization.
- `tests/test_backend.py`: local backend tests using a fake Jittor object.
- `tests/test_ascend_environment_contract.py`: environment-script contract tests.
- `tests/test_device_wiring.py`: entry-point wiring tests.
- `tests/test_acl_primitives.py`: real single-NPU ACL compatibility matrix.
- `main.py`: device-aware doctor.
- `infer.py`, `generate_rot.py`, `train.py`, `train_rot.py`, `run_plr003.py`: shared device setup.
- `plr3d/ops/selective_scan.py`: ACL reference routing.
- `tests/smoke_jittor.py`: CPU/CUDA/ACL numerical and reduced-model smoke.
- `tools/build_reproduce_config.py`: parameterized generated device.
- `README.md`: accepted Ascend workflow and limitations.

## Task 1: Reproducible Offline Ascend Environment

**Files:**
- Create: `requirements-ascend.txt`
- Create: `scripts/ascend_env.sh`
- Create: `scripts/setup_ascend_env.sh`
- Create: `scripts/prepare_ascend_bundle.sh`
- Create: `scripts/sync_ascend.sh`
- Create: `tests/test_ascend_environment_contract.py`

- [ ] **Step 1: Write the failing environment tests**

Create `tests/test_ascend_environment_contract.py`:

```python
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
SHA = "06f5d3d271555682c95aa3505518f47eeab2bd9c"

class AscendEnvironmentContractTests(unittest.TestCase):
    def test_activation_is_pinned(self):
        text = (ROOT / "scripts/ascend_env.sh").read_text(encoding="utf-8")
        self.assertIn("/usr/local/Ascend/cann-9.1.0-beta.1", text)
        self.assertIn(SHA, text)
        self.assertIn("/data/ldc/cache/jittor-track2-ascend", text)

    def test_remote_setup_is_offline(self):
        text = (ROOT / "scripts/setup_ascend_env.sh").read_text(encoding="utf-8")
        self.assertIn("--no-index", text)
        self.assertIn("--no-build-isolation", text)
        self.assertIn(f"jittor-{SHA}", text)
        self.assertNotIn("github.com", text)

    def test_sync_excludes_generated_state(self):
        text = (ROOT / "scripts/sync_ascend.sh").read_text(encoding="utf-8")
        for value in (".git", "__pycache__", "*.pyc", ".venv", "train_logs"):
            self.assertIn(value, text)
        self.assertIn("zhiyuan-huawei:/data/ldc/Track2-new/", text)
```

- [ ] **Step 2: Verify RED**

Run `python3 -m unittest tests.test_ascend_environment_contract`.

Expected: failures report missing scripts.

- [ ] **Step 3: Create the dependency and activation files**

`requirements-ascend.txt`:

```text
numpy==1.24.4
tqdm==4.67.1
pillow==11.3.0
astunparse==1.6.3
six==1.17.0
```

`scripts/ascend_env.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail
export ASCEND_TOOLKIT_HOME=/usr/local/Ascend/cann-9.1.0-beta.1
source "$ASCEND_TOOLKIT_HOME/set_env.sh"
export tikcc_path="$ASCEND_TOOLKIT_HOME/bin/ccec"
export JITTOR_HOME=/data/ldc/cache/jittor-track2-ascend
export NKAI_JITTOR_COMMIT=06f5d3d271555682c95aa3505518f47eeab2bd9c
export PATH="/data/ldc/envs/track2-ascend/bin:$PATH"
```

- [ ] **Step 4: Create the offline installer**

`scripts/setup_ascend_env.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT=/data/ldc/Track2-new
ENV_ROOT=/data/ldc/envs/track2-ascend
WHEEL_ROOT=/data/ldc/packages/track2-ascend
JITTOR_SHA=06f5d3d271555682c95aa3505518f47eeab2bd9c
JITTOR_ROOT="/data/ldc/vendor/jittor-$JITTOR_SHA"
test -f "$PROJECT_ROOT/requirements-ascend.txt"
test -f "$JITTOR_ROOT/setup.py"
test -d "$WHEEL_ROOT"
mkdir -p /data/ldc/envs /data/ldc/cache/jittor-track2-ascend
if [ ! -x "$ENV_ROOT/bin/python" ]; then
    python -m venv --system-site-packages "$ENV_ROOT"
fi
"$ENV_ROOT/bin/python" -m pip install --no-index --find-links "$WHEEL_ROOT" -r "$PROJECT_ROOT/requirements-ascend.txt"
"$ENV_ROOT/bin/python" -m pip install --no-index --no-build-isolation --no-deps -e "$JITTOR_ROOT"
```

- [ ] **Step 5: Create local bundle and sync scripts**

Create `scripts/prepare_ascend_bundle.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail
BUNDLE_ROOT=${1:?usage: prepare_ascend_bundle.sh BUNDLE_ROOT}
JITTOR_SHA=06f5d3d271555682c95aa3505518f47eeab2bd9c
JITTOR_ROOT="$BUNDLE_ROOT/jittor-$JITTOR_SHA"
WHEEL_ROOT="$BUNDLE_ROOT/wheels"
test ! -e "$JITTOR_ROOT"
mkdir -p "$BUNDLE_ROOT" "$WHEEL_ROOT"
git clone --filter=blob:none https://github.com/Jittor/jittor.git "$JITTOR_ROOT"
git -C "$JITTOR_ROOT" checkout "$JITTOR_SHA"
test "$(git -C "$JITTOR_ROOT" rev-parse HEAD)" = "$JITTOR_SHA"
python3 -m pip download --only-binary=:all: --platform manylinux2014_aarch64 --python-version 311 --implementation cp --dest "$WHEEL_ROOT" -r requirements-ascend.txt
```

Create `scripts/sync_ascend.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail
ROOT=$(git rev-parse --show-toplevel)
test "$(git -C "$ROOT" branch --show-current)" = new
rsync -az --delete --exclude '.git' --exclude '__pycache__' --exclude '*.pyc' --exclude '.venv' --exclude 'train_logs' --exclude 'checkpoints' --exclude 'predictions' "$ROOT/" zhiyuan-huawei:/data/ldc/Track2-new/
```

- [ ] **Step 6: Verify GREEN and commit**

Run `python3 -m unittest tests.test_ascend_environment_contract`.

Expected: 3 tests pass.

Commit:

```bash
git add requirements-ascend.txt scripts tests/test_ascend_environment_contract.py
git commit -m "build: add offline Ascend environment workflow"
```

## Task 2: Shared Device Backend

**Files:**
- Create: `plr3d/backend.py`
- Create: `tests/test_backend.py`

- [ ] **Step 1: Write failing backend tests**

Create `tests/test_backend.py`:

```python
import unittest
from plr3d.backend import BackendError, configure_device, device_status

class Flags:
    use_cuda = 0
    use_acl = 0

class Compiler:
    has_cuda = True
    has_acl = True
    nvcc_path = "/cuda/nvcc"
    tikcc_path = "/ascend/ccec"

class FakeJittor:
    flags = Flags()
    compiler = Compiler()
    rank = 0
    world_size = 1
    in_mpi = False

class BackendTests(unittest.TestCase):
    def setUp(self):
        FakeJittor.flags.use_cuda = 0
        FakeJittor.flags.use_acl = 0

    def test_acl_enables_only_acl(self):
        configure_device("acl", FakeJittor)
        self.assertEqual(FakeJittor.flags.use_acl, 1)
        self.assertEqual(FakeJittor.flags.use_cuda, 0)

    def test_cuda_enables_only_cuda(self):
        configure_device("cuda", FakeJittor)
        self.assertEqual(FakeJittor.flags.use_cuda, 1)
        self.assertEqual(FakeJittor.flags.use_acl, 0)

    def test_cpu_disables_accelerators(self):
        configure_device("cpu", FakeJittor)
        self.assertEqual(FakeJittor.flags.use_cuda, 0)
        self.assertEqual(FakeJittor.flags.use_acl, 0)

    def test_missing_acl_is_hard_error(self):
        FakeJittor.compiler.has_acl = False
        try:
            with self.assertRaisesRegex(BackendError, "ACL backend is unavailable"):
                configure_device("acl", FakeJittor)
        finally:
            FakeJittor.compiler.has_acl = True

    def test_status_reports_environment_and_rank(self):
        status = device_status("acl", FakeJittor, {
            "NKAI_JITTOR_COMMIT": "abc",
            "ASCEND_TOOLKIT_HOME": "/ascend",
            "JITTOR_HOME": "/cache",
        })
        self.assertEqual(status["jittor_commit"], "abc")
        self.assertEqual(status["ascend_toolkit_home"], "/ascend")
        self.assertEqual(status["rank"], 0)
        self.assertEqual(status["world_size"], 1)
```

- [ ] **Step 2: Verify RED**

Run `python3 -m unittest tests.test_backend`.

Expected: `ModuleNotFoundError: plr3d.backend`.

- [ ] **Step 3: Implement `plr3d/backend.py`**

```python
from __future__ import annotations
import argparse
import os

DEVICES = ("cpu", "cuda", "acl")

class BackendError(RuntimeError):
    pass

def add_device_argument(parser: argparse.ArgumentParser, default: str = "cuda") -> None:
    parser.add_argument("--device", choices=DEVICES, default=default)

def configure_device(device: str, jt_module=None) -> None:
    if device not in DEVICES:
        raise BackendError(f"unsupported device: {device}")
    if jt_module is None:
        import jittor as jt_module
    if device == "cuda" and not bool(getattr(jt_module.compiler, "has_cuda", False)):
        raise BackendError("CUDA backend is unavailable")
    if device == "acl" and not bool(getattr(jt_module.compiler, "has_acl", False)):
        raise BackendError("ACL backend is unavailable")
    if hasattr(jt_module.flags, "use_acl"):
        jt_module.flags.use_acl = int(device == "acl")
    elif device == "acl":
        raise BackendError("Jittor does not expose the ACL device flag")
    jt_module.flags.use_cuda = int(device == "cuda")

def device_status(device: str, jt_module=None, environ=None):
    if jt_module is None:
        import jittor as jt_module
    env = os.environ if environ is None else environ
    return {
        "device": device,
        "jittor_commit": env.get("NKAI_JITTOR_COMMIT", "unknown"),
        "has_cuda": bool(getattr(jt_module.compiler, "has_cuda", False)),
        "has_acl": bool(getattr(jt_module.compiler, "has_acl", False)),
        "use_cuda": bool(getattr(jt_module.flags, "use_cuda", 0)),
        "use_acl": bool(getattr(jt_module.flags, "use_acl", 0)),
        "nvcc_path": str(getattr(jt_module.compiler, "nvcc_path", "")),
        "tikcc_path": str(getattr(jt_module.compiler, "tikcc_path", "")),
        "ascend_toolkit_home": env.get("ASCEND_TOOLKIT_HOME", ""),
        "jittor_home": env.get("JITTOR_HOME", ""),
        "in_mpi": bool(getattr(jt_module, "in_mpi", False)),
        "rank": int(getattr(jt_module, "rank", 0) or 0),
        "world_size": int(getattr(jt_module, "world_size", 1) or 1),
    }
```

- [ ] **Step 4: Verify GREEN and commit**

Run `python3 -m unittest tests.test_backend`; expect all backend tests to pass.

Commit:

```bash
git add plr3d/backend.py tests/test_backend.py
git commit -m "feat: add CPU CUDA and ACL backend selection"
```

## Task 3: Wire All Device-Aware Entry Points

**Files:**
- Modify: `infer.py`
- Modify: `generate_rot.py`
- Modify: `train.py`
- Modify: `train_rot.py`
- Modify: `run_plr003.py`
- Modify: `tools/build_reproduce_config.py`
- Create: `tests/test_device_wiring.py`

- [ ] **Step 1: Write failing wiring tests**

Create `tests/test_device_wiring.py`:

```python
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
ENTRYPOINTS = ("infer.py", "generate_rot.py", "train.py", "train_rot.py", "run_plr003.py")

class DeviceWiringTests(unittest.TestCase):
    def test_entrypoints_use_shared_backend(self):
        for relative in ENTRYPOINTS:
            text = (ROOT / relative).read_text(encoding="utf-8")
            self.assertIn("add_device_argument", text, relative)
            self.assertIn("configure_device(args.device", text, relative)
            self.assertNotIn('choices=("cuda", "cpu")', text, relative)
            self.assertNotIn("jt.flags.use_cuda =", text, relative)

    def test_config_builder_is_parameterized(self):
        text = (ROOT / "tools/build_reproduce_config.py").read_text(encoding="utf-8")
        self.assertIn('parser.add_argument("--device", choices=("cpu", "cuda", "acl"), default="cuda")', text)
        self.assertNotIn('"--device", "cuda"', text)
```

- [ ] **Step 2: Verify RED**

Run `python3 -m unittest tests.test_device_wiring`.

Expected: failures identify all direct device assignments and hardcoded CUDA config values.

- [ ] **Step 3: Wire the shared backend**

In every device-aware entry point import:

```python
from plr3d.backend import add_device_argument, configure_device
```

Replace the parser declaration with `add_device_argument(parser)` and replace the direct flag assignment with `configure_device(args.device, jt)`.

- [ ] **Step 4: Parameterize config generation**

Add `parse_args()` to `tools/build_reproduce_config.py` with the exact device choices above. Pass `args.device` through `main`, `train`, and `infer`, and use it in every generated `--device` argument. Running the generator without arguments must continue to produce the checked-in CUDA configuration.

- [ ] **Step 5: Verify GREEN and commit**

Run:

```bash
python3 -m unittest tests.test_device_wiring tests.test_source_policy
python3 tools/build_reproduce_config.py
git diff --exit-code configs/reproduce_full.json
```

Expected: tests pass and default regeneration does not alter the CUDA config.

Commit:

```bash
git add infer.py generate_rot.py train.py train_rot.py run_plr003.py tools/build_reproduce_config.py tests/test_device_wiring.py
git commit -m "feat: wire ACL device through training and inference"
```

## Task 4: Device-Aware Doctor

**Files:**
- Modify: `main.py`
- Modify: `tests/test_backend.py`

- [ ] **Step 1: Add a failing parser test**

Add:

```python
from main import parse_args

def test_doctor_accepts_acl(self):
    args = parse_args(["doctor", "--device", "acl"])
    self.assertEqual(args.device, "acl")
```

- [ ] **Step 2: Verify RED**

Run `python3 -m unittest tests.test_backend`.

Expected: argparse rejects `--device acl`.

- [ ] **Step 3: Implement device-aware doctor**

In `main.py`:

- import `add_device_argument`, `configure_device`, and `device_status`;
- change `doctor(deep)` to `doctor(device, deep)`;
- call `prepare_conda_cuda_layout()` only when the selected doctor device is CUDA;
- configure the selected device immediately after importing Jittor;
- merge `device_status(device, jt)` into the JSON report;
- add `add_device_argument(doctor_parser)`;
- dispatch deep smoke with `tests/smoke_jittor.py --device <device>`.

- [ ] **Step 4: Verify GREEN and commit**

Run:

```bash
python3 -m unittest tests.test_backend tests.test_source_policy
python3 -m compileall -q main.py plr3d tests
```

Expected: tests and compilation pass.

Commit:

```bash
git add main.py tests/test_backend.py
git commit -m "feat: report and validate ACL environment in doctor"
```

## Task 5: ACL Selective Scan Reference Path

**Files:**
- Modify: `plr3d/ops/selective_scan.py`
- Modify: `tests/smoke_jittor.py`

- [x] **Step 1: Write the failing ACL smoke behavior**

Extend `tests/smoke_jittor.py` with `--device {cpu,cuda,acl}`. Configure the backend before creating tensors. In ACL mode, run both `selective_scan` and `selective_scan_reference`, compare the forward arrays with `rtol=2e-5, atol=2e-6`, compare all eight gradients with `rtol=2e-4, atol=2e-5`, and assert `jt.flags.use_acl == 1`.

- [x] **Step 2: Install the pinned remote environment before testing**

Prepare the bundle locally, transfer its Jittor checkout to `/data/ldc/vendor/jittor-06f5d3d271555682c95aa3505518f47eeab2bd9c`, transfer wheels to `/data/ldc/packages/track2-ascend`, synchronize project code, then run inside the container:

```bash
source /data/ldc/Track2-new/scripts/ascend_env.sh
bash /data/ldc/Track2-new/scripts/setup_ascend_env.sh
python -c 'import jittor as jt; print(jt.__version__, jt.compiler.has_acl, jt.compiler.tikcc_path)'
```

Expected: import succeeds, the pinned source is active, and `has_acl` is true.

- [x] **Step 3: Verify RED remotely**

Run:

```bash
python /data/ldc/Track2-new/tests/smoke_jittor.py --device acl
```

Expected: fail because current selection does not explicitly route ACL to the reference path.

- [x] **Step 4: Implement explicit ACL routing**

At the end of `selective_scan` use:

```python
if bool(getattr(jt.flags, "use_acl", 0)):
    return selective_scan_reference(u, delta, a, b_var, c_var, d_skip, z, delta_bias)
if bool(jt.flags.use_cuda):
    return _SelectiveScanCUDA.apply(
        u.contiguous(), delta.contiguous(), a.contiguous(), b_var.contiguous(),
        c_var.contiguous(), d_skip.contiguous(), z.contiguous(), delta_bias.contiguous(),
    )
return selective_scan_reference(u, delta, a, b_var, c_var, d_skip, z, delta_bias)
```

- [x] **Step 5: Verify GREEN remotely and commit**

Re-synchronize and rerun the ACL smoke. Expected: forward and all gradients pass.

Commit:

```bash
git add plr3d/ops/selective_scan.py tests/smoke_jittor.py
git commit -m "feat: add ACL selective scan reference path"
```

### Task 5 Verification Evidence — 2026-08-10

- The remote Jittor checkout root was exactly `/data/ldc/vendor/jittor-06f5d3d271555682c95aa3505518f47eeab2bd9c`, `HEAD` was the full pinned SHA `06f5d3d271555682c95aa3505518f47eeab2bd9c`, and `git status --porcelain` was empty.
- The activated runtime reported Python 3.11.15 on aarch64, Jittor 1.3.11.0 from the pinned checkout, CANN 9.1 at `/usr/local/Ascend/cann-9.1.0-beta.1`, `tikcc_path=/usr/local/Ascend/cann-9.1.0-beta.1/bin/ccec`, and `has_acl=1`.
- After ACL configuration the real flags were `(use_acl, use_cuda)=(1,1)`. At this pinned Jittor commit, `use_acl` is an alias of the generic `use_cuda` device flag; the dual-flags routing regression test verifies that selective scan still chooses the ACL reference path and never calls the CUDA custom operator.
- `/data/ldc/envs/track2-ascend/bin/python /data/ldc/Track2-new/tests/smoke_jittor.py --device acl` exited 0 and ended with `JITTOR_SMOKE_OK 1280 16278412`. The smoke compared the selective-scan forward output and all eight gradients with the Task 5 tolerances.
- This evidence completes only Task 5. Final doctor, primitive-matrix, reduced-model, optimizer, and checkpoint evidence remains assigned to Tasks 7 and 8; Task 6 and Task 7 are not marked complete here.

## Task 6: ACL Primitive Compatibility Matrix

**Files:**
- Create: `tests/test_acl_primitives.py`
- Modify only when a focused failure proves it necessary: `plr3d/ops/geometry.py`, `rot_jittor/src/model/feature.py`, or the smallest affected model file

- [x] **Step 1: Write focused primitive tests**

Create a unittest class skipped unless `jt.compiler.has_acl` is true. Its `setUpClass` sets `jt.flags.use_acl=1`. Add independent tests named:

```python
test_matmul_and_bmm_fp32
test_linear_conv1d_and_depthwise_conv1d
test_batchnorm_layernorm_and_activations
test_gather_scatter_argmax_and_topk
test_coordinate_and_feature_knn
test_optimizer_step_and_checkpoint_roundtrip
```

Use fixed float32 inputs. Compare small outputs with NumPy or a CPU-Jittor reference. Use `rtol=1e-4, atol=1e-5` unless the pinned Jittor ACL test for the same operator documents a looser tolerance. The checkpoint test uses `tempfile.TemporaryDirectory`.

- [x] **Step 2: Run the matrix and stop at the first unsupported operation**

Run remotely:

```bash
python -m unittest -v tests.test_acl_primitives
```

Expected: supported primitives pass. If one fails, record its operator, dtype, shape, Jittor log, and CANN error before changing code.

- [x] **Step 3: Apply systematic debugging to each failure**

For one failure at a time:

1. reproduce the single operation in a standalone test;
2. compare with the official Jittor ACL implementation/tests at the pinned commit;
3. determine whether the cause is project shape usage, unsupported ACL routing, or a Jittor defect;
4. add a regression test that fails for the identified cause;
5. implement the smallest Jittor composition that remains on ACL;
6. reject `.numpy()` and `.item()` inside formal forward/backward paths.

- [x] **Step 4: Verify the complete matrix and commit**

Expected: all six tests pass on one Ascend 910 while `use_acl` remains enabled,
and the repaired gather/KNN branches contain no explicit host-transfer fallback.

Commit only the files actually required:

```bash
git add tests/test_acl_primitives.py plr3d rot_jittor
git commit -m "test: validate PLR primitives on Jittor ACL"
```

### Task 6 Verification Evidence — 2026-08-10

- `python -m unittest -v tests.test_acl_primitives` ran 6/6 tests successfully in the activated one-NPU runtime: Jittor `06f5d3d271555682c95aa3505518f47eeab2bd9c`, CANN 9.1, and `has_acl=1`. Every test asserts `use_acl==1` at its beginning and end; the fixed Jittor runtime reports `(use_acl, use_cuda)=(1,1)` after ACL selection.
- The matrix covers float32 MatMul/BMM plus a gradient, Linear/Conv1d/depthwise Conv1d, BatchNorm train/backward/running-stat/eval semantics, LayerNorm/ReLU/SiLU, project gather forward/backward, scatter/ArgMax/top-k, coordinate and feature KNN, and an exact SGD update plus model-checkpoint round trip. Numerical comparisons use NumPy references at `rtol=1e-4`, `atol=1e-5`.
- Compatibility repair required: ACL does not support the project’s flattened advanced `Index` gather or CUDA-only `jt.misc.knn`, and the pinned vendor `GatherACL.grad` allocates an index-shaped rather than source-shaped gradient. The project now uses ACL Gather with a source-shaped scatter-add gradient, and both PLR and ROT coordinate KNN use pairwise squared-distance/top-k whenever `use_acl=1`. The matrix proves that these paths pass while the ACL flag remains enabled. The repaired gather and KNN branches contain no explicit `.numpy()`, `.item()`, or host-transfer fallback; this evidence does not claim per-operator backend provenance inside vendor Jittor. No vendor source was changed.

## Task 7: Single-NPU Model and Training Smoke

**Files:**
- Modify: `tests/smoke_jittor.py`
- Modify: `README.md`

- [ ] **Step 1: Add reduced-model acceptance tests before compatibility changes**

In ACL mode the smoke must:

1. construct `DenoiseNet(frame_knn=4, num_modules=1)`;
2. verify finite forward output on a fixed small point tensor;
3. calculate a scalar loss and gradients;
4. apply one optimizer update;
5. save and reload a checkpoint in a temporary directory;
6. compare pre-save and post-load output within float32 tolerance;
7. print JSON containing device status, elapsed seconds, input/output shapes, and parameter count.

- [ ] **Step 2: Verify RED at the first model incompatibility**

Run remotely:

```bash
python main.py doctor --device acl --deep
```

Expected before all compatibility work is complete: doctor reports the first failing model operation rather than silently using CPU.

- [ ] **Step 3: Fix one reported incompatibility at a time**

Use `systematic-debugging` for every model-level failure. Add the smallest focused test to `tests/test_acl_primitives.py` or `tests/smoke_jittor.py`, verify it fails for the right reason, implement one fix, and rerun both the focused test and full smoke.

- [ ] **Step 4: Run the Milestone A gate**

Inside the activated remote environment run:

```bash
python -m unittest tests.test_source_policy tests.test_routing
python -m unittest -v tests.test_acl_primitives
python main.py doctor --device acl --deep
```

Expected: every command exits 0 on one Ascend 910.

- [ ] **Step 5: Document accepted scope and commit**

Update `README.md` with:

- the pinned Jittor commit;
- offline bundle/setup commands;
- `source scripts/ascend_env.sh`;
- `python main.py doctor --device acl --deep`;
- the statement that ACL selective scan is the correctness reference path pending Milestone B optimization.

Commit:

```bash
git add tests/smoke_jittor.py tests/test_acl_primitives.py README.md
git commit -m "feat: complete single-NPU Ascend correctness smoke"
```

## Task 8: Final Synchronization and Evidence

**Files:**
- Modify: `docs/superpowers/plans/2026-08-10-ascend-single-npu.md`

- [ ] **Step 1: Run final local verification**

```bash
python3 -m compileall -q .
python3 -m unittest tests.test_source_policy tests.test_backend tests.test_ascend_environment_contract tests.test_device_wiring
python3 tools/audit_source_archive.py --root .
git diff --check origin/new...HEAD
```

Expected: all local checks pass.

- [ ] **Step 2: Push and synchronize the validated tree**

```bash
git push origin new
scripts/sync_ascend.sh
```

- [ ] **Step 3: Run final remote evidence**

Inside `flagtree-dev-ldc` with `scripts/ascend_env.sh` sourced:

```bash
cd /data/ldc/Track2-new
python main.py doctor --device acl --deep
python -m unittest -v tests.test_acl_primitives
```

- [ ] **Step 4: Record results and commit the plan status**

Record the Jittor SHA, CANN path, `has_acl`, primitive test count, model-smoke result, elapsed time, and explicitly deferred performance/HCCL work in this plan. Then run:

```bash
git add docs/superpowers/plans/2026-08-10-ascend-single-npu.md
git commit -m "docs: record single-NPU Ascend verification"
git push origin new
```

Milestone A is complete only when both final local checks and remote ACL evidence pass. Selective-scan optimization, KNN optimization, and HCCL enablement remain separate plans.
