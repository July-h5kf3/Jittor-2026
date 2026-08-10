# PLR-003 Ascend Migration Design

## 1. Goal

Migrate the best-performing PLR-003 Jittor baseline from NVIDIA CUDA to Huawei Ascend while preserving the model architecture, training graph, frozen routing policy, checkpoint schema, and canonical output constraints.

The migration is staged:

1. establish numerically credible single-NPU training and inference;
2. optimize Ascend bottlenecks without changing model semantics;
3. enable and validate 2-, 4-, and 8-NPU data parallelism through HCCL.

The historical CUDA implementation remains available as a compatibility and numerical reference. The Ascend migration must not silently replace unsupported device work with CPU execution.

## 2. Fixed Environment

- Host: `ssh zhiyuan-huawei`
- Source deployment: `/data/ldc/Track2-new`
- Container: running container `flagtree-dev-ldc`
- Image: `flagtree-ldc-base:cann9-working`
- Accelerator: 8 × Ascend 910 exposed as `/dev/davinci0` through `/dev/davinci7`
- Python: 3.11.15 inside the container
- CANN: use the environment resolved by the image entrypoint, currently `ASCEND_TOOLKIT_HOME=/usr/local/Ascend/cann-9.1.0-beta.1`
- Jittor source: official commit `06f5d3d271555682c95aa3505518f47eeab2bd9c`
- Python environment: `/data/ldc/envs/track2-ascend`
- Jittor source snapshot: `/data/ldc/vendor/jittor-06f5d3d271555682c95aa3505518f47eeab2bd9c`
- Jittor cache: `/data/ldc/cache/jittor-track2-ascend`

The remote host is not expected to access GitHub. Jittor source, dependency wheels, and project changes are downloaded or prepared locally and transferred over SSH.

## 3. Source and Synchronization Policy

All durable code changes occur first in the local `new` branch. Each validated batch follows this order:

1. edit and test locally;
2. commit to local `new`;
3. push to `origin/new` from the local machine;
4. synchronize the reviewed working tree to `/data/ldc/Track2-new` with `.git`, caches, virtual environments, logs, datasets, checkpoints, and generated predictions excluded;
5. run Ascend-only validation in `flagtree-dev-ldc`.

Remote hot fixes are not authoritative. If an emergency diagnostic edit is made remotely, it must be reproduced locally before the next synchronization.

## 4. Runtime Architecture

### 4.1 Backend abstraction

Introduce a small runtime module responsible for selecting and validating one of three backends:

- `cpu`: `jt.flags.use_cuda=0`, `jt.flags.use_acl=0`;
- `cuda`: preserve the historical CUDA behavior;
- `acl`: require `jt.compiler.has_acl`, set `jt.flags.use_acl=1`, and use the Ascend runtime.

Every executable entry point accepts the same `--device {cpu,cuda,acl}` interface. Configuration generation uses a device parameter instead of embedding `cuda` literals.

Selecting `acl` when the ACL backend is unavailable is a hard error. No automatic CPU fallback is allowed in training or production inference.

### 4.2 Environment doctor

`main.py doctor` reports:

- Python and Jittor versions;
- the exact Jittor Git commit;
- `ASCEND_TOOLKIT_HOME` and `tikcc_path`;
- `has_acl`, `use_acl`, visible device count, current device, MPI rank, and world size;
- HCCL availability when launched under MPI;
- the active Jittor cache directory.

`doctor --deep --device acl` runs representative forward and backward operations on the NPU and fails if work is executed through an unintended CPU fallback.

## 5. Phase 1: Single-NPU Correctness

### 5.1 Selective scan

The existing CUDA `jt.code` forward/backward remains unchanged for `cuda`. For the first Ascend milestone, `acl` uses `selective_scan_reference`, which is composed of ordinary differentiable Jittor tensor operations.

This path is expected to be slower but must preserve:

- float32 state computation;
- `d_state=16` behavior;
- softplus delta and SiLU gating;
- checkpoint/state-dict names and shapes;
- automatic differentiation for training.

The backend selection is explicit and independently tested. The ACL path must never enter `_SelectiveScanCUDA`.

### 5.2 Operator compatibility audit

Test the actual shapes used by PLR-003 for:

- array creation, elementwise math, reductions, broadcast, reshape, transpose, concatenate, and indexing;
- matmul and batched matmul using the FP32 precision fix in the pinned Jittor commit;
- Linear, depthwise Conv1d, BatchNorm, LayerNorm, ReLU, LeakyReLU, sigmoid, and optimizer updates;
- gather, scatter, argmax, argsort/topk, coordinate KNN, feature KNN, and farthest-point sampling;
- checkpoint save/load and NumPy input/output.

When an operator is unsupported, the first fallback is another composition of Jittor operations that remains on ACL. CPU transfer is allowed only in an explicitly named diagnostic test, never in a formal train or inference path.

### 5.3 Correctness gates

The first milestone requires:

- Jittor ACL import and official ACL smoke tests pass;
- project source-policy and routing tests pass;
- selective scan forward and gradients agree with the CPU reference within documented float32 tolerances;
- a small DenoiseNet forward/backward/update succeeds on one NPU;
- checkpoint save/reload produces the same output within tolerance;
- a reduced point-cloud inference smoke preserves point count, order, dtype, and finite values;
- one short training smoke produces finite loss and gradients.

Historical CUDA weights and frozen sample arrays may be used as numerical references, but no result is described as performance-equivalent until the complete Ascend pipeline is evaluated.

## 6. Phase 2: Single-NPU Performance

Profile before replacing the reference implementation. Expected priority order:

1. selective scan forward/backward;
2. coordinate and feature KNN/topk;
3. gather/scatter and patch stitching;
4. normalization and depthwise Conv1d;
5. Python-controlled farthest-point sampling loops.

The optimized selective scan first uses Jittor's ACL integration and ACL/ACLNN operator runners when profiling confirms that the recurrence can be expressed without per-step host synchronization. If that condition is not met, implement the recurrence as a CANN custom operator. Both implementations retain the reference path for numerical verification.

Optimization is accepted only when:

- forward and gradient tolerances remain satisfied;
- checkpoint keys and tensor shapes remain unchanged;
- train and inference outputs remain finite and canonical;
- measured latency or throughput improves on the same NPU and workload.

## 7. Phase 3: Multi-NPU HCCL

### 7.1 Jittor patch boundary

The pinned Jittor commit contains HCCL operators, device/rank mapping, and MPI routing, but `setup_hccl()` is currently disabled in `compile_extern.py`. Maintain a minimal, reviewable patch set outside upstream source history that:

1. calls `setup_hccl()` when ACL and MPI are both available;
2. avoids initializing NCCL/CUDA external libraries in ACL mode;
3. preserves ordinary MPI as the control plane;
4. routes device tensor collectives through HCCL;
5. exposes enough state for `doctor` and tests to verify HCCL rather than host-memory collectives.

Store the patch in the project as `patches/jittor/06f5d3d271555682c95aa3505518f47eeab2bd9c-ascend-hccl.patch`. The environment setup script verifies the upstream commit before applying it and fails if the patch does not apply cleanly.

### 7.2 Project distributed behavior

The existing 3DMambaIPF training path already uses Jittor rank/world-size values, per-rank seeding, data sharding, parameter broadcast, effective global batch accounting, and rank-zero checkpointing. The migration will preserve and test these behaviors.

`train_rot.py` receives the same distributed audit before multi-NPU acceptance. Any training path that is not safe under MPI must fail explicitly instead of allowing multiple ranks to overwrite the same output.

### 7.3 Scaling sequence

Validate in this order:

1. two ranks on devices 0–1;
2. four ranks on devices 0–3;
3. eight ranks on devices 0–7.

For each scale, verify:

- one rank is bound to one NPU;
- parameters are identical after broadcast;
- gradient all-reduce matches the single-NPU reference after accounting for global batch size;
- loss remains finite and comparable;
- only rank zero writes shared checkpoints and manifests;
- all ranks terminate cleanly;
- throughput and scaling efficiency are recorded after warmup.

## 8. Error Handling

- Missing ACL compiler, headers, libraries, or devices: fail during doctor before model construction.
- Unsupported ACL operator: report the exact operator, dtype, and shape; do not silently use CPU.
- Selective scan numerical mismatch: stop before full-model testing.
- HCCL initialization or rank/device mismatch: abort all ranks and preserve logs per rank.
- Non-finite forward, gradient, loss, checkpoint, or prediction: stop the current stage and record the smallest reproducing configuration.
- Remote and local source mismatch: refuse experiments until synchronization is restored.

## 9. Testing Strategy

Add tests in increasing cost order:

1. local CPU tests for backend selection and configuration generation;
2. single-NPU ACL primitive matrix;
3. selective scan forward/gradient parity;
4. model-block forward/backward parity;
5. single-NPU training and inference smoke;
6. two-rank HCCL collective and training-step tests;
7. four- and eight-rank scaling tests;
8. reduced end-to-end pipeline;
9. full training/inference only after all lower-cost gates pass.

Test outputs include environment metadata, Jittor commit, project commit, CANN path, device/rank mapping, tolerances, latency, throughput, and peak memory where available.

## 10. Success Criteria

### Milestone A: functional single NPU

- official Jittor ACL backend builds from the pinned commit;
- project model trains and infers on one Ascend 910;
- numerical and canonical-output checks pass;
- no unintended CPU fallback occurs.

### Milestone B: practical single-NPU performance

- selective scan and point-neighborhood bottlenecks have Ascend-optimized paths;
- representative training and inference workloads complete within measured, documented resource limits.

### Milestone C: multi-NPU

- 2-, 4-, and 8-NPU HCCL training passes correctness checks;
- scaling throughput and efficiency are reported;
- checkpointing and recovery remain correct under eight ranks.

The historical leaderboard score remains a property of the original CUDA run. Ascend quality equivalence requires evaluation of newly produced predictions; it is not inferred solely from unit-test parity.
