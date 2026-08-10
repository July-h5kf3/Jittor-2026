# New Baseline Import Design

## Goal

Make `contest2_NKAI_031.zip` the sole development baseline on branch `new`. Existing uncommitted work on `main` is intentionally excluded, and no remote changes are made.

## Source

- Archive: `/Users/lorn/Downloads/contest2_NKAI_031.zip`
- Code source: `contest2_NKAI_031/code/`
- Environment source: `contest2_NKAI_031/environment.yaml`
- Submission document source: the PDF at the archive root

## Target Structure

The archive's `code/` directory is promoted to the repository root:

```text
Track2/
├── configs/
├── docs/
│   ├── submission-document.md
│   ├── submission-document.tex
│   └── submission-document.pdf
├── plr3d/
├── rot_jittor/
├── tests/
├── tools/
├── main.py
├── train.py
├── train_rot.py
├── infer.py
├── generate_rot.py
├── assemble*.py
├── package_results.py
├── environment.yaml
├── requirements.txt
└── project documentation
```

## Import Rules

1. Start `new` from clean `origin/main` in an isolated worktree.
2. Remove the tracked legacy project files from the `new` worktree before importing the archive.
3. Promote all meaningful files from the archive's `code/` directory to the repository root.
4. Move the archive-level environment file to the repository root.
5. Normalize the three submission-document formats under `docs/` with readable names.
6. Keep one root `requirements.txt`; use the documented minimal Jittor/NumPy runtime requirements.
7. Omit the Windows artifact `NUL`, empty directories, generated caches, weights, data, predictions, and logs.
8. Update documentation paths affected by removing the outer `code/` directory.
9. Normalize imported text files to LF line endings and remove trailing horizontal whitespace without changing program values or behavior.

## Validation

- Confirm the branch is `new` and the original `main` worktree remains unchanged.
- Confirm all expected archive source files are represented or intentionally omitted.
- Run Python bytecode compilation.
- Run `tests.test_source_policy` and `tests.test_routing`.
- Run the source archive audit tool.
- Record any GPU/Jittor-only checks that cannot run on the local host without treating them as silently passing.

## Remote Policy

Do not push, force-update, or otherwise modify the remote. The user will review the local `new` branch first.

After approval, transfer the reviewed local source through `ssh zhiyuan-huawei` into `/data/ldc`; the remote host must not be expected to access GitHub directly. Run experiments only inside the available image whose name or configuration includes `ldc`.
