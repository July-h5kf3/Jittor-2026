# New Baseline Import Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the legacy Track2 tree on local branch `new` with the cleaned, root-level contents of `contest2_NKAI_031.zip`.

**Architecture:** Treat the archive as the authoritative source. Extract it into a temporary staging directory, remove legacy tracked application files from the isolated worktree, copy `code/` to the repository root, normalize documentation and environment paths, then validate the resulting source tree without pushing it.

**Tech Stack:** Git worktrees, BSD tar, rsync, Python 3.10, Jittor, NumPy, unittest

---

### Task 1: Replace the legacy source tree

**Files:**
- Remove: `.gitignore`, `configs/`, `datalist/`, `evaluate.py`, `requirements.txt`, `run.py`, `scripts/`, `src/`, `tests/`
- Preserve: `docs/superpowers/specs/2026-08-10-new-baseline-import-design.md`
- Preserve: `docs/superpowers/plans/2026-08-10-new-baseline-import.md`
- Create: all source files contained under archive path `contest2_NKAI_031/code/`, except `NUL`

- [x] **Step 1: Verify branch and clean plan-only state**

Run:

```bash
git branch --show-current
git status --short
```

Expected: branch is `new`; only the plan file is untracked before its documentation commit.

- [x] **Step 2: Remove the tracked legacy application tree**

Run:

```bash
git rm -r .gitignore README.md configs datalist evaluate.py requirements.txt run.py scripts src tests
```

Expected: Git stages deletion of the old baseline while retaining `docs/superpowers/`.

- [x] **Step 3: Extract and import the authoritative code**

Run:

```bash
staging_dir=$(mktemp -d)
bsdtar -xf /Users/lorn/Downloads/contest2_NKAI_031.zip -C "$staging_dir"
rsync -a --exclude NUL "$staging_dir/contest2_NKAI_031/code/" ./
cp "$staging_dir/contest2_NKAI_031/environment.yaml" ./environment.yaml
mv docs/提交说明文档.md docs/submission-document.md
mv docs/提交说明文档.tex docs/submission-document.tex
cp "$staging_dir/contest2_NKAI_031/提交说明文档.pdf" docs/submission-document.pdf
```

Expected: archive `code/` contents are at repository root, `environment.yaml` is at root, and submission documents have stable ASCII filenames.

- [x] **Step 4: Confirm excluded artifacts**

Run:

```bash
test ! -e NUL
find . -name '__pycache__' -o -name '*.pyc'
```

Expected: `NUL` does not exist and the generated-cache search prints nothing.

- [x] **Step 5: Normalize source-provided text formatting**

Run `rg -l $'\r$'` to identify CRLF files and `git diff --check` to identify other trailing whitespace. Convert only the reported files to LF and remove trailing horizontal whitespace.

Expected: `rg -l $'\r$'` prints nothing and `git diff --check` succeeds after the files are staged.

### Task 2: Normalize root-level documentation

**Files:**
- Modify: `README.md`

- [x] **Step 1: Update commands for the flattened repository layout**

Apply these exact changes in `README.md`:

```diff
-conda env create -f ../environment.yaml
+conda env create -f environment.yaml
```

```diff
-cd code
```

Replace the sentence before that command block with `在仓库根目录执行：`. Also replace the text `文件位于 ZIP 根目录` with `文件位于仓库根目录`.

- [x] **Step 2: Search for stale root-layout instructions**

Run:

```bash
rg -n '\.\./environment\.yaml|^cd code$|文件位于 ZIP 根目录' README.md
```

Expected: no matches.

### Task 3: Audit imported structure

**Files:**
- Verify: repository root and archive manifest

- [x] **Step 1: Compare imported code paths with the archive**

Run:

```bash
staging_dir=$(mktemp -d)
archive_manifest=$(mktemp)
repo_manifest=$(mktemp)
bsdtar -xf /Users/lorn/Downloads/contest2_NKAI_031.zip -C "$staging_dir"
(
  cd "$staging_dir/contest2_NKAI_031/code"
  find . -type f ! -name NUL -print \
    | sed 's#^\./docs/提交说明文档\.md$#./docs/submission-document.md#' \
    | sed 's#^\./docs/提交说明文档\.tex$#./docs/submission-document.tex#' \
    | sort
) > "$archive_manifest"
find . -type f \
  ! -path './.git' \
  ! -path './.git/*' \
  ! -path './docs/superpowers/*' \
  ! -path './environment.yaml' \
  ! -path './docs/submission-document.pdf' \
  -print | sort > "$repo_manifest"
diff -u "$archive_manifest" "$repo_manifest"
cmp "$staging_dir/contest2_NKAI_031/environment.yaml" environment.yaml
cmp "$staging_dir/contest2_NKAI_031/提交说明文档.pdf" docs/submission-document.pdf
```

Expected: no archive source file is missing; extra files are limited to `.git`, `docs/superpowers/`, and the normalized environment/PDF placement.

- [x] **Step 2: Review Git changes**

Run:

```bash
git status --short
git diff --stat
git diff --check
```

Expected: replacement is limited to the approved baseline import, and `git diff --check` prints nothing.

- [x] **Step 3: Commit the baseline import**

Run:

```bash
git add -A
git commit -m "feat: import PLR-003 best-performing baseline"
```

Expected: the import is committed on `new` with no remote operation.

### Task 4: Validate the new baseline

**Files:**
- Test: `tests/test_source_policy.py`
- Test: `tests/test_routing.py`
- Test: `tests/smoke_jittor.py`
- Verify: `tools/audit_source_archive.py`

- [x] **Step 1: Compile Python sources**

Run:

```bash
python -m compileall -q .
```

Expected: exit code 0.

Local result: passed with the host command `python3 -m compileall -q .` because this macOS host does not provide a `python` alias.

- [ ] **Step 2: Run CPU-safe unit tests**

Run:

```bash
python -m unittest tests.test_source_policy tests.test_routing
```

Expected: all tests pass.

Local result: both source-policy checks passed; routing-test import is deferred because the local host does not have Jittor. It will run in the approved remote `ldc` image.

- [x] **Step 3: Run the source policy audit**

Run:

```bash
python tools/audit_source_archive.py --root .
```

Expected: the audit reports success. If it rejects `docs/superpowers/` planning metadata, rerun against a temporary export of the deliverable source tree and report the distinction explicitly.

Local result: passed with `audit_pass: true`, zero errors, 69 audited files, and 2,855,640 source bytes.

- [ ] **Step 4: Attempt the Jittor smoke test when dependencies permit**

Run:

```bash
python tests/smoke_jittor.py
```

Expected: pass on a compatible Jittor/CUDA host; otherwise record the exact missing dependency or hardware limitation.

Local result: deferred to `ssh zhiyuan-huawei` under `/data/ldc` using the `ldc` image, after user review and transfer approval.

- [ ] **Step 5: Remove generated caches and record final state**

Run:

```bash
find . -type d -name __pycache__ -prune -exec rm -rf {} +
git status --short --branch
git log -3 --oneline --decorate
```

Expected: branch is `new`; worktree is clean after any cache cleanup commit if needed; no push has occurred.

### Deferred Remote Validation (after user review)

- Transfer the reviewed local branch contents through `ssh zhiyuan-huawei` to a staging location under `/data/ldc`; do not use GitHub from the remote host.
- Select the experiment image whose name or configuration includes `ldc`.
- Run the Jittor/CUDA smoke test and later experiments inside that image.
- This remote transfer is intentionally excluded from the current local-organization phase.
