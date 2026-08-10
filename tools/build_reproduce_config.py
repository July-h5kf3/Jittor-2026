#!/usr/bin/env python3
"""Generate the checked-in resumable from-raw reproduction pipeline."""

import argparse
import json
from pathlib import Path

W = "${NKAI_WORK_ROOT}"
TRAIN = "${NKAI_TRAIN_ROOT}"
TEST = "${NKAI_TEST_ROOT}"
STAGES = []


def add(name, cmd, skip=None):
    values = [str(value) for value in cmd]
    if not values:
        raise ValueError("empty pipeline command")
    row = {"name": name, "entrypoint": values[0], "args": values[1:]}
    if skip is not None:
        row["skip_if_exists"] = skip
    STAGES.append(row)


def train(spec, device):
    core = spec[:18]
    accumulation = spec[18] if len(spec) > 18 else 4
    name, base, data, prefix, epochs, samples, count, steps, smin, smax, gp, outliers, lo, hi, lr, dcd, seed, namespace = core
    output = f"{W}/training/{name}"
    command = ["train"]
    command += ["--random-init"] if base is None else ["--checkpoint", base]
    command += [
        "--data-root", f"{W}/data/{data}", "--train-list", f"{W}/lists/{data}.txt",
        "--output-root", output, "--checkpoint-prefix", prefix,
        "--epochs", epochs, "--samples-per-shape", samples, "--batch-size", 4,
        "--grad-accum-steps", accumulation, "--num-workers", 4, "--patch-size", 1000,
        "--patch-ratio", 1.2, "--sigma-min", smin, "--sigma-max", smax,
        "--gaussian-probability", gp, "--outlier-fraction", outliers,
        "--outlier-scale", 3, "--scale-min", lo, "--scale-max", hi,
        "--lr", lr, "--weight-decay", 0, "--grad-clip", 1,
        "--dcd-alpha", 100, "--dcd-n-lambda", 0.5, "--dcd-weight", dcd,
        "--seed", seed, "--seed-namespace", namespace, "--device", device,
        "--expected-train-count", count,
    ]
    if steps is not None:
        command += ["--expected-steps-per-epoch", steps]
    checkpoint = f"{output}/{prefix}-epoch{int(epochs):02d}.pkl"
    add(f"train_{name}", command, checkpoint)
    return checkpoint


def filter_list(name, categories, count):
    output = f"{W}/lists/{name}.txt"
    command = [
        "filter-keys", "--input", f"{W}/lists/test_all.txt", "--output", output,
        "--expected-count", count,
    ]
    for category in categories:
        command += ["--category", category]
    add(f"filter_{name}", command, output)
    return output


def infer(name, checkpoint, key_list, count, device):
    output = f"{W}/predictions/{name}"
    manifest = f"{W}/manifests/infer_{name}.json"
    add(
        f"infer_{name}",
        ["infer", "--checkpoint", checkpoint, "--input-root", TEST,
         "--key-list", key_list, "--expected-count", count,
         "--output-root", output, "--manifest", manifest, "--device", device],
        manifest,
    )
    return output


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device", choices=("cpu", "cuda", "acl"), default="cuda"
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    STAGES.clear()
    prepared = [
        ("base", "train_base.txt", None),
        ("mbi009_650", "train_mbi009_650.txt", None),
        ("dcd001_3000", "train_dcd001_3000.txt", None),
        ("table_1200", "train_table_1200.txt", None),
        ("airplane_1000", "train_airplane_1000.txt", None),
        ("sofa_1000", "train_sofa_1000.txt", None),
        ("tail_1885", "train_tail_1885.txt", "configs/lists/train_tail_surface_seeds.json"),
    ]
    for preset, selected, seed_map in prepared:
        manifest = f"{W}/manifests/data_{preset}.json"
        command = [
            "prepare-data", "--preset", preset,
            "--candidate-list", "configs/lists/train_base.txt",
            "--selected-list", f"configs/lists/{selected}",
            "--mesh-root", TRAIN, "--output-root", f"{W}/data/{preset}",
            "--output-list", f"{W}/lists/{preset}.txt", "--manifest", manifest,
        ]
        if seed_map:
            command += ["--seed-map-json", seed_map]
        add(f"prepare_{preset}", command, manifest)

    base = train(("base", None, "base", "base-random", 100, 4, 15733, None,
                  .005, .02, .5, .02, .8, 1.2, 1e-4, 0, 2020, "base-item", 8), args.device)
    mbi = train(("mbi009", base, "mbi009_650", "mbi009-ft-mix", 3, 4, 650, 163,
                 .005, .02, .5, .02, .8, 1.2, 1e-5, 0, 20260814, "mbi009-item"), args.device)
    dcd = train(("dcd001", mbi, "dcd001_3000", "dcd001-freq-dcd", 2, 1, 3000, 188,
                 .005, .02, .5, .02, .8, 1.2, 5e-6, .5, 20261303, "dcd001-item"), args.device)
    mbi11 = train(("mbi011_table", mbi, "table_1200", "mbi011-cat04379243-heavy", 2, 2, 1200, 150,
                   .005, .02, .4, .02, .8, 1.2, 5e-6, 0, 20260903, "mbi009-item"), args.device)
    tsd3 = train(("tsd003", mbi11, "table_1200", "tsd003-dcd", 2, 1, 1200, 75,
                  .005, .02, .4, .02, .8, 1.2, 2e-6, .5, 20261411, "dcd001-item"), args.device)
    tsd4 = train(("tsd004", tsd3, "table_1200", "tsd004-strong", 2, 1, 1200, 75,
                  .005, .02, .4, .02, .8, 1.2, 1e-6, 1, 20261511, "dcd001-item"), args.device)
    ssd = train(("ssd009", dcd, "sofa_1000", "ssd009-strong", 2, 1, 1000, 63,
                 .005, .02, .4, .02, .8, 1.2, 2e-6, 1, 20262011, "dcd001-item"), args.device)
    airplane = train(("plr_airplane", dcd, "airplane_1000", "plr001-airplane", 2, 4, 1000, 250,
                      .012489692407870212, .014616960844581714, 0, 0, 1, 1, 1e-6, .5, 20262701, "dcd001-item"), args.device)
    table = train(("plr_table", tsd4, "table_1200", "plr001-table", 2, 4, 1200, 300,
                   .009048618504872253, .012903604290095811, 0, 0, 1, 1, 1e-6, 1, 20262702, "dcd001-item"), args.device)
    sofa = train(("plr_sofa", ssd, "sofa_1000", "plr001-sofa", 2, 4, 1000, 250,
                  .008374454205880252, .011346570064822026, 0, 0, 1, 1, 1e-6, 1, 20262703, "dcd001-item"), args.device)

    tail = f"{W}/training/plr_tail/plr003-tail-epoch02.pkl"
    add("train_plr_tail", [
        "train", "--checkpoint", dcd, "--data-root", f"{W}/data/tail_1885",
        "--train-list", f"{W}/lists/tail_1885.txt", "--output-root", f"{W}/training/plr_tail",
        "--checkpoint-prefix", "plr003-tail", "--epochs", 2, "--samples-per-shape", 4,
        "--batch-size", 4, "--grad-accum-steps", 4, "--num-workers", 4,
        "--patch-size", 1000, "--patch-ratio", 1.2,
        "--sigma-map-json", "configs/plr003_tail_sigma.json",
        "--gaussian-probability", 0, "--outlier-fraction", 0, "--outlier-scale", 3,
        "--scale-min", 1, "--scale-max", 1, "--lr", "1e-6", "--weight-decay", 0,
        "--grad-clip", 1, "--dcd-alpha", 100, "--dcd-n-lambda", .5, "--dcd-weight", .5,
        "--seed", 20262731, "--seed-namespace", "plr003-tail-item",
        "--expected-train-count", 1885, "--expected-steps-per-epoch", 472, "--device", args.device,
    ], tail)

    rot_checkpoint = f"{W}/training/rot/spcf-final.pkl"
    add("train_rot", ["train-rot", "--mesh-root", TRAIN,
                       "--train-list", "configs/lists/rot_train.txt",
                       "--validation-list", "configs/lists/rot_validate.txt",
                       "--output-root", f"{W}/training/rot", "--resume",
                       "--device", args.device], rot_checkpoint)

    all_keys = f"{W}/lists/test_all.txt"
    add("discover_test_keys", ["prepare-keys", "--root", TEST,
                                "--filename", "noisy.npy", "--output", all_keys], all_keys)
    dcd_keys = filter_list("test_dcd84", ["02691156", "03046257", "03642806", "04330267", "04468005", "04256520"], 84)
    table_keys = filter_list("test_table92", ["04379243"], 92)
    sofa_keys = filter_list("test_sofa30", ["04256520"], 30)
    airplane_keys = filter_list("test_airplane35", ["02691156"], 35)
    tail_keys = filter_list("test_tail40", ["02871439", "02876657", "03046257", "03642806", "04330267", "04401088", "04468005"], 40)

    rot = f"{W}/predictions/rot/canonical"
    add("generate_rot", ["generate-rot", "--input-root", TEST, "--key-list", all_keys,
                         "--checkpoint", rot_checkpoint, "--output-root", f"{W}/predictions/rot",
                         "--expected-count", 200, "--device", args.device],
        f"{W}/predictions/rot/canonical_manifest.json")
    incumbent_member = infer("mbi009", mbi, all_keys, 200, args.device)
    dcd_pred = infer("dcd84", dcd, dcd_keys, 84, args.device)
    full_table = infer("full_table92", tsd4, table_keys, 92, args.device)
    full_sofa = infer("full_sofa30", ssd, sofa_keys, 30, args.device)
    plr_airplane = infer("plr_airplane35", airplane, airplane_keys, 35, args.device)
    plr_table = infer("plr_table92", table, table_keys, 92, args.device)
    plr_sofa = infer("plr_sofa30", sofa, sofa_keys, 30, args.device)
    plr_tail = infer("plr_tail40", tail, tail_keys, 40, args.device)

    incumbent = f"{W}/predictions/incumbent"
    inc_manifest = f"{W}/manifests/incumbent.json"
    add("assemble_incumbent", ["assemble-incumbent", "--key-list", all_keys,
         "--noisy-root", TEST, "--rot-root", rot, "--member-root", incumbent_member,
         "--output-root", incumbent, "--manifest", inc_manifest, "--expected-count", 200], inc_manifest)
    full = f"{W}/predictions/full"
    full_manifest = f"{W}/manifests/full.json"
    add("assemble_full", ["assemble-full", "--key-list", all_keys, "--noisy-root", TEST,
         "--rot-root", rot, "--incumbent-root", incumbent, "--dcd-root", dcd_pred,
         "--table-root", full_table, "--sofa-root", full_sofa,
         "--output-root", full, "--manifest", full_manifest, "--expected-count", 200], full_manifest)
    plr = f"{W}/predictions/plr003"
    plr_manifest = f"{W}/manifests/plr003.json"
    add("assemble_plr003", ["assemble-plr003", "--key-list", all_keys,
         "--noisy-root", TEST, "--full-root", full, "--rot-root", rot,
         "--full-dcd-root", dcd_pred, "--airplane-root", plr_airplane,
         "--table-root", plr_table, "--sofa-root", plr_sofa, "--tail-root", plr_tail,
         "--output-root", plr, "--manifest", plr_manifest, "--expected-count", 200], plr_manifest)
    result = f"{W}/result_plr003.zip"
    result_audit = f"{W}/manifests/result_zip_audit.json"
    add("package_result", ["package-results", "--root", plr, "--key-list", all_keys,
         "--zip", result, "--audit", result_audit, "--expected-count", 200], result_audit)
    source_audit = f"{W}/manifests/source_policy_audit.json"
    add("audit_source", ["audit-source", "--root", ".",
                         "--output", source_audit], source_audit)

    payload = {
        "version": 1,
        "name": "NKAI Track2 PLR-003 from-raw reproduction",
        "required_environment": ["NKAI_TRAIN_ROOT", "NKAI_TEST_ROOT", "NKAI_WORK_ROOT"],
        "notes": [
            "All checkpoints are created in NKAI_WORK_ROOT from random initialization.",
            "Included key lists contain only public training identifiers, not point arrays or weights.",
            "Delete a completed stage artifact before intentionally rerunning that stage.",
        ],
        "stages": STAGES,
    }
    destination = Path(__file__).resolve().parents[1] / "configs" / "reproduce_full.json"
    destination.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(STAGES)} stages to {destination}")


if __name__ == "__main__":
    main()
