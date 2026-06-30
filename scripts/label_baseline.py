"""Label-sanity baseline: per-class pixel distribution + majority-class floor.

Runs the ``MetalScrapDataModule`` (label-only, no cubes) + ``RgbLabelToClassIndex`` over the
real dataset-level split and writes ``reports/metal_scrap_label_baseline.md``. This validates
the provisional 29 -> 14 merge and the DataSet0-2 / DataSet3 split produce a plausible
training target before any model work.

Run:
  PYTHONPATH=<plugin> <cuvis-ai venv python> scripts/label_baseline.py [--root <HSIMetalScrap>]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from cuvis_ai_inspecscrap.data import MetalScrapDataModule
from cuvis_ai_inspecscrap.node.labels import PAPER_CLASSES, RgbLabelToClassIndex

_DEFAULT_ROOT = "data/HSIMetalScrap"
_LABELMAP = "/LabelMap.txt"


def _counts_for_stage(ds, mapper: RgbLabelToClassIndex, n_classes: int) -> tuple[torch.Tensor, int]:
    """Return per-class pixel counts [n_classes] and the ignored-pixel count for a dataset."""
    counts = torch.zeros(n_classes, dtype=torch.int64)
    ignored = 0
    for i in range(len(ds)):
        label = torch.as_tensor(ds[i]["label_rgb"]).unsqueeze(0)  # [1,H,W,3]
        targets = mapper.forward(label_rgb=label)["targets"].reshape(-1)
        ignored += int((targets == mapper.ignore_index).sum())
        valid = targets[targets != mapper.ignore_index]
        if valid.numel():
            counts += torch.bincount(valid, minlength=n_classes)
    return counts, ignored


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=_DEFAULT_ROOT)
    ap.add_argument("--split-mode", default="random_frame", choices=["random_frame", "dataset"])
    ap.add_argument("--test-datasets", nargs="+", default=["DataSet3"])
    args = ap.parse_args()

    root = Path(args.root)
    mapper = RgbLabelToClassIndex(labelmap_path=str(root / "LabelMap.txt"))
    n = mapper.num_classes

    dm = MetalScrapDataModule(
        root=str(root),
        split_mode=args.split_mode,
        test_datasets=args.test_datasets,
        read_cube=False,
    )
    dm.setup()
    stages = {"train": dm.train_ds, "val": dm.val_ds, "test": dm.test_ds}

    per_stage = {}
    ignored = {}
    for name, ds in stages.items():
        c, ig = _counts_for_stage(ds, mapper, n)
        per_stage[name] = c
        ignored[name] = ig

    total = sum(per_stage.values())
    grand = int(total.sum())

    lines: list[str] = []
    split_desc = (
        "seeded random frame-level 80/20 split (split_mode=random_frame)"
        if args.split_mode == "random_frame"
        else f"dataset-level split (test = {', '.join(args.test_datasets)})"
    )
    lines.append("# Metal-scrap label-sanity baseline\n")
    lines.append(
        f"Per-class labeled-pixel distribution over the {split_desc}, with the provisional "
        "29 -> 14 merge from `RgbLabelToClassIndex`. Ignored pixels (background, the "
        "`(255,204,51)` collision, and unmapped colours) are excluded from class counts.\n"
    )
    sizes = {k: len(v) for k, v in stages.items()}
    lines.append(
        f"Frames: train={sizes['train']}, val={sizes['val']}, test={sizes['test']} "
        f"(total {sum(sizes.values())}).\n"
    )
    lines.append("| id | class | train px | val px | test px | total px | % of labeled |")
    lines.append("|---|---|---:|---:|---:|---:|---:|")
    for cid in range(n):
        tot = int(total[cid])
        pct = (100.0 * tot / grand) if grand else 0.0
        lines.append(
            f"| {cid} | {PAPER_CLASSES[cid]} | {int(per_stage['train'][cid])} | "
            f"{int(per_stage['val'][cid])} | {int(per_stage['test'][cid])} | {tot} | {pct:.3f} |"
        )
    lines.append(f"\nLabeled pixels total: {grand:,}")
    lines.append(
        f"Ignored pixels: train={ignored['train']:,}, val={ignored['val']:,}, "
        f"test={ignored['test']:,}\n"
    )

    lines.append("## Majority-class floor (sanity)\n")
    lines.append("Accuracy if always predicting the most-frequent labeled class, per split.\n")
    lines.append("| split | majority class | floor accuracy |")
    lines.append("|---|---|---:|")
    for name in ("train", "val", "test"):
        c = per_stage[name]
        s = int(c.sum())
        if s:
            cid = int(c.argmax())
            lines.append(f"| {name} | {PAPER_CLASSES[cid]} | {100.0 * int(c[cid]) / s:.2f}% |")
        else:
            lines.append(f"| {name} | - | - |")

    absent_test = [PAPER_CLASSES[c] for c in range(n) if int(per_stage["test"][c]) == 0]
    lines.append("\n## Notes\n")
    lines.append(
        "- A **dataset-level** split (DataSet0-2 train / DataSet3 test) was tried first and rejected: "
        "DataSet3 is ~68% `dark_rust_metal`, a class with ~0 pixels in DataSet0-2, while train-only "
        "classes are absent from DataSet3. The four DataSets are not comparable batches, so a "
        "random frame-level split is used to keep every class in train and test. This reinstates a "
        "file-level (not piece-level) leakage guard, pending JOANNEUM piece ids.\n"
        f"- Classes absent from the test split (excluded from macro metrics via zero_division): "
        f"{', '.join(absent_test) or 'none'}.\n"
        "- The `light_rust_metal` class is ~0 px: its only source colour `(255,204,51)` is the "
        "`Me-LightRusty` / `Plastic_Packaging` collision, resolved to ignore pending the original "
        "JOANNEUM per-pixel class ids.\n"
        "- Background dominates raw pixels (~75%) and is excluded here via ignore_index.\n"
        "- This is the provisional merge; sign-off pending before final numbers."
    )

    out = Path(__file__).resolve().parent.parent / "reports" / "metal_scrap_label_baseline.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\n[written] {out}")


if __name__ == "__main__":
    main()
