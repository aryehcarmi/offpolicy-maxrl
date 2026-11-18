"""Prepare grouped, size-stratified splits and frozen panels from the pinned Parquet."""
import argparse
import hashlib
import json
import random
import shutil
import tempfile
import urllib.request
from collections import Counter
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

DATASET = "Jiayi-Pan/Countdown-Tasks-3to4-Unique"
REVISION = "8af28fc6a897ea9f8d343d450e3f3b16b5d4fdfa"
SHA256 = "73e62affc7e4ba26462d2edc6a4f7472f344ec432c7d1383868be8c3d3f44dea"
URL = f"https://huggingface.co/datasets/{DATASET}/resolve/{REVISION}/data/train-00000-of-00001.parquet"
RESERVE = {"validation": 128, "discovery": 32, "confirmation": 32, "test": 256}


@dataclass(frozen=True)
class Puzzle:
    puzzle_id: str
    numbers: list[int]
    target: int


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _reserve(strata, count, rng):
    """Sample a fixed total, proportional to each remaining stratum's size."""
    total = sum(map(len, strata))
    if count > total:
        raise ValueError(f"requested {count} puzzles from {total}")
    size3 = (count * len(strata[0]) + total // 2) // total if total else 0
    selected = []
    for population, take in zip(strata, (size3, count - size3), strict=True):
        selected.extend(population[:take])
        del population[:take]
    rng.shuffle(selected)
    return selected


def split_rows(rows: Iterable[dict], seed: int, reserve=RESERVE):
    # One observed presentation per canonical key; result is independent of source row order.
    groups = {}
    n_rows = 0
    for row in rows:
        n_rows += 1
        numbers, target = tuple(row["nums"]), row["target"]
        if len(numbers) not in (3, 4) or any(type(i) is not int or i <= 0 for i in numbers):
            raise ValueError(f"invalid numbers at source row {n_rows}")
        if type(target) is not int:
            raise ValueError(f"invalid target at source row {n_rows}")
        key = (tuple(sorted(numbers)), target)
        if key not in groups or numbers < groups[key]:
            groups[key] = numbers

    rng = random.Random(seed)
    by_size = [[], []]
    for (key, target), numbers in sorted(groups.items()):
        puzzle_id = "-".join(map(str, key)) + "=" + str(target)
        by_size[len(key) - 3].append(Puzzle(puzzle_id, list(numbers), target))

    train, dev, holdout = [], [], []
    for puzzles in by_size:
        rng.shuffle(puzzles)
        n = len(puzzles)
        a, b = round(n * 0.90), round(n * 0.95)
        train.extend(puzzles[:a])
        dev.append(puzzles[a:b])
        holdout.append(puzzles[b:])
    rng.shuffle(train)
    panels = {"train": train, "dev_population": dev[0] + dev[1],
              "test_population": holdout[0] + holdout[1]}
    for name in ("validation", "discovery", "confirmation"):
        panels[name] = _reserve(dev, reserve[name], rng)
    panels["test"] = _reserve(holdout, reserve["test"], rng)
    return panels, n_rows


def save_manifest(folder: Path, panels: dict, n_rows: int, seed: int, source_hash: str) -> None:
    info = {"version": 1, "dataset": DATASET, "revision": REVISION, "source_sha256": source_hash,
            "source_rows": n_rows, "split_seed": seed, "panels": {}}
    for name, panel in panels.items():
        path = folder / f"{name}.jsonl"
        with path.open("w", encoding="utf-8") as stream:
            for puzzle in panel:
                stream.write(json.dumps(asdict(puzzle), separators=(",", ":")) + "\n")
        info["panels"][name] = {"file": path.name, "sha256": digest(path), "count": len(panel),
                                "by_size": dict(Counter(len(p.numbers) for p in panel))}
    (folder / "manifest.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")


def load_panel(manifest: str | Path, panel: str) -> list[Puzzle]:
    path = Path(manifest)
    info = json.loads(path.read_text(encoding="utf-8"))
    if info["version"] != 1:
        raise ValueError("unsupported split manifest")
    if panel not in ("train", *RESERVE, "midpoint"):
        raise ValueError(f"unknown experiment panel: {panel}")
    entry = info["panels"]["discovery" if panel == "midpoint" else panel]
    shard = path.parent / entry["file"]
    if digest(shard) != entry["sha256"]:
        raise ValueError(f"split checksum mismatch: {shard}")
    with shard.open(encoding="utf-8") as stream:
        puzzles = [Puzzle(**json.loads(line)) for line in stream]
    if len(puzzles) != entry["count"]:
        raise ValueError(f"unexpected panel size: {panel}")
    return puzzles[:8] if panel == "midpoint" else puzzles


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("data"))
    parser.add_argument("--parquet", type=Path, help="local copy of the pinned Parquet")
    parser.add_argument("--seed", type=int, default=623)
    args = parser.parse_args()
    if args.out.exists():
        parser.error(f"destination already exists: {args.out}")
    import pyarrow.parquet as pq

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=args.out.parent) as tmp:
        work = Path(tmp)
        source = args.parquet
        if source is None:
            source = work / "source.parquet"
            with urllib.request.urlopen(URL, timeout=120) as response, source.open("wb") as output:
                shutil.copyfileobj(response, output)
        if digest(source) != SHA256:
            raise ValueError("source Parquet SHA-256 differs from the pinned dataset")
        batches = pq.ParquetFile(source).iter_batches(columns=["nums", "target"])
        panels, count = split_rows((row for batch in batches for row in batch.to_pylist()), args.seed)
        staged = work / "prepared"
        staged.mkdir()
        save_manifest(staged, panels, count, args.seed, SHA256)
        staged.rename(args.out)
    print(args.out / "manifest.json")
    print({name: len(puzzles) for name, puzzles in panels.items()})


if __name__ == "__main__":
    main()
