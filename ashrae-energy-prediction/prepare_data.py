"""Prepare the tables the notebook stages start from: raw Kaggle CSVs -> artifacts/*.parquet.

The code lives in research.ipynb; this script only executes those cells by id, so the
notebook stays the single source of truth:

    8fa98841   read CSV files           861ffc77   UTC offsets and alignment helper
    cbad36d1   aligned weather + temperature windows (3/24/72 h, lag, difference)
    9764b273   train + building metadata + weather        constant-runs   constant_run flag

Usage (project root, data in ashrae-energy-prediction/):
    python prepare_data.py [--output-dir artifacts]

Writes train_full.parquet, test.parquet and weather_test_aligned.parquet.
Peak memory is about 5 GB because test.csv has 41.7 million rows.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--notebook", default="research.ipynb")
parser.add_argument("--data-dir", default="ashrae-energy-prediction")
parser.add_argument("--output-dir", default="artifacts")
args = parser.parse_args()

cells = {
    cell["id"]: "".join(cell["source"])
    for cell in json.load(open(args.notebook, encoding="utf-8"))["cells"]
}
output = Path(args.output_dir)
output.mkdir(parents=True, exist_ok=True)

load = cells["8fa98841"].replace('DATA_DIR = "ashrae-energy-prediction"', f'DATA_DIR = "{args.data_dir}"')
# Compact dtypes for test.csv and no sample_submission: the notebook cell would need ~8 GB.
load = load.replace(
    'pd.read_csv(f"{DATA_DIR}/test.csv")',
    'pd.read_csv(f"{DATA_DIR}/test.csv", dtype={"row_id": "int32", "building_id": "int16", "meter": "int8"}, '
    'usecols=["row_id", "building_id", "meter", "timestamp"], parse_dates=["timestamp"])',
)
load = load.replace('pd.read_csv(f"{DATA_DIR}/sample_submission.csv")', "pd.DataFrame()")

namespace = {
    "np": np, "pd": pd,
    "display": lambda obj: print(obj.to_string() if hasattr(obj, "to_string") else obj),
}
started = time.time()

for name, code in [
    ("read CSV", load),
    ("UTC offsets", cells["861ffc77"]),
    ("weather windows", cells["cbad36d1"]),
    ("train_full", cells["9764b273"]),
    ("timestamp", 'train_full["timestamp"] = pd.to_datetime(train_full["timestamp"])'),
    ("constant runs", cells["constant-runs"]),
]:
    exec(compile(code, name, "exec"), namespace)
    print(f"[{time.time() - started:5.0f} s] {name}", flush=True)

namespace["train_full"].to_parquet(output / "train_full.parquet", compression="zstd", index=False)
namespace["weather_test_aligned"].to_parquet(output / "weather_test_aligned.parquet", index=False)
namespace["test"].to_parquet(output / "test.parquet", compression="zstd", index=False)
print("written:", sorted(path.name for path in output.glob("*.parquet")))
