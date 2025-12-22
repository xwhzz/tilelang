import contextlib
import subprocess
import re
import os
import json
from tabulate import tabulate
import pandas as pd

try:
    import tilelang

    tilelang.disable_cache()
    os.environ["TILELANG_DISABLE_CACHE"] = "1"
except Exception:
    tilelang = None

OLD_PYTHON = os.environ.get("OLD_PYTHON", "./old/bin/python")
NEW_PYTHON = os.environ.get("NEW_PYTHON", "./new/bin/python")
OUT_MD = os.environ.get("PERF_REGRESSION_MD", "regression_result.md")
OUT_PNG = os.environ.get("PERF_REGRESSION_PNG", "regression_result.png")
_RESULTS_JSON_PREFIX = "__TILELANG_PERF_RESULTS_JSON__="


def parse_output(output):
    for line in output.splitlines():
        if line.startswith(_RESULTS_JSON_PREFIX):
            return json.loads(line[len(_RESULTS_JSON_PREFIX) :])

    # Fallback to regex parsing
    data = {}
    for line in output.split("\n"):
        line = line.strip()
        m = re.match(r"\|\s*([^\|]+)\s*\|\s*([0-9\.]+)\s*\|", line)
        if m is not None:
            with contextlib.suppress(ValueError):
                data[m.group(1)] = float(m.group(2))
    return data


def run_cmd(cmd, env=None):
    full_env = os.environ.copy()
    if env:
        full_env.update(env)
    # Don't capture stderr so that tqdm progress bar is visible
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=None, text=True, env=full_env)
    if p.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\nSTDOUT:\n{p.stdout}")
    return p.stdout


def draw(df):
    import matplotlib.pyplot as plt
    import seaborn as sns

    if len(df) == 0:
        return

    plt.figure(figsize=(max(len(df) * 2.2, 6), 20))
    sns.set_theme(style="whitegrid", font_scale=0.9)
    top3_idx = df.nlargest(min(3, len(df)), "Speedup").index
    bot3_idx = df.nsmallest(min(3, len(df)), "Speedup").index
    label_idx = set(top3_idx.tolist() + bot3_idx.tolist())

    for i, val in enumerate(df["Speedup"]):
        if i in label_idx:
            plt.text(i, val + 0.02, f"{val:.2f}x", ha="center", va="bottom", color="red", fontsize=8, fontweight="bold")

    plt.xticks(range(len(df)), df["File"], rotation=70, ha="right", fontsize=12)
    plt.ylabel("Current Speedup vs Original", fontsize=14)
    plt.title("Current Speedup vs Original", fontsize=14, fontweight="bold")
    plt.ylim(0, max(df["Speedup"]) * 1.2)
    sns.despine()
    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=300)


env = {"TL_PERF_REGRESSION_FORMAT": "json"}
print("Running regression on OLD version...")
output_v1 = run_cmd([OLD_PYTHON, "-c", "import tilelang.testing.perf_regression as pr; pr.regression_all()"], env=env)
print("Running regression on NEW version...")
output_v2 = run_cmd([NEW_PYTHON, "-c", "import tilelang.testing.perf_regression as pr; pr.regression_all()"], env=env)

data_v1 = parse_output(output_v1)
data_v2 = parse_output(output_v2)

common_keys = sorted(set(data_v1) & set(data_v2))
if not common_keys:
    print("No common entries between old and new versions")
    # Write empty file or message
    with open(OUT_MD, "w") as f:
        f.write("No common benchmarks found between old and new versions.\n")
    exit(0)

table = []
for key in common_keys:
    if data_v2[key] == 0:
        speedup = 0.0
    else:
        speedup = data_v1[key] / data_v2[key]
    table.append([key, data_v1[key], data_v2[key], speedup])

if not table:
    raise RuntimeError("All results are invalid")

table.sort(key=lambda x: x[-1])

headers = ["File", "Original Latency", "Current Latency", "Speedup"]

with open(OUT_MD, "w") as f:
    f.write(tabulate(table, headers=headers, tablefmt="github", stralign="left", numalign="decimal"))
    f.write("\n")

df = pd.DataFrame(table, columns=headers)
df = df.sort_values("Speedup", ascending=False).reset_index(drop=True)
draw(df)
