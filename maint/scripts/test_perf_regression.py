import subprocess
import re
import os
from tabulate import tabulate
import pandas as pd

try:
    import tilelang

    tilelang.disable_cache()
except Exception:
    tilelang = None

OLD_PYTHON = os.environ.get("OLD_PYTHON", "./old/bin/python")
NEW_PYTHON = os.environ.get("NEW_PYTHON", "./new/bin/python")
OUT_MD = os.environ.get("PERF_REGRESSION_MD", "regression_result.md")
OUT_PNG = os.environ.get("PERF_REGRESSION_PNG", "regression_result.png")

def parse_output(output):
    data = {}
    for line in output.split("\n"):
        line = line.strip()
        m = re.match(r"\|\s*([^\|]+)\s*\|\s*([0-9\.]+)\s*\|", line)
        if m is not None:
            data[m.group(1)] = float(m.group(2))
    return data


def run_cmd(cmd):
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\nSTDOUT:\n{p.stdout}\nSTDERR:\n{p.stderr}")
    return p.stdout


def draw(df):
    import matplotlib.pyplot as plt
    import seaborn as sns

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


output_v1 = run_cmd([OLD_PYTHON, "-c", "import tilelang.testing.perf_regression as pr; pr.regression_all()"])
output_v2 = run_cmd([NEW_PYTHON, "-c", "import tilelang.testing.perf_regression as pr; pr.regression_all()"])

data_v1 = parse_output(output_v1)
data_v2 = parse_output(output_v2)

common_keys = sorted(set(data_v1) & set(data_v2))
if not common_keys:
    raise RuntimeError("No common entries between old and new versions")

table = []
for key in data_v1.keys():
    speedup = data_v1[key] / data_v2[key]
    table.append([key, data_v1[key], data_v2[key], speedup])

if not table:
    raise RuntimeError("All results are invalid (<= 0)")

table.sort(key=lambda x: x[-1])

headers = ["File", "Original Latency", "Current Latency", "Speedup"]

with open(OUT_MD, "w") as f:
    f.write(tabulate(table, headers=headers, tablefmt="github", stralign="left", numalign="decimal"))
    f.write("\n")

df = pd.DataFrame(table, columns=headers)
df = df.sort_values("Speedup", ascending=False).reset_index(drop=True)
draw(df)
