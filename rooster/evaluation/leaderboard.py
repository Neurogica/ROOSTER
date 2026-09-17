"""Append-only results store and the LEADERBOARD.md renderer.

Design constraints this exists to satisfy:

* Three people run experiments in parallel on one GPU. Their numbers must merge
  into one table without manual reconciliation, so every run records the git
  SHA, seed, split protocol and sample count alongside the metric.
* The store is **append-only JSONL**, one record per (model, task, dataset,
  horizon, seed) cell. Re-running a cell appends rather than overwrites; the
  renderer keeps the newest record per cell. That means a bad run can always be
  traced rather than silently vanishing, and two people appending concurrently
  cannot clobber each other.
* The rendered markdown is a *view*. Never hand-edit LEADERBOARD.md -- it is
  regenerated in full on every run.
"""

import json
import os
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone

RESULTS_DIR = "results"
RESULTS_PATH = os.path.join(RESULTS_DIR, "results.jsonl")
LEADERBOARD_PATH = "LEADERBOARD.md"

def budget_label(max_steps, n_samples, max_test_windows, selection="mse", recipe=None, epochs=None):
    """Canonical string for a training/evaluation configuration.

    `n_samples` is the **requested** ensemble size, not the per-model effective
    one. Deterministic models are evaluated with a single sample regardless, so
    keying on the effective count would put baselines and generative candidates
    in different groups and make them incomparable -- which is the opposite of
    what this label is for. The effective count per cell is in
    `metrics["n_samples"]`.
    """
    # A named training recipe (e.g. the DecompSSM paper's TSLib schedule) is
    # epoch-based, so its step count depends on the dataset; the label carries
    # the recipe and epoch count instead, which is what identifies the cell.
    if recipe:
        label = f"recipe={recipe}/epochs={epochs}/samples={n_samples}/testwin={max_test_windows or 'all'}"
    else:
        label = f"steps={max_steps}/samples={n_samples}/testwin={max_test_windows or 'all'}"
    if selection != "mse":
        label += f"/sel={selection}"
    return label


# Columns rendered per task family, in order. Keys not present in a record render
# as "--" rather than breaking the table, so adding a metric never invalidates
# historical records.
FORECAST_COLUMNS = (
    ("mse", "MSE", 4),
    ("mae", "MAE", 4),
    ("crps", "CRPS", 4),
    ("crps_sum", "CRPS-sum", 4),
    ("coverage_80", "cov@80", 3),
    ("n_samples", "N", 0),
    ("train_seconds", "train s", 1),
)
RECONSTRUCTION_COLUMNS = (
    ("rate_mae_bpm", "rate MAE (bpm)", 2),
    ("rmse", "RMSE", 4),
    ("mae", "MAE", 4),
    ("pearson_r", "r", 3),
    ("rate_unresolved_frac", "unres.", 3),
    ("n_samples", "N", 0),
    ("train_seconds", "train s", 1),
)


def git_sha():
    """Short SHA of the working tree, with a `-dirty` marker when uncommitted.

    Recorded per run because a leaderboard row whose code you cannot recover is
    not evidence of anything.
    """
    try:
        sha = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain"], text=True, stderr=subprocess.DEVNULL).strip()
        return f"{sha}-dirty" if dirty else sha
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


@dataclass
class RunRecord:
    """One evaluated cell of the leaderboard.

    `budget` records the training and evaluation configuration ("steps=5000/
    samples=8/testwin=1024") and is **part of the cell identity**. Without it, a
    5 000-step run and a 20 000-step run of the same model produce records that
    are indistinguishable, and whichever finished last silently wins -- which is
    exactly what happened when two sweeps overlapped on one machine. Two budgets
    now coexist as two rows instead of one row of unknown provenance.
    """

    task: str  # "forecast" | "reconstruct"
    dataset: str
    model: str
    horizon: int  # forecast horizon, or window length in samples for reconstruction
    seed: int
    metrics: dict
    lookback: int = 0
    protocol: str = ""
    budget: str = ""
    git_sha: str = field(default_factory=git_sha)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"))

    @property
    def key(self):
        """Identity of the cell this record fills, budget included."""
        return (self.task, self.dataset, self.model, self.horizon, self.seed, self.budget)


def append_records(records, path=RESULTS_PATH):
    """Append records to the JSONL store, creating it if needed."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(asdict(record), sort_keys=True) + "\n")


def load_records(path=RESULTS_PATH):
    """Read every record. Malformed lines are skipped loudly rather than fatally."""
    if not os.path.exists(path):
        return []
    records = []
    with open(path, encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(RunRecord(**json.loads(line)))
            except (json.JSONDecodeError, TypeError) as error:
                print(f"  [leaderboard] skipping malformed record at {path}:{line_number}: {error}")
    return records


def latest_per_cell(records):
    """Keep only the newest record for each cell, so re-runs supersede."""
    newest = {}
    for record in records:
        current = newest.get(record.key)
        if current is None or record.timestamp >= current.timestamp:
            newest[record.key] = record
    return list(newest.values())


def _format(value, decimals):
    if value is None:
        return "--"
    if isinstance(value, float) and value != value:  # NaN
        return "n/a"
    if decimals == 0:
        return f"{int(value)}"
    return f"{value:.{decimals}f}"


def _render_table(records, columns, horizon_label):
    """One markdown table: rows are (model, dataset), columns are metrics."""
    header = ["Model", "Dataset", horizon_label, "Seed", *(label for _, label, _ in columns), "Budget", "Commit"]
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]

    # Sort so a diff of LEADERBOARD.md between two runs is readable.
    for record in sorted(records, key=lambda r: (r.dataset, r.model, r.horizon, r.seed, r.budget)):
        cells = [record.model, record.dataset, str(record.horizon), str(record.seed)]
        cells += [_format(record.metrics.get(key), decimals) for key, _, decimals in columns]
        cells.append(record.budget or "--")
        cells.append(f"`{record.git_sha}`")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _render_averages(records, columns):
    """Per-model mean over all datasets and horizons, for an at-a-glance ranking.

    Averaging MSE across datasets with different variances is not a meaningful
    quantity in itself -- it is a summary for spotting regressions, and the
    caption says so.
    """
    by_model = {}
    for record in records:
        by_model.setdefault(record.model, []).append(record)

    header = ["Model", "Cells", *(label for _, label, _ in columns)]
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    for model in sorted(by_model):
        model_records = by_model[model]
        cells = [model, str(len(model_records))]
        for key, _, decimals in columns:
            values = [r.metrics[key] for r in model_records if isinstance(r.metrics.get(key), (int, float)) and r.metrics[key] == r.metrics[key]]
            cells.append(_format(sum(values) / len(values), decimals) if values else "--")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def render(records, path=LEADERBOARD_PATH):
    """Regenerate LEADERBOARD.md in full from the store."""
    records = latest_per_cell(records)
    forecast = [r for r in records if r.task == "forecast"]
    reconstruct = [r for r in records if r.task == "reconstruct"]
    generated = datetime.now(timezone.utc).isoformat(timespec="seconds")

    sections = [
        "# Leaderboard",
        "",
        f"Generated {generated} by `scripts/run_leaderboard.py`. **Do not hand-edit** --",
        "this file is regenerated in full from `results/results.jsonl` on every run.",
        "",
        "Protocol is fixed in [docs/02_benchmark_protocol.md](docs/02_benchmark_protocol.md):",
        "chronological splits, scaler fit on train only, MSE/MAE in scaled space for",
        "forecasting, subject-wise splits for reconstruction. Numbers produced any other",
        "way do not belong here.",
        "",
    ]

    if forecast:
        sections += [
            "## Forecasting (DecompSSM-side task)",
            "",
            "All available LTSF datasets, every horizon. MSE/MAE are in **scaled space**,",
            "the LTSF convention -- see the protocol doc before comparing against published",
            "numbers. `N` is the ensemble size behind CRPS; `N=1` means the model is",
            "deterministic and its CRPS degenerates to MAE.",
            "",
            _render_table(forecast, FORECAST_COLUMNS, "Horizon"),
            "",
            "### Mean over all forecasting cells",
            "",
            "A regression detector, not a scientific quantity: averaging MSE across datasets",
            "with different variances is not meaningful on its own.",
            "",
            _render_averages(forecast, FORECAST_COLUMNS),
            "",
        ]

    if reconstruct:
        sections += [
            "## PPG reconstruction (PENGUIN-side task, light datasets only)",
            "",
            "BIDMC and CapnoBase only -- both are ~40-50 recordings of 8 minutes, so a full",
            "cycle is cheap enough to run on every refresh. PPG-DaLiA, WESAD, WildPPG and",
            "the cuff-less BP set are deliberately excluded from the per-commit loop.",
            "",
            "**`rate MAE (bpm)` is the headline number** -- heart rate from R-peak",
            "intervals for `-ECG` tasks, respiratory rate from the dominant spectral",
            "component for `-RESP` tasks. Waveform RMSE/MAE/r are secondary. `unres.` is",
            "the fraction of windows where no rate could be resolved.",
            "",
            "> **Read the `CopyInput` row before reading anything else.** `CopyInput`",
            "> returns the PPG window unchanged. On the `-ECG` tasks it scores ~1.5 bpm,",
            "> because PPG already carries the cardiac rhythm -- so on these clean clinical",
            "> recordings, HR error is close to saturated by a model that does nothing. A",
            "> reconstruction model that improves waveform correlation while *losing* to",
            "> `CopyInput` on rate has learned the morphology and broken the beat timing.",
            "> Treat HR error on BIDMC/CapnoBase as a sanity check, not as the claim; the",
            "> `-RESP` tasks, where `CopyInput` is genuinely weak, carry more signal.",
            "",
            "These numbers are **not comparable to PENGUIN's or RespDiff's published",
            "tables** -- different splits, window lengths and rate estimators. Comparability",
            "requires re-running those models in this harness.",
            "",
            _render_table(reconstruct, RECONSTRUCTION_COLUMNS, "Window"),
            "",
        ]

    if not forecast and not reconstruct:
        sections += ["_No results yet. Run `python scripts/run_leaderboard.py`._", ""]

    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(sections))
    return path
