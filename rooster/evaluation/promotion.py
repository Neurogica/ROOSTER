"""The rule for deciding whether a candidate is beating the baselines.

Defined once, here, because two places need it and they must agree: the ranking
script that reports a verdict after a sweep, and the runner's gate that aborts a
sweep early when no candidate is winning. If those two ever disagreed, the sweep
would stop on one criterion and be judged on another.

The rule:

    A cell counts as a WIN only if the candidate is lower by more than
    `MIN_EFFECT` in relative terms; smaller gaps count as ties and are excluded
    from both the numerator and the denominator.
    A candidate BEATS a baseline when it wins a majority of the decisive cells
    on **every** required metric, and there are at least `MIN_DECISIVE_CELLS`
    of them.
    A candidate is WINNING when it beats every baseline.

    For forecasting the required metrics are MSE and MAE -- the two the paper
    reports. Winning one while losing the other is not a result.

**Why the effect threshold exists.** The rule used to be a bare majority of
shared cells, and that has no power. With seven datasets, four wins is what a
coin gives: P(>= 4 of 7) = 0.50. DecompDict-K3-q1 duly passed at 4/7 on both
metrics, and on inspection the per-dataset gaps were -0.1%, +2.0%, -1.8%, -3.8%,
+0.1%, -15.6% and -3.2% against seed noise of 1.2-7.6% -- every one inside the
noise except Exchange, which is disqualified anyway because RepeatLast beats both
models there. A rule that promotes on that is a rule that promotes noise.

`MIN_EFFECT` is set at 2%, which is above the measured seed spread on the ETT
datasets (MSE sd 1.6-2.5% of the mean) and therefore the smallest gap this
protocol can resolve at all. It is not a significance test -- doing that properly
needs per-cell seed replication, which the search deliberately does not pay for --
but it stops the gate from counting differences the protocol cannot see.

Per-cell win rate rather than an average of the metric, because averaging MSE
across datasets with different variances is dominated by whichever dataset
happens to have the largest scale -- one easy dataset would carry a model that
loses everywhere else.

Only shared cells are compared, so a partially finished sweep still yields an
honest, if lower-confidence, verdict. Shared-cell counts travel with every
comparison so that confidence is visible rather than assumed.
"""

from collections import defaultdict

BASELINES = {
    "forecast": ("RepeatLast", "LinearForecaster", "DLinear", "DecompSSM"),
    "reconstruct": ("CopyInput", "ConvReconstructor", "PENGUIN"),
}
# MSE **and** MAE for forecasting: those are the numbers going in the paper, so
# those are the numbers a candidate has to win. A candidate must beat every
# baseline on a majority of shared cells for EVERY metric listed here -- winning
# one while losing the other is not a result.
#
# CRPS is still computed and still in the leaderboard, and the FlowSSM family does
# beat DecompSSM on it on 8/8 datasets. It is reported, not gated on.
REQUIRED_METRICS = {"forecast": ("mse", "mae"), "reconstruct": ("rate_mae_bpm", "rmse")}

# Relative gap below which a cell is a tie rather than a win. See the module
# docstring: at 7 datasets a bare majority is what chance produces, and the
# measured seed spread on the ETT datasets is 1.6-2.5% of the metric, so anything
# under 2% is not a difference this protocol can resolve.
MIN_EFFECT = 0.02

# A verdict from one or two decisive cells is not a verdict.
MIN_DECISIVE_CELLS = 3
PRIMARY_METRIC = {task: metrics[0] for task, metrics in REQUIRED_METRICS.items()}


def scores_by_model(records, task, metric=None):
    """`{model: {cell_key: value}}` over the finite values of one task's metric."""
    metric = metric or PRIMARY_METRIC[task]
    table = defaultdict(dict)
    for record in records:
        if record.task != task:
            continue
        value = record.metrics.get(metric)
        if isinstance(value, (int, float)) and value == value:  # finite, not NaN
            # Budget is part of cell identity, exactly as in RunRecord.key. Dropping
            # it here let records from different budgets collide in this dict, so
            # the surviving value was whichever was appended last and a candidate
            # could be compared against a baseline trained for a different number
            # of steps. That is how DecompDict-K3-q1 was first reported as beating
            # DecompSSM: its cells were logged at samples=1 while some of the
            # baseline's were at samples=20.
            table[record.model][(record.dataset, record.horizon, record.seed, record.budget)] = value
    return table


def head_to_head(candidate_scores, baseline_scores, min_effect=MIN_EFFECT):
    """`(wins, decisive)` for a candidate against one baseline, lower-is-better.

    A cell is decisive only when the two differ by more than `min_effect` in
    relative terms. Ties are dropped from the denominator as well as the
    numerator: counting an unresolvable cell as a loss would be as wrong as
    counting it as a win.
    """
    wins = decisive = 0
    for key in set(candidate_scores) & set(baseline_scores):
        candidate, baseline = candidate_scores[key], baseline_scores[key]
        scale = abs(baseline)
        if scale == 0 or abs(candidate - baseline) / scale <= min_effect:
            continue
        decisive += 1
        wins += candidate < baseline
    return wins, decisive


def evaluate_candidates(records, task, min_shared_cells=1):
    """Per-candidate comparison against every present baseline.

    Returns `{candidate: {"beats_all": bool, "versus": {baseline: (wins, shared)}}}`,
    plus the list of baselines actually found. A candidate with fewer than
    `min_shared_cells` against some baseline is not counted as beating it --
    silence is not a win.
    """
    metrics = REQUIRED_METRICS[task]
    tables = {metric: scores_by_model(records, task, metric) for metric in metrics}
    primary = tables[PRIMARY_METRIC[task]]
    baselines = [name for name in BASELINES[task] if name in primary]
    candidates = sorted(name for name in primary if name not in BASELINES[task])

    verdicts = {}
    for candidate in candidates:
        versus, beats_all = {}, bool(baselines)
        for baseline in baselines:
            per_metric = {}
            for metric in metrics:
                wins, shared = head_to_head(tables[metric].get(candidate, {}), tables[metric].get(baseline, {}))
                per_metric[metric] = (wins, shared)
                if shared < max(min_shared_cells, MIN_DECISIVE_CELLS) or wins * 2 <= shared:
                    beats_all = False
            versus[baseline] = per_metric
        verdicts[candidate] = {"beats_all": beats_all, "versus": versus}
    return verdicts, baselines


def winning_candidates(records, task, min_shared_cells=1):
    """Names of the candidates currently beating every baseline."""
    verdicts, _baselines = evaluate_candidates(records, task, min_shared_cells)
    return [name for name, verdict in verdicts.items() if verdict["beats_all"]]


def format_report(records, task, min_shared_cells=1):
    """Human-readable table of the current standing, for logs and for the gate."""
    verdicts, baselines = evaluate_candidates(records, task, min_shared_cells)
    if not verdicts:
        return f"no {task} candidates evaluated yet"

    metrics = REQUIRED_METRICS[task]
    lines = [f"{task}: must win {' AND '.join(metrics)} (lower is better) against {baselines}"]
    for candidate, verdict in sorted(verdicts.items()):
        marker = "WINNING" if verdict["beats_all"] else "losing"
        parts = []
        for baseline, per_metric in verdict["versus"].items():
            scores = " ".join(f"{metric} {wins}/{shared}" for metric, (wins, shared) in per_metric.items())
            parts.append(f"{baseline}[{scores}]")
        lines.append(f"  {candidate:<24}{marker:<9}" + "  ".join(parts))
    return "\n".join(lines)
