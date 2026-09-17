"""Vendored upstream implementations of this team's two prior papers.

Both are copied **verbatim** from the authors' public repositories so that the
leaderboard compares against the real models rather than our reading of the
papers. That distinction turned out to matter: a from-the-paper reimplementation
got the branch step-size ordering backwards, summed the components instead of
concatenating them, and missed PENGUIN's dual-stream update entirely.

* DecompSSM -- https://github.com/Neurogica/DecompSSM (`models/DecompSSM.py`)
* PENGUIN   -- https://github.com/Neurogica/PENGUIN (`src/models/PENGUIN.py`
  plus its vendored S5 layer)

Both are Clear BSD licensed; see LICENSE.neurogica. Every local change is marked
`# VENDOR EDIT` and is an import path or an optional-dependency guard, nothing
else. Adapters that fit these into the benchmark interface live in
`models/paper_baselines.py`, so this package stays a clean mirror.
"""
