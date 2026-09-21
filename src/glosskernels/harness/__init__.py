"""The grading harness: voices walked over public monthly panels, one
step ahead, scored the way the witness plane scores a band — coverage,
quantile loss, and whether the PITs come out uniform.

A voice is anything that answers "next month's quantiles for every
series, from what was known at the origin". The walk's own graded
recipe is one voice; the same rows pooled across the panel is another;
a time-series model reading each series alone is a third. Nothing here
is served — it decides what gets served.

    uv sync --group harness
    uv run python -m glosskernels.harness --panel tourism_monthly --series 40
"""
