"""`python -m glosskernels.harness --panel tourism_monthly --series 40`"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import panels, score, voices

DEFAULT = "seasonal_naive,walk:tabicl,pooled:tabicl"


def main() -> None:
    ap = argparse.ArgumentParser(description="Walk the voices over a monthly panel and score their bands.")
    ap.add_argument("--panel", default="tourism_monthly", help=f"{', '.join(sorted(panels.PANELS))} or synthetic")
    ap.add_argument("--series", type=int, default=40, help="a seeded sample of this many series")
    ap.add_argument("--months", type=int, default=6, help="the walk's length; the door walks six")
    ap.add_argument("--burn", type=int, default=0, help="months walked first, unscored: the record a cal:<voice> reads")
    ap.add_argument("--project", type=int, default=0, metavar="ORIGINS", help="score twelve-month projections from this many origins instead of the walk")
    ap.add_argument("--whatif", action="store_true", help="grade the what-if read on a simulated panel instead: in and out of support, clean and confounded")
    ap.add_argument("--voices", default=DEFAULT, help="comma-separated; also walk:nori, pooled:nori, chronos2, and cal:<voice>")
    ap.add_argument("--out", type=Path, help="write the scores here (the models print to stdout)")
    args = ap.parse_args()

    if args.whatif:
        from . import whatif

        backend = voices.nori() if "nori" in args.voices else voices.tabicl(ensemble=True)
        scores = {f"confounding={c}": whatif.grade_worlds(backend, members=args.series, confounding=c) for c in (0.0, 1.0)}
        return _emit(scores, args.out)
    panel = (panels.synthetic() if args.panel == "synthetic" else panels.load(args.panel)).head(args.series)
    built = voices.build([v.strip() for v in args.voices.split(",") if v.strip()])
    if args.project:
        scores = score.project(panel, built, origins=args.project, burn=args.burn)
    else:
        scores = score.walk(panel, built, months=args.months, burn=args.burn)
    _emit(scores, args.out)


def _emit(scores: dict, out: Path | None) -> None:
    text = json.dumps(scores, indent=1)
    if out:
        out.write_text(text + "\n")
    else:
        print(text)


if __name__ == "__main__":
    main()
