"""`python -m glosskernels.harness --panel tourism_monthly --series 40`"""

from __future__ import annotations

import argparse
import json

from . import panels, score, voices

DEFAULT = "seasonal_naive,walk:tabicl,pooled:tabicl"


def main() -> None:
    ap = argparse.ArgumentParser(description="Walk the voices over a monthly panel and score their bands.")
    ap.add_argument("--panel", default="tourism_monthly", help=f"{', '.join(sorted(panels.PANELS))} or synthetic")
    ap.add_argument("--series", type=int, default=40, help="a seeded sample of this many series")
    ap.add_argument("--months", type=int, default=6, help="the walk's length; the door walks six")
    ap.add_argument("--voices", default=DEFAULT, help="comma-separated; also walk:nori, pooled:nori, chronos2")
    args = ap.parse_args()

    panel = panels.synthetic() if args.panel == "synthetic" else panels.load(args.panel)
    built = voices.build([v.strip() for v in args.voices.split(",") if v.strip()])
    print(json.dumps(score.walk(panel.head(args.series), built, months=args.months), indent=1))


if __name__ == "__main__":
    main()
