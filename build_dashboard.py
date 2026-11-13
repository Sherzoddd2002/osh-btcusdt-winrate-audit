"""Build the fully static dashboard from a pipeline_v2.py report.

Usage:
    python build_dashboard.py --report E:/OSH/real_report.json --out E:/OSH/osh_winrate_site

The produced index.html has the report embedded, so it works from file://,
GitHub Pages and any static host with zero backend and zero network calls.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", default=str(HERE / "report.json"))
    ap.add_argument("--template", default=str(HERE / "dashboard_template.html"))
    ap.add_argument("--out", default=str(HERE))
    args = ap.parse_args()

    report_path = Path(args.report)
    if not report_path.exists():
        raise SystemExit(f"Report topilmadi: {report_path}")

    report = json.loads(report_path.read_text(encoding="utf-8"))
    if "summary" not in report or "cost_sensitivity" not in report:
        raise SystemExit("Report formati eski: pipeline_v2.py ni qayta ishga tushiring.")

    template = Path(args.template).read_text(encoding="utf-8")
    payload = json.dumps(report, ensure_ascii=False, separators=(",", ":"))
    if "</script" in payload.lower():
        raise SystemExit("Reportda </script> belgisi bor — xavfsizlik uchun to'xtatildi.")
    html = template.replace("__REPORT_JSON__", payload)
    if "__REPORT_JSON__" in html:
        raise SystemExit("Template ichida placeholder topilmadi.")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "index.html").write_text(html, encoding="utf-8")
    # keep the raw report next to the page for reproducibility
    if report_path.resolve() != (out / "report.json").resolve():
        (out / "report.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    s = report["summary"]
    print(f"[ok] {out / 'index.html'}")
    print(f"     {s['total_paper_trades']:,} paper trades | "
          f"win rate {s['win_rate_pct']:.2f}% @ {s['cost_bps']:.0f} bps | "
          f"{s['folds']} folds | {s['verdict']}")


if __name__ == "__main__":
    main()
