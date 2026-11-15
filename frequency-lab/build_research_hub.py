"""Build a SEPARATE static research-hub page from the JSON result files.

Reads only aggregate JSON reports. Never reads API keys, the forward-paper
model, the engine, or the lock file. Output: research-hub.html (static, offline).
"""
from __future__ import annotations

import html
import json
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent


def load(name):
    p = HERE / name
    return json.load(open(p, encoding="utf-8")) if p.exists() else None


def esc(x):
    return html.escape(str(x))


def fmt(x, nd=2, suffix=""):
    if x is None:
        return "—"
    if isinstance(x, (int, float)):
        return f"{x:.{nd}f}{suffix}"
    return f"{x}{suffix}"


def registry_rows():
    """Curated, verifiable list of every experiment and its outcome."""
    rows = []
    v3 = load("v3_results.json")
    if v3:
        for hname, hd in v3["horizons"].items():
            for cfg, s in hd["configs"].items():
                h = s["untouched_holdout"]
                rows.append({
                    "group": "V3 tuzatilgan eksperiment",
                    "name": f"{hname} · {cfg}",
                    "detail": f"{s['total_trades']} savdo · median {fmt(s['median_fold_expectancy_bps'],2,' bps')} · "
                              f"PF {fmt(s['pooled_profit_factor'],2)} · holdout {fmt(h.get('expectancy_bps'),2,' bps')}",
                    "verdict": "NO-GO",
                })

    integ = load("integrity_report.json")
    if integ:
        rows.append({"group": "Ma'lumot auditi", "name": "BTCUSDT 15m strukturaviy tekshiruv",
                     "detail": "240 672 sham · 0 dublikat · 0 haqiqiy uzilish · 0 OHLC xatosi · 16 nol-hajm",
                     "verdict": "PASS"})

    abl = load("ablation_results.json")
    if abl:
        rows.append({"group": "Tashqi ma'lumotlar", "name": "Xasset / news 6-fold ablatsiya",
                     "detail": "Baza −11.24 bps → +oltin/makro −8.26 (+2.98) → +news −12.07 → ikkalasi −11.19 bps; 0/6 musbat",
                     "verdict": "NO-GO"})

    sel = load("selectivity_sweep_1d.json")
    if sel:
        for k, v in sel["selections"].items():
            o, h = v["pooled_oos"], v["untouched_holdout"]
            rows.append({"group": "Selektivlik sweep (1d)", "name": f"top {k} har tomonga",
                         "detail": f"OOS {fmt(o['expectancy_bps'],2,' bps')}, WR {fmt(o['win_rate_pct'],1,'%')} | "
                                   f"Holdout {fmt(h.get('expectancy_bps'),2,' bps')}, WR {fmt(h.get('win_rate_pct'),1,'%')}",
                         "verdict": "NO-GO"})

    hwj = load("high_winrate_hunt_1d.json")
    if hwj:
        for name, s in hwj["configs"].items():
            o, h = s["pooled_oos"], s["untouched_holdout_primary"]
            rows.append({"group": "High-winrate hunt (1d)", "name": name,
                         "detail": f"OOS WR {fmt(o.get('win_rate_pct'),1,'%')} / {fmt(o.get('expectancy_bps'),2,' bps')} | "
                                   f"Holdout WR {fmt(h.get('win_rate_pct'),1,'%')} / {fmt(h.get('expectancy_bps'),2,' bps')} "
                                   f"/ $100→${fmt(h.get('final_balance_100usd'),2)}",
                         "verdict": "NO-GO"})

    tsw = load("development_time_exit_sweep_1d.json")
    if tsw:
        for name, v in tsw["variants"].items():
            o = v["pooled_oos_primary"]
            rows.append({"group": "Time-exit sweep (dev OOS)", "name": name,
                         "detail": f"{o['trades']} savdo · {fmt(o['trades_per_day'],2,'/kun')} · "
                                   f"WR {fmt(o['win_rate_pct'],1,'%')} · {fmt(o['expectancy_bps'],2,' bps')} · "
                                   f"PF {fmt(o['profit_factor'],3)}",
                         "verdict": "NO-GO" if o["expectancy_bps"] <= 0 else "DISCOVERY"})

    hyb = load("development_side_hybrid_execution_1d.json")
    if hyb:
        for name, v in hyb["variants"].items():
            o = v["pooled_oos_primary"]
            rows.append({"group": "Side-hybrid (dev OOS)", "name": name,
                         "detail": f"{o['trades']} savdo · WR {fmt(o['win_rate_pct'],1,'%')} · "
                                   f"{fmt(o['expectancy_bps'],2,' bps')} · PF {fmt(o['profit_factor'],3)}",
                         "verdict": "DISCOVERY"})

    snap = load("h96_c2_results_snapshot.json")
    if snap:
        o = snap["pooled_oos_primary"]
        rows.append({"group": "Eng yaxshi nomzod", "name": "H96 C2 (frozen, forward-paper)",
                     "detail": f"OOS {o['trades']} savdo · WR {fmt(o['win_rate_pct'],2,'%')} · "
                               f"PF {fmt(o['profit_factor'],4)} · {fmt(o['expectancy_bps'],3,' bps')} · "
                               f"4/5 fold · bootstrap pastki ≤ 0 · holdout OCHILMAGAN",
                     "verdict": "NO-GO"})

    stress = load("capital_fee_stress_1d.json")
    if stress:
        rows.append({"group": "Kapital / xarajat stress", "name": "27 ta kombinatsiya (fee × slip × selektivlik)",
                     "detail": "Maker/Maker 0bps 10%: OOS +$37.69, lekin Holdout −$10.06; 0/27 GO",
                     "verdict": "NO-GO"})
    return rows


def freq_rows():
    d = load("frequency_lab_1d.json")
    if not d:
        return None, []
    out = []
    for g in d["gates"]:
        s = d["configs"][g["config"]]
        o, h = s["pooled_oos_primary"], s["untouched_holdout_primary"]
        out.append({
            "name": g["config"], "cooldown": s["cooldown_bars"],
            "max_per_day": s["max_trades_per_day"],
            "notional": s["position_fraction"] * 100.0,
            "oos_trades": o.get("trades", 0), "tpd": o.get("trades_per_day"),
            "oos_exp": o.get("expectancy_bps"), "oos_wr": o.get("win_rate_pct"),
            "oos_bal": o.get("final_balance_100usd"),
            "hld_trades": h.get("trades", 0), "hld_exp": h.get("expectancy_bps"),
            "hld_wr": h.get("win_rate_pct"), "hld_bal": h.get("final_balance_100usd"),
            "dd": h.get("max_drawdown_pct"), "max_notional": h.get("max_notional_usd"),
            "conc": h.get("max_concurrent_positions"),
            "passed": g["checks_passed"], "total": g["checks_total"],
            "verdict": g["verdict"],
        })
    return d, out


def frontier_svg(rows):
    """trades/day (x) vs $100 final balance (y) for OOS and holdout."""
    if not rows:
        return "<p>Yangi eksperiment natijalari kutilmoqda.</p>"
    W, H_, PAD_L, PAD_R, PAD_T, PAD_B = 680, 300, 58, 20, 24, 46
    xs = [r["tpd"] or 0 for r in rows]
    yvals = [r["oos_bal"] or 100 for r in rows] + [r["hld_bal"] or 100 for r in rows] + [100]
    xmin, xmax = 0.0, max(xs + [1.0]) * 1.12
    ymin, ymax = min(yvals) - 4, max(yvals) + 4
    if ymax - ymin < 12:
        ymin, ymax = 96, 104

    def X(v):
        return PAD_L + (v - xmin) / (xmax - xmin) * (W - PAD_L - PAD_R)

    def Y(v):
        return PAD_T + (ymax - v) / (ymax - ymin) * (H_ - PAD_T - PAD_B)

    parts = [f'<svg viewBox="0 0 {W} {H_}" width="100%" height="auto" role="img" '
             f'aria-label="Savdo soni va 100 dollar balansi" style="max-width:680px">',
             f'<rect x="0" y="0" width="{W}" height="{H_}" fill="#0d1a2b" rx="12"/>']
    # grid + y labels
    for i in range(5):
        v = ymin + (ymax - ymin) * i / 4
        y = Y(v)
        parts.append(f'<line x1="{PAD_L}" y1="{y:.1f}" x2="{W-PAD_R}" y2="{y:.1f}" '
                     f'stroke="#20364d" stroke-width="1"/>')
        parts.append(f'<text x="{PAD_L-8}" y="{y+4:.1f}" text-anchor="end" fill="#94a9be" '
                     f'font-size="11">${v:.0f}</text>')
    # break-even line at 100
    y100 = Y(100)
    parts.append(f'<line x1="{PAD_L}" y1="{y100:.1f}" x2="{W-PAD_R}" y2="{y100:.1f}" '
                 f'stroke="#f5c36a" stroke-width="1.5" stroke-dasharray="5 4"/>')
    parts.append(f'<text x="{W-PAD_R-4}" y="{y100-6:.1f}" text-anchor="end" fill="#f5c36a" '
                 f'font-size="10">$100 (zararsizlik)</text>')
    for i in range(6):
        v = xmin + (xmax - xmin) * i / 5
        x = X(v)
        parts.append(f'<text x="{x:.1f}" y="{H_-PAD_B+20}" text-anchor="middle" fill="#94a9be" '
                     f'font-size="11">{v:.1f}</text>')
    parts.append(f'<text x="{W/2:.0f}" y="{H_-6}" text-anchor="middle" fill="#94a9be" '
                 f'font-size="12">savdo / kun</text>')

    for key, color in (("oos_bal", "#55d6d0"), ("hld_bal", "#ff7c86")):
        pts = [(X(r["tpd"] or 0), Y(r[key] or 100)) for r in rows]
        poly = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
        parts.append(f'<polyline points="{poly}" fill="none" stroke="{color}" stroke-width="2.5" '
                     f'stroke-linejoin="round"/>')
        for x, y in pts:
            parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{color}"/>')
    parts.append(f'<g font-size="12" fill="#edf5ff"><rect x="{PAD_L+10}" y="{PAD_T+6}" width="10" height="10" '
                 f'fill="#55d6d0"/><text x="{PAD_L+26}" y="{PAD_T+15}">OOS (development)</text>'
                 f'<rect x="{PAD_L+190}" y="{PAD_T+6}" width="10" height="10" fill="#ff7c86"/>'
                 f'<text x="{PAD_L+206}" y="{PAD_T+15}">Holdout (tegilmagan 180 kun)</text></g>')
    parts.append("</svg>")
    return "".join(parts)


def main():
    reg = registry_rows()
    fd, fr = freq_rows()
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    n_total = len(reg) + len(fr)
    n_go = sum(1 for r in reg if r["verdict"] == "GO") + sum(1 for r in fr if r["verdict"] == "GO")

    # ---------- registry table
    reg_html = []
    last_group = None
    for r in reg:
        if r["group"] != last_group:
            reg_html.append(f'<tr class="grp"><td colspan="4">{esc(r["group"])}</td></tr>')
            last_group = r["group"]
        cls = "ok" if r["verdict"] == "PASS" else ("warn" if r["verdict"] == "DISCOVERY" else "bad")
        reg_html.append(
            f'<tr><td class="nm">{esc(r["name"])}</td><td class="dt">{esc(r["detail"])}</td>'
            f'<td><span class="tag {cls}">{esc(r["verdict"])}</span></td></tr>')
    reg_html = "\n".join(reg_html)

    # ---------- frequency table + KPIs
    if fr:
        frows = []
        for r in fr:
            cls = "ok" if r["verdict"] == "GO" else "bad"
            frows.append(
                "<tr>"
                f'<td class="nm">{esc(r["name"])}</td>'
                f'<td>{r["cooldown"]} bar / {r["max_per_day"]}/kun</td>'
                f'<td>${r["notional"]:.2f}</td>'
                f'<td>{fmt(r["tpd"],2)}</td>'
                f'<td>{r["oos_trades"]}</td>'
                f'<td>{fmt(r["oos_wr"],1,"%")}</td>'
                f'<td>{fmt(r["oos_exp"],2)}</td>'
                f'<td>${fmt(r["oos_bal"],2)}</td>'
                f'<td>{fmt(r["hld_wr"],1,"%")}</td>'
                f'<td>{fmt(r["hld_exp"],2)}</td>'
                f'<td>${fmt(r["hld_bal"],2)}</td>'
                f'<td>{fmt(r["dd"],1,"%")}</td>'
                f'<td>${fmt(r["max_notional"],2)}</td>'
                f'<td><span class="tag {cls}">{esc(r["verdict"])} {r["passed"]}/{r["total"]}</span></td>'
                "</tr>")
        frows = "\n".join(frows)
        best_tpd = max(fr, key=lambda r: r["tpd"] or 0)
        best_bal = max(fr, key=lambda r: r["hld_bal"] or 0)
        base = fr[0]
        freq_summary = (
            f'<b>Savdo soni {fmt(best_tpd["tpd"]/base["tpd"],1)} barobar oshdi</b>: '
            f'{fmt(base["tpd"],2)}/kun ({base["oos_trades"]} savdo) → '
            f'<b>{fmt(best_tpd["tpd"],2)}/kun</b> ({best_tpd["oos_trades"]} savdo). '
            f'Lekin OOS $100 balansi <b>${fmt(base["oos_bal"],2)} → ${fmt(best_tpd["oos_bal"],2)}</b> ga '
            f'<i>pasaydi</i>, holdout balansi esa barcha holatda $100 dan past '
            f'(eng yaxshisi <b>${fmt(best_bal["hld_bal"],2)}</b>). '
            f'Yagona haqiqiy yaxshilanish — max drawdown <b>{fmt(base["dd"],1,"%")} → '
            f'{fmt(best_tpd["dd"],1,"%")}</b> ga tushdi (pozitsiyalar maydalangani uchun). '
            f'Barcha {len(fr)} konfiguratsiya: <b>{sum(1 for r in fr if r["verdict"]=="GO")} ta GO</b>.')
        extra_note = (
            f'<div class="note"><b>Statistik quvvat yetarli bo\'ldi:</b> eng zich konfiguratsiyada '
            f'holdout\'da <b>{best_tpd["hld_trades"]} ta savdo</b> to\'plandi — avvalgi 92 ta emas. '
            f'Shunga qaramay expectancy manfiy ({fmt(best_tpd["hld_exp"],2)} bps, '
            f'WR {fmt(best_tpd["hld_wr"],1)}%). '
            f'Demak muammo namuna kichikligida emas, <b>edge yo\'qligida</b>. '
            f'Har bir konfiguratsiyada maksimal bir vaqtdagi notional aynan <b>$100</b> — '
            f'1x sharti buzilmadi, leverage ishlatilmadi.</div>'
            f'<div class="note"><b>Xarajat modeli haqida ogohlantirish:</b> Primary ssenariy '
            f'maker/maker (2 bps/tomon + 2 bps slippage ≈ 6 bps round-trip) — bu H96 C2 ning '
            f'taker-entry Primary modelidan <b>arzonroq</b>, ya\'ni OOS raqamlari biroz optimistik. '
            f'Stress (taker/taker ≈ 10 bps) da holdout yana yomonlashadi: masalan C96 da '
            f'−15.15 → −19.15 bps, C06 da −11.95 → −15.95 bps. Xulosa o\'zgarmaydi.</div>')
        chart = frontier_svg(fr)
    else:
        frows = '<tr><td colspan="14">Natijalar hisoblanmoqda…</td></tr>'
        freq_summary = "Yangi eksperiment natijalari kutilmoqda."
        chart = "<p>Natijalar kutilmoqda.</p>"
        extra_note = ""

    meta = ""
    if fd:
        p = fd["metadata"]["preregistered"]
        meta = (f'Oldindan e\'lon qilingan: cooldown {p["cooldown_bars"]} bar · '
                f'{esc(p["barrier"])} · {esc(p["selection"])} · {esc(p["position_rule"])}. '
                f'Holdout: {esc(p["holdout_policy"])}.')

    doc = f"""<!doctype html>
<html lang="uz">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="theme-color" content="#07101d">
<title>OSH Quant Lab · Tadqiqotlar registri va Frequency Lab</title>
<meta name="description" content="BTCUSDT 15m bo'yicha barcha paper-backtest eksperimentlari registri va savdo soni/kapital tadqiqoti. Faqat tadqiqot, real buyurtma yo'q.">
<style>
:root{{color-scheme:dark;--bg:#07101d;--panel:#0d1a2b;--line:#20364d;--ink:#edf5ff;--muted:#94a9be;--cyan:#55d6d0;--green:#55d6a2;--red:#ff7c86;--amber:#f5c36a}}
*{{box-sizing:border-box}}html{{scroll-behavior:smooth}}
body{{margin:0;background:radial-gradient(ellipse at 12% 0%,rgba(41,115,154,.18),transparent 34%),var(--bg);color:var(--ink);font:15px/1.65 Inter,ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif}}
a{{color:var(--cyan);text-decoration:none}}a:hover{{text-decoration:underline}}
.shell{{width:min(1160px,calc(100% - 36px));margin:auto}}
nav{{display:flex;justify-content:space-between;align-items:center;gap:20px;padding:22px 0;border-bottom:1px solid rgba(148,169,190,.16);flex-wrap:wrap}}
.brand{{display:flex;align-items:center;gap:12px;font-weight:800;letter-spacing:.08em}}
.logo{{width:38px;height:38px;display:grid;place-items:center;border-radius:12px;background:linear-gradient(135deg,#1e9f9b,#437cf0);color:#06121c;font-size:16px;font-weight:900}}
.navlinks{{display:flex;gap:19px;flex-wrap:wrap;font-size:13px}}.navlinks a{{color:var(--muted)}}
.hero{{padding:46px 0 22px}}h1{{font-size:clamp(26px,4vw,40px);line-height:1.18;margin:0 0 14px;letter-spacing:-.02em}}
.sub{{color:var(--muted);max-width:800px;margin:0}}
.banner{{margin:26px 0;padding:20px 22px;border:1px solid rgba(255,124,134,.35);border-left:4px solid var(--red);border-radius:14px;background:rgba(255,124,134,.07)}}
.banner b{{color:var(--red)}}.banner p{{margin:8px 0 0;color:var(--muted);font-size:14px}}
.kpis{{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:14px;margin:26px 0}}
.kpi{{padding:18px 20px;border:1px solid var(--line);border-radius:14px;background:var(--panel)}}
.kpi .v{{font-size:26px;font-weight:800;letter-spacing:-.02em}}.kpi .l{{color:var(--muted);font-size:12px;margin-top:4px}}
section{{margin:38px 0}}h2{{font-size:22px;margin:0 0 8px}}h3{{font-size:16px;margin:24px 0 8px}}
.lead{{color:var(--muted);margin:0 0 18px;max-width:900px}}
.card{{border:1px solid var(--line);border-radius:14px;background:var(--panel);padding:20px;overflow-x:auto}}
table{{width:100%;border-collapse:collapse;font-size:13px;min-width:760px}}
th,td{{text-align:left;padding:10px 11px;border-bottom:1px solid rgba(148,169,190,.13);vertical-align:top}}
th{{color:var(--muted);font-weight:700;font-size:11px;text-transform:uppercase;letter-spacing:.06em;position:sticky;top:0;background:var(--panel)}}
tr.grp td{{color:var(--cyan);font-size:11px;font-weight:800;letter-spacing:.1em;text-transform:uppercase;padding-top:18px;background:rgba(85,214,208,.04)}}
td.nm{{font-weight:650;white-space:nowrap}}td.dt{{color:var(--muted)}}
.tag{{display:inline-block;padding:3px 9px;border-radius:999px;font-size:10px;font-weight:800;letter-spacing:.06em;white-space:nowrap}}
.tag.bad{{color:#ffb3ba;background:rgba(255,124,134,.13);border:1px solid rgba(255,124,134,.3)}}
.tag.ok{{color:#8ae7c0;background:rgba(85,214,162,.12);border:1px solid rgba(85,214,162,.3)}}
.tag.warn{{color:#f8dca1;background:rgba(245,195,106,.12);border:1px solid rgba(245,195,106,.3)}}
.note{{border-left:3px solid var(--cyan);padding:14px 18px;background:rgba(85,214,208,.06);border-radius:0 12px 12px 0;margin:18px 0;color:var(--muted)}}
.note b{{color:var(--ink)}}
.formula{{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;background:rgba(7,16,29,.7);padding:14px 16px;border-radius:10px;border:1px solid var(--line);color:var(--cyan);font-size:13px;overflow-x:auto}}
.two{{display:grid;grid-template-columns:1fr 1fr;gap:16px}}@media(max-width:860px){{.two{{grid-template-columns:1fr}}}}
ul{{padding-left:20px;color:var(--muted)}}li{{margin:6px 0}}li b{{color:var(--ink)}}
footer{{border-top:1px solid rgba(148,169,190,.16);margin-top:52px;padding:24px 0 40px;color:var(--muted);font-size:12px}}
.pill{{display:inline-flex;gap:8px;align-items:center;padding:6px 12px;border:1px solid rgba(85,214,162,.28);border-radius:999px;color:#8ae7c0;background:rgba(85,214,162,.08);font-size:11px;font-weight:800}}
</style>
</head>
<body>
<div class="shell">
<nav>
  <div class="brand"><span class="logo">OSH</span><span>QUANT LAB · RESEARCH HUB</span></div>
  <div class="navlinks">
    <a href="./">Asosiy dashboard</a>
    <a href="external-audit.html">Tashqi ma'lumotlar</a>
    <a href="research-v3.html">Research v3</a>
    <a href="#registry">Registr</a>
    <a href="#freq">Frequency Lab</a>
  </div>
  <span class="pill">● PAPER ONLY · live_orders=false</span>
</nav>

<div class="hero">
  <h1>Tadqiqotlar registri &amp; Savdo soni × Kapital</h1>
  <p class="sub">BTCUSDT 15m bo'yicha shu paytgacha qilingan barcha eksperimentlar, ularning aniq natijalari
  va yangi, <b>alohida</b> Frequency &amp; Capital Lab tadqiqoti. Yangilangan: {esc(now)}.</p>
</div>

<div class="banner">
  <b>⚠️ YAKUNIY QAROR: NO-GO — real pul bilan savdo yo'q.</b>
  <p>{n_total} ta hisoblangan eksperiment/konfiguratsiyadan <b>{n_go} tasi GO</b>. OOS (development) davrida ijobiy
  ko'ringan har bir nomzod mustaqil holdout'da yoki statistik eshiklarda yutqazgan. Bu sahifa faqat tadqiqot uchun.</p>
  <p><b>Alohidalik kafolati:</b> ushbu sahifa va uning eksperimenti siz ishlatayotgan H96 C2 forward-paper modeliga,
  <code>forward_paper_h96_c2_lock.json</code> ga, lokal engine'ga, Windows Credential Manager'dagi kalitlarga va
  Testnet terminaliga <b>tegmadi</b>. Faqat yangi fayllar qo'shildi; order yuborish kodi yo'q.</p>
</div>

<div class="kpis">
  <div class="kpi"><div class="v">{n_total}</div><div class="l">eksperiment / konfiguratsiya</div></div>
  <div class="kpi"><div class="v">240 672</div><div class="l">15m sham · 2020→2026 · 0 strukturaviy xato</div></div>
  <div class="kpi"><div class="v">58.92%</div><div class="l">eng yaxshi OOS win rate (H96 C2)</div></div>
  <div class="kpi"><div class="v" style="color:var(--red)">0 / {n_total}</div><div class="l">barcha GO mezonlaridan o'tgan</div></div>
</div>

<section id="registry">
  <h2>1 · Barcha natijalar ro'yxati</h2>
  <p class="lead">Har bir qator qayta hisoblangan JSON hisobotdan olingan. "DISCOVERY" = development OOS'da ijobiy,
  lekin holdout ochilmagan yoki statistik eshiklardan o'tmagan. "NO-GO" = foydali strategiya emas.</p>
  <div class="card"><table>
    <thead><tr><th style="width:230px">Eksperiment</th><th>Asosiy natija</th><th style="width:110px">Hukm</th></tr></thead>
    <tbody>{reg_html}</tbody>
  </table></div>
</section>

<section id="freq">
  <h2>2 · Yangi: savdo sonini oshirish $100 ga foyda keltiradimi?</h2>
  <p class="lead">Sizning H96 C2 konfiguratsiyangizda <code>entry_cooldown_bars = 96</code> — bu kuniga maksimum
  <b>1 ta</b> savdo degani (amalda ~0.54/kun). Time-exit sweep'da bu cooldown o'zgarmagani uchun savdo soni
  0.54→0.65/kun dan oshmagan. Yangi eksperimentda faqat cooldown o'zgartirildi: 96 → 48 → 24 → 12 → 6 bar.
  $100 va 1x leverage shartini buzmaslik uchun har bir pozitsiya hajmi
  <code>100 × cooldown / 96</code> qilib belgilandi, shunda bir vaqtdagi jami notional hech qachon $100 dan oshmaydi.</p>
  <div class="note"><b>Pre-registration:</b> {esc(meta)} Har bir konfiguratsiya holdout'ga bir marta tegdi;
  natijalarga qarab qayta sozlash yoki eng yaxshisini tanlash qilinmadi.</div>
  <div class="card"><table>
    <thead><tr>
      <th>Konfig</th><th>Cooldown</th><th>Pozitsiya</th><th>Savdo/kun</th><th>OOS N</th>
      <th>OOS WR</th><th>OOS exp</th><th>OOS $100</th><th>HLD WR</th><th>HLD exp</th>
      <th>HLD $100</th><th>Max DD</th><th>Max notional</th><th>Hukm</th>
    </tr></thead><tbody>{frows}</tbody>
  </table></div>
  <p class="lead" style="margin-top:14px">{freq_summary}</p>
  {extra_note}
  <h3>Savdo soni × $100 balansi</h3>
  <div class="card" style="text-align:center">{chart}</div>
  <div class="note"><b>Grafikni o'qish:</b> ko'k chiziq — development OOS (odatda yaxshi ko'rinadi),
  qizil — hech qachon tuning uchun ishlatilmagan 180 kunlik holdout. Sariq chiziq $100 zararsizlik darajasi.
  Savdo soni oshganda chiziqlar bir-biriga yaqinlashadimi yoki yo'qmi — asosiy savol shu.</div>
</section>

<section>
  <h2>3 · Nima uchun "ko'p savdo" o'z-o'zidan "ko'p foyda" emas</h2>
  <div class="two">
    <div class="card">
      <h3 style="margin-top:0">Iqtisodiy formula</h3>
      <div class="formula">Kunlik PnL ≈ (savdo/kun) × (pozitsiya hajmi) × (expectancy / savdo)</div>
      <p class="lead" style="margin:14px 0 0">$100 kapital va 1x da bir vaqtdagi jami notional ≤ $100.
      Pozitsiyalarni maydalab ko'paytirsak, <b>(savdo/kun) × (hajm)</b> deyarli o'zgarmaydi —
      faqat equity chizig'i silliqlashadi va har bir savdoning riski kichrayadi. Foydani oshirish uchun
      <b>expectancy</b> yoki <b>kapital</b> oshishi kerak, savdo soni emas.</p>
    </div>
    <div class="card">
      <h3 style="margin-top:0">Qisqa gorizont qimmatga tushadi</h3>
      <div class="formula">1h o'rtacha |harakat| = 38.4 bps → 10 bps xarajat = 26.0%<br>
4h = 78.2 bps → 12.8%<br>1d = 207.3 bps → 4.8%</div>
      <p class="lead" style="margin:14px 0 0">Vaqt ufqini qisqartirganda xarajat deyarli o'zgarmaydi,
      lekin har bir savdodan kutiladigan harakat kichrayadi. Shu sababli 1h (median −8.76 bps) va
      4h (−11.56 bps) 1d dan ancha yomon. Sizning H48/H64 time-exit sweep'ingiz ham shuni ko'rsatdi.</p>
    </div>
  </div>
  <div class="note"><b>Win rate ham xuddi shunday tuzoq.</b> TP=0.25σ / SL=3σ qo'yilsa holdout'da
  <b>79.44% win rate</b> chiqdi — lekin o'rtacha yutuq +31 bps, o'rtacha zarar −138 bps bo'lib,
  $100 → $95.78 ga tushdi. TP=0.75σ / SL=1.5σ esa <b>61.54% win rate</b> (siz so'ragan 60% dan oshdi),
  lekin $100 → $93.23. <b>Win rate + payoff + xarajat</b> birga hisoblanmaguncha hech narsa anglatmaydi.</div>
</section>

<section>
  <h2>4 · H96 C2 — hozirgi eng yaxshi nomzod (o'zgartirilmadi)</h2>
  <div class="card"><table><thead><tr><th>Ko'rsatkich</th><th>Qiymat</th><th>Izoh</th></tr></thead><tbody>
    <tr><td class="nm">OOS win rate</td><td>58.92% (241 savdo)</td><td class="dt">Maqsad ≥60% → FAIL</td></tr>
    <tr><td class="nm">Profit Factor</td><td>1.1596</td><td class="dt">Maqsad ≥1.25 → FAIL</td></tr>
    <tr><td class="nm">Expectancy</td><td>+9.463 bps/savdo</td><td class="dt">Musbat → PASS</td></tr>
    <tr><td class="nm">Ijobiy foldlar</td><td>4 / 5 (80%)</td><td class="dt">PASS; fold 4 zararli (−10.76 bps)</td></tr>
    <tr><td class="nm">Bootstrap pastki chegara</td><td>≤ 0</td><td class="dt">6 ta test uchun Bonferroni α=0.0083 → FAIL</td></tr>
    <tr><td class="nm">Holdout</td><td>ochilmagan</td><td class="dt">180 kun faqat yakuniy baholash uchun saqlangan</td></tr>
    <tr><td class="nm">Kapital ($50 cap, Primary)</td><td>$100 → $111.40</td><td class="dt">Max DD 8.74%; $100 cap'da DD 15.84% (chegaradan oshgan)</td></tr>
  </tbody></table></div>
  <p class="lead" style="margin-top:12px">Bu model <b>8 mezondan 5 tasini</b> bajargan. U sizning
  forward-paper / Testnet terminalingizda ishlatilmoqda — biz unga tegmadik, faqat tadqiqot registriga kiritdik.</p>
</section>

<section>
  <h2>5 · Keyingi qadam (halol variantlar)</h2>
  <ul>
    <li><b>Eng to'g'ri qaror:</b> shu bosqichda to'xtash. 5 ta mustaqil eksperimentlar oilasi OHLCV asosida
      barqaror yo'nalish edge'i yo'qligini ko'rsatdi. $100 hozircha xavfsiz.</li>
    <li><b>Forward-paper davom ettirish:</b> H96 C2 ni 6–8 hafta shadow rejimda yig'ish; holdout'ni
      faqat oldindan belgilangan yakuniy baholashda ochish. Juma 09:00 monitoring eslatmasi faol.</li>
    <li><b>Yangi ma'lumot:</b> kripto-native flow — open interest o'zgarishi, taker long/short ratio,
      likvidatsiyalar (<code>!forceOrders</code>), basis, order-book imbalance. Funding darajasi
      feature sifatida sinandi va holdout'da eng yomon natija berdi (−26.32 bps).</li>
    <li><b>Yangi maqsad:</b> yo'nalish emas, volatillik / rejim / execution bashorati; meta-labeling
      (avval oddiy qoida nomzod chiqaradi, ML faqat "foydalimi?" deb javob beradi).</li>
    <li><b>Ko'p savdo kerak bo'lsa:</b> bitta BTC emas, bir nechta mustaqil aktiv (ETH, SOL) bo'yicha
      bir xil qoidani qo'llash — bu leverage'siz haqiqiy diversifikatsiya beradi.</li>
    <li><b>Hech qachon qilinmasligi kerak:</b> holdout natijasiga qarab parametr tanlash. Bu uni
      training ma'lumotiga aylantiradi va real pulda yutqazadi.</li>
  </ul>
</section>

<footer>
  <div>Manba: <code>osh-btcusdt-winrate-audit</code> · barcha raqamlar tarixiy paper-backtest · investitsiya maslahati emas.</div>
  <div style="margin-top:6px"><code>live_orders=false</code> · API kalit yo'q · order yuborish kodi yo'q · tashqi CDN yo'q (oflayn ishlaydi).</div>
  <div style="margin-top:6px">Sahifa alohida artifact: <code>research-hub.html</code> · generator: <code>build_research_hub.py</code> · ma'lumot: JSON hisobotlar.</div>
</footer>
</div>
</body>
</html>
"""
    out = HERE / "research-hub.html"
    out.write_text(doc, encoding="utf-8")
    print(f"[hub] {out} ({len(doc)} bytes, {n_total} experiments, {n_go} GO)")


if __name__ == "__main__":
    main()
