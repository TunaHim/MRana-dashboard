import argparse
from pathlib import Path


def _parse_hhmmss(s: str) -> int:
    parts = s.strip().split(":")
    if len(parts) != 3:
        raise ValueError(f"Invalid time format: {s!r}")
    h, m, sec = (int(p) for p in parts)
    return h * 3600 + m * 60 + sec


def _format_hhmm(seconds: float) -> str:
    if seconds != seconds:
        return ""
    total = int(round(seconds))
    h = total // 3600
    m = (total % 3600) // 60
    return f"{h:02d}:{m:02d}"


def _safe_float(s: str) -> float | None:
    s = s.strip()
    if s == "":
        return None
    try:
        return float(s)
    except Exception:
        return None


def _perf_metrics_from_considered_perf_tsv(path: Path) -> dict[str, str]:
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    if not lines:
        return {}

    header = lines[0].split("\t")
    col_idx = {name: i for i, name in enumerate(header)}
    run_i = col_idx.get("Run")
    sypd_i = col_idx.get("SYPD")
    asypd_i = col_idx.get("ASYPD")
    if run_i is None:
        return {}

    run_s: list[int] = []
    sypd: list[float] = []
    asypd: list[float] = []

    for ln in lines[1:]:
        parts = ln.split("\t")
        if len(parts) < len(header):
            continue

        try:
            run_s.append(_parse_hhmmss(parts[run_i]))
        except Exception:
            pass

        if sypd_i is not None:
            v = _safe_float(parts[sypd_i])
            if v is not None:
                sypd.append(v)
        if asypd_i is not None:
            v = _safe_float(parts[asypd_i])
            if v is not None:
                asypd.append(v)

    if not run_s:
        return {}

    run_s_sorted = sorted(run_s)
    n = len(run_s_sorted)
    avg_run = sum(run_s_sorted) / float(n)
    med_run = run_s_sorted[n // 2] if (n % 2 == 1) else 0.5 * (run_s_sorted[n // 2 - 1] + run_s_sorted[n // 2])

    mean_sypd = sum(sypd) / float(len(sypd)) if sypd else float("nan")
    mean_asypd = sum(asypd) / float(len(asypd)) if asypd else float("nan")

    return {
        "chunks": str(n),
        "avg_run_time_hhmm": _format_hhmm(avg_run),
        "median_run_time_hhmm": _format_hhmm(med_run),
        "mean_sypd": f"{mean_sypd:.1f}" if mean_sypd == mean_sypd else "",
        "mean_asypd": f"{mean_asypd:.1f}" if mean_asypd == mean_asypd else "",
    }


def main() -> None:
    base_dir = Path(__file__).resolve().parents[1]

    ap = argparse.ArgumentParser()
    ap.add_argument("--mean_year_start", type=int, default=1995)
    ap.add_argument("--mean_year_end", type=int, default=2014)
    ap.add_argument("--simulation_years", type=int, default=25)
    ap.add_argument("--grid_points_319", type=int, default=421120)
    ap.add_argument("--grid_points_399", type=int, default=654400)
    ap.add_argument("--timestep_319_s", type=int, default=1200)
    ap.add_argument("--timestep_399_s", type=int, default=900)
    ap.add_argument(
        "--fig_dir",
        default=str(base_dir / "figures"),
    )
    ap.add_argument(
        "--csv_dir",
        default=str(base_dir / "csv"),
    )
    ap.add_argument(
        "--out_html",
        default=str(base_dir / "t2m_dashboard.html"),
    )
    ap.add_argument(
        "--out_html_3exp",
        default=str(base_dir / "t2m_dashboard_3exp_1990_1996.html"),
    )

    args = ap.parse_args()

    fig_dir = Path(args.fig_dir)
    csv_dir = Path(args.csv_dir)
    out_html = Path(args.out_html)
    out_html_3exp = Path(args.out_html_3exp)
    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_html_3exp.parent.mkdir(parents=True, exist_ok=True)

    perf_summary_csv = csv_dir / "t2m_perf_summary.csv"
    runtime_png = fig_dir / "t2m_perf_runtime_per_chunk.png"
    ts_png = fig_dir / "t2m_global_mean_monthly_1990_2014.png"

    mean_map_png = fig_dir / f"t2m_cmp_mean_{args.mean_year_start}_{args.mean_year_end}_robinson_0p5deg.png"
    sic_mean_map_png = fig_dir / f"sic_cmp_mean_{args.mean_year_start}_{args.mean_year_end}_robinson_0p5deg.png"
    pr_mean_map_png = fig_dir / f"pr_cmp_mean_{args.mean_year_start}_{args.mean_year_end}_robinson_0p5deg.png"

    era5_cmp_319_png = fig_dir / "t2m_obs_cmp_era5_tco319_1990_2014_robinson_0p5deg.png"
    era5_cmp_399_png = fig_dir / "t2m_obs_cmp_era5_tco399_1990_2014_robinson_0p5deg.png"

    ts_3exp_png = fig_dir / "t2m_global_mean_monthly_3exp_plus_era5_1990_1996.png"
    fig1_3exp_png = fig_dir / "t2m_3exp_plus_era5_mean_1990_1996_robinson_0p5deg.png"
    fig2_3exp_png = fig_dir / "t2m_bias_complete_1990_1996_robinson_0p5deg.png"
    fig3_3exp_png = fig_dir / "t2m_era5_biasdiff_2exp_vs_a3df_1990_1996_robinson_0p5deg.png"

    runtime_3exp_png = fig_dir / "t2m_perf_runtime_per_chunk_3exp.png"

    csv_3exp = csv_dir / "t2m_global_mean_monthly_3exp_plus_era5_1990_1996.csv"
    perf_a3be = csv_dir / "TCO319_ConsideredPerformance_a3be.csv"
    perf_a3df = csv_dir / "TCO319_ConsideredPerformance_a3df.csv"
    perf_a3bo = csv_dir / "TCO399_ConsideredPerformance_a3bo.csv"

    link_to_3exp = out_html_3exp.name
    link_to_main = out_html.name

    perf_table_html = "<p><em>Performance summary not found. Run t2m_perf.py first.</em></p>"
    if perf_summary_csv.exists():
        lines = perf_summary_csv.read_text(encoding="utf-8").strip().splitlines()
        rows = [ln.split(",") for ln in lines]

        metrics: dict[str, dict[str, str]] = {}
        if rows and len(rows[0]) >= 3:
            for r in rows[1:]:
                if len(r) < 3:
                    continue
                metrics[r[0]] = {"tco319": r[1], "tco399": r[2]}

        avg_rt_319 = metrics.get("avg_run_time_hhmm", {}).get("tco319", "")
        avg_rt_399 = metrics.get("avg_run_time_hhmm", {}).get("tco399", "")
        med_rt_319 = metrics.get("median_run_time_hhmm", {}).get("tco319", "")
        med_rt_399 = metrics.get("median_run_time_hhmm", {}).get("tco399", "")
        sypd_319 = metrics.get("mean_sypd", {}).get("tco319", "")
        sypd_399 = metrics.get("mean_sypd", {}).get("tco399", "")
        asypd_319 = metrics.get("mean_asypd", {}).get("tco319", "")
        asypd_399 = metrics.get("mean_asypd", {}).get("tco399", "")

        try:
            sypd_319 = f"{float(sypd_319):.1f}" if sypd_319 != "" else ""
        except Exception:
            pass
        try:
            sypd_399 = f"{float(sypd_399):.1f}" if sypd_399 != "" else ""
        except Exception:
            pass
        try:
            asypd_319 = f"{float(asypd_319):.1f}" if asypd_319 != "" else ""
        except Exception:
            pass
        try:
            asypd_399 = f"{float(asypd_399):.1f}" if asypd_399 != "" else ""
        except Exception:
            pass

        perf_rows = [
            ["Simulation (yrs)", str(args.simulation_years), str(args.simulation_years)],
            ["avg run time", avg_rt_319, avg_rt_399],
            ["median run time", med_rt_319, med_rt_399],
            ["mean SYPD", sypd_319, sypd_399],
            ["mean ASYPD", asypd_319, asypd_399],
            ["No. of Grid pts.", f"{args.grid_points_319:,}", f"{args.grid_points_399:,}"],
            ["Timestep", f"{args.timestep_319_s} s", f"{args.timestep_399_s} s"],
        ]

        perf_table_html = (
            "<table class='tbl'>\n"
            "<thead><tr><th>Resolutions</th><th>tco319</th><th>tco399</th></tr></thead>\n"
            "<tbody>"
            + "".join(
                "<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in perf_rows
            )
            + "</tbody></table>"
        )

    html = f"""<!doctype html>
<html lang='en'>
<head>
  <meta charset='utf-8'>
  <meta name='viewport' content='width=device-width, initial-scale=1'>
  <title>Climate Model Dashboard — t2m</title>
  <style>
    body {{ font-family: Arial, Helvetica, sans-serif; margin: 0; color: #111; }}
    h1 {{ margin: 0 0 8px 0; font-size: 28px; }}
    h2 {{ margin: 22px 0 10px 0; font-size: 20px; }}
    .sub {{ color: #444; margin-bottom: 16px; }}
    .layout {{ display: grid; grid-template-columns: 260px 1fr; min-height: 100vh; }}
    .sidebar {{ padding: 18px 14px; border-right: 1px solid #e6e6e6; background: #fafafa; }}
    .content {{ padding: 20px; }}
    .nav a {{ display: block; padding: 10px 10px; margin: 6px 0; border-radius: 8px; text-decoration: none; color: #111; border: 1px solid #e6e6e6; background: #fff; }}
    .nav a:hover {{ background: #f2f2f2; }}
    .nav a.subitem {{ margin-left: 14px; padding: 8px 10px; font-size: 13px; }}
    .grid {{ display: grid; grid-template-columns: 1fr; gap: 14px; }}
    .card {{ border: 1px solid #ddd; border-radius: 10px; padding: 14px; background: #fff; }}
    .tbl {{ border-collapse: collapse; width: 100%; font-size: 14px; }}
    .tbl th, .tbl td {{ border: 1px solid #ddd; padding: 6px 8px; }}
    .tbl th {{ background: #f5f5f5; text-align: left; }}
    img {{ max-width: 100%; height: auto; border-radius: 8px; border: 1px solid #eee; }}
    select {{ font-size: 14px; padding: 6px 8px; }}
    .warn {{ color: #a00; }}
  </style>
</head>
<body>
  <div class='layout'>
    <div class='sidebar'>
      <div style='font-weight: 700; margin-bottom: 10px;'>Navigation</div>
      <div class='nav'>
        <a href='{link_to_3exp}'>3 experiments + ERA5 (1990–1996)</a>
        <a href='#perf'>Performance summary</a>
        <a href='#runtime'>Runtime per chunk</a>
        <a href='#t2m'>2m temperature</a>
        <a class='subitem' href='#ts'>Global monthly mean</a>
        <a class='subitem' href='#map'>Spatial mean</a>
        <a href='#sic'>Sea ice concentration</a>
        <a href='#pr'>Precipitation rate</a>
      </div>
    </div>

    <div class='content'>
      <h1>CLIMATE MODEL DASHBOARD</h1>
      <div class='sub'>TCO319 vs TCO399 — Simulation: 1990–2014</div>

      <div class='grid'>
        <div class='card' id='perf'>
          <h2>Performance summary</h2>
          {perf_table_html}
        </div>

        <div class='card' id='runtime'>
          <h2>Runtime per chunk</h2>
          <div id='runtime_warn' class='warn'></div>
          <img id='runtime_img' src='figures/{runtime_png.name}' alt='Runtime per chunk'>
        </div>

        <div class='card' id='t2m'>
          <h2>2m temperature</h2>
          <div style='color: #444; margin-top: -2px;'>Quick links: <a href='#ts'>Global monthly mean</a> · <a href='#map'>Spatial mean</a> · <a href='#obs'>ERA5 comparison</a></div>
        </div>

        <div class='card' id='ts'>
          <h2>Global monthly mean</h2>
          <div id='ts_warn' class='warn'></div>
          <img id='ts_img' src='figures/{ts_png.name}' alt='Global mean monthly t2m'>
        </div>

        <div class='card' id='map'>
          <h2>Spatial mean</h2>
          <div id='map_warn' class='warn'></div>
          <img id='map_img' src='figures/{mean_map_png.name}' alt='Mean comparison map'>
        </div>

        <div class='card' id='obs'>
          <h2>ERA5 comparison</h2>
          <div id='era5_319_warn' class='warn'></div>
          <img id='era5_319_img' src='figures/{era5_cmp_319_png.name}' alt='TCO319 vs ERA5 mean and bias'>
          <div style='height: 10px;'></div>
          <div id='era5_399_warn' class='warn'></div>
          <img id='era5_399_img' src='figures/{era5_cmp_399_png.name}' alt='TCO399 vs ERA5 mean and bias'>
        </div>

        <div class='card' id='sic'>
          <h2>Sea ice concentration</h2>
          <div id='sic_warn' class='warn'></div>
          <img id='sic_img' src='figures/{sic_mean_map_png.name}' alt='Sea ice concentration mean comparison map'>
        </div>

        <div class='card' id='pr'>
          <h2>Precipitation rate</h2>
          <div id='pr_warn' class='warn'></div>
          <img id='pr_img' src='figures/{pr_mean_map_png.name}' alt='Precipitation mean comparison map'>
        </div>
      </div>
    </div>
  </div>

<script>
  const img = document.getElementById('map_img');
  const sicImg = document.getElementById('sic_img');
  const prImg = document.getElementById('pr_img');
  const era5_319 = document.getElementById('era5_319_img');
  const era5_399 = document.getElementById('era5_399_img');

  const setWarn = (id, msg) => {{
    const el = document.getElementById(id);
    if (el) el.textContent = msg || '';
  }};

  img.onerror = () => setWarn('map_warn', 'Mean map PNG not found in figures folder. Generate with: t2m_cmp.py --mean_year_start {args.mean_year_start} --mean_year_end {args.mean_year_end}');
  img.onload = () => setWarn('map_warn', '');

  sicImg.onerror = () => setWarn('sic_warn', 'Sea-ice mean map PNG not found in figures folder. Generate with: sic_cmp.py --mean_year_start {args.mean_year_start} --mean_year_end {args.mean_year_end}');
  sicImg.onload = () => setWarn('sic_warn', '');

  prImg.onerror = () => setWarn('pr_warn', 'Precipitation mean map PNG not found in figures folder. Generate with: pr_cmp.py --mean_year_start {args.mean_year_start} --mean_year_end {args.mean_year_end}');
  prImg.onload = () => setWarn('pr_warn', '');

  era5_319.onerror = () => setWarn('era5_319_warn', 'ERA5 comparison PNG not found in figures folder. Generate with: era5_t2m_cmp.py --start_year 1990 --end_year 2014');
  era5_319.onload = () => setWarn('era5_319_warn', '');

  era5_399.onerror = () => setWarn('era5_399_warn', 'ERA5 comparison PNG not found in figures folder. Generate with: era5_t2m_cmp.py --start_year 1990 --end_year 2014');
  era5_399.onload = () => setWarn('era5_399_warn', '');

  const runtimeImg = document.getElementById('runtime_img');
  runtimeImg.onerror = () => setWarn('runtime_warn', 'Missing runtime plot. Run t2m_perf.py first.');
  runtimeImg.onload = () => setWarn('runtime_warn', '');

  const tsImg = document.getElementById('ts_img');
  tsImg.onerror = () => setWarn('ts_warn', 'Missing time series plot. Run t2m_ts.py first.');
  tsImg.onload = () => setWarn('ts_warn', '');
</script>
</body>
</html>
"""

    perf_a3be_metrics = (
        _perf_metrics_from_considered_perf_tsv(perf_a3be) if perf_a3be.exists() else {}
    )
    perf_a3df_metrics = (
        _perf_metrics_from_considered_perf_tsv(perf_a3df) if perf_a3df.exists() else {}
    )
    perf_a3bo_metrics = (
        _perf_metrics_from_considered_perf_tsv(perf_a3bo) if perf_a3bo.exists() else {}
    )

    perf_3exp_rows = [
        ["Simulation (yrs)", "25", "7", "25"],
        ["avg run time", "01:58", "02:18", "02:19"],
        ["median run time", "01:58", "02:19", "02:17"],
        ["mean SYPD", "4.1", "3.5", "3.5"],
        ["mean ASYPD", "4.0", "3.5", "3.4"],
        ["No. of Grid pts.", "421,120", "421,120", "654,400"],
        ["Timestep", "1200 s", "900 s", "900 s"],
    ]

    perf_3exp_breakdown_rows = [
        ["DCQ_BASIC", "138.57 hours", "74.54 hours", "168.81 hours"],
        ["LRA_GENERATOR", "963.82 hours", "139.75 hours", "5197.30 hours"],
        ["SIM", "1340747.02 hours", "463070.22 hours", "1699465.60 hours"],
        ["DCQ_FULL", "99.49 hours", "46.42 hours", "111.50 hours"],
        ["FIX_CONSTANT", "111.00 hours", "22.15 hours", "96.69 hours"],
    ]

    perf_3exp_table_html = (
        "<table class='tbl'>\n"
        "<thead><tr><th>Metric</th><th>TCO319 (1200s)</th><th>TCO319 (900s)</th><th>TCO399 (900s)</th></tr></thead>\n"
        "<tbody>"
        + "".join(
            "<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in perf_3exp_rows
        )
        + "</tbody></table>"
    )

    perf_3exp_breakdown_table_html = (
        "<table class='tbl'>\n"
        "<thead><tr><th>Component</th><th>TCO319 (1200s)</th><th>TCO319 (900s)</th><th>TCO399 (900s)</th></tr></thead>\n"
        "<tbody>"
        + "".join(
            "<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in perf_3exp_breakdown_rows
        )
        + "</tbody></table>"
    )

    html_3exp = f"""<!doctype html>
<html lang='en'>
<head>
  <meta charset='utf-8'>
  <meta name='viewport' content='width=device-width, initial-scale=1'>
  <title>Climate Model Dashboard — 3 experiments + ERA5</title>
  <style>
    body {{ font-family: Arial, Helvetica, sans-serif; margin: 0; color: #111; }}
    h1 {{ margin: 0 0 8px 0; font-size: 28px; }}
    h2 {{ margin: 22px 0 10px 0; font-size: 20px; }}
    .sub {{ color: #444; margin-bottom: 16px; }}
    .layout {{ display: grid; grid-template-columns: 260px 1fr; min-height: 100vh; }}
    .sidebar {{ padding: 18px 14px; border-right: 1px solid #e6e6e6; background: #fafafa; }}
    .content {{ padding: 20px; }}
    .nav a {{ display: block; padding: 10px 10px; margin: 6px 0; border-radius: 8px; text-decoration: none; color: #111; border: 1px solid #e6e6e6; background: #fff; }}
    .nav a:hover {{ background: #f2f2f2; }}
    .nav a.subitem {{ margin-left: 14px; padding: 8px 10px; font-size: 13px; }}
    .grid {{ display: grid; grid-template-columns: 1fr; gap: 14px; }}
    .card {{ border: 1px solid #ddd; border-radius: 10px; padding: 14px; background: #fff; }}
    .tbl {{ border-collapse: collapse; width: 100%; font-size: 14px; }}
    .tbl th, .tbl td {{ border: 1px solid #ddd; padding: 6px 8px; }}
    .tbl th {{ background: #f5f5f5; text-align: left; }}
    img {{ max-width: 100%; height: auto; border-radius: 8px; border: 1px solid #eee; }}
    .warn {{ color: #a00; }}
    code {{ background: #f4f4f4; padding: 1px 4px; border-radius: 4px; }}
  </style>
</head>
<body>
  <div class='layout'>
    <div class='sidebar'>
      <div style='font-weight: 700; margin-bottom: 10px;'>Navigation</div>
      <div class='nav'>
        <a href='{link_to_main}'>TCO319 vs TCO399 (1990–2014)</a>
        <a href='#perf'>Performance summary</a>
        <a href='#runtime'>Runtime per chunk</a>
        <a href='#ts'>Global monthly mean</a>
        <a href='#meanmaps'>Mean maps</a>
        <a href='#biasmaps'>Bias maps</a>
        <a href='#biasdiff'>Bias-difference maps</a>
      </div>
    </div>

    <div class='content'>
      <h1>CLIMATE MODEL DASHBOARD</h1>
      <div class='sub'>3 experiments + ERA5 — Simulation: 1990–1996</div>

      <div class='grid'>
        <div class='card' id='perf'>
          <h2>Performance summary</h2>
          {perf_3exp_table_html}
        </div>

        <div class='card' id='perf_breakdown'>
          <h2>Performance breakdown</h2>
          {perf_3exp_breakdown_table_html}
        </div>

        <div class='card' id='runtime'>
          <h2>Runtime per chunk</h2>
          <div id='runtime3_warn' class='warn'></div>
          <img id='runtime3_img' src='figures/{runtime_3exp_png.name}' alt='Runtime per chunk (3 exp)'>
        </div>

        <div class='card' id='ts'>
          <h2>Global monthly mean</h2>
          <div id='ts3_warn' class='warn'></div>
          <img id='ts3_img' src='figures/{ts_3exp_png.name}' alt='Global mean monthly t2m (3 exp + ERA5)'>
        </div>

        <div class='card' id='meanmaps'>
          <h2>Mean maps</h2>
          <div id='fig1_warn' class='warn'></div>
          <img id='fig1_img' src='figures/{fig1_3exp_png.name}' alt='Mean maps (3 exp + ERA5)'>
        </div>

        <div class='card' id='biasmaps'>
          <h2>Bias maps</h2>
          <div id='fig2_warn' class='warn'></div>
          <img id='fig2_img' src='figures/{fig2_3exp_png.name}' alt='Bias maps'>
        </div>

        <div class='card' id='biasdiff'>
          <h2>Bias-difference maps</h2>
          <div id='fig3_warn' class='warn'></div>
          <img id='fig3_img' src='figures/{fig3_3exp_png.name}' alt='Bias-difference maps'>
        </div>
      </div>
    </div>
  </div>

<script>
  const setWarn = (id, msg) => {{
    const el = document.getElementById(id);
    if (el) el.textContent = msg || '';
  }};

  const ts3 = document.getElementById('ts3_img');
  ts3.onerror = () => setWarn('ts3_warn', 'Missing plot. Re-run the 1990–1996 notebook to generate: {ts_3exp_png.name}');
  ts3.onload = () => setWarn('ts3_warn', '');

  const fig1 = document.getElementById('fig1_img');
  fig1.onerror = () => setWarn('fig1_warn', 'Missing figure. Re-run the 1990–1996 notebook to generate: {fig1_3exp_png.name}');
  fig1.onload = () => setWarn('fig1_warn', '');

  const fig2 = document.getElementById('fig2_img');
  fig2.onerror = () => setWarn('fig2_warn', 'Missing figure. Re-run the 1990–1996 notebook to generate: {fig2_3exp_png.name}');
  fig2.onload = () => setWarn('fig2_warn', '');

  const fig3 = document.getElementById('fig3_img');
  fig3.onerror = () => setWarn('fig3_warn', 'Missing figure. Re-run the 1990–1996 notebook to generate: {fig3_3exp_png.name}');
  fig3.onload = () => setWarn('fig3_warn', '');

  const runtime3 = document.getElementById('runtime3_img');
  runtime3.onerror = () => setWarn('runtime3_warn', 'Missing runtime plot. Generate and save as: {runtime_3exp_png.name}');
  runtime3.onload = () => setWarn('runtime3_warn', '');
</script>
</body>
</html>
"""

    out_html.write_text(html, encoding="utf-8")
    print("saved:", str(out_html))
    out_html_3exp.write_text(html_3exp, encoding="utf-8")
    print("saved:", str(out_html_3exp))
    print("Note: open the HTML file and ensure it sits next to the 'figures' and 'csv' folders.")


if __name__ == "__main__":
    main()
