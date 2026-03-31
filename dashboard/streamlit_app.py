from pathlib import Path

import pandas as pd
import streamlit as st


def _read_tsv_table(path: Path) -> list[dict[str, str]]:
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    if not lines:
        return []
    header = lines[0].split("\t")
    out: list[dict[str, str]] = []
    for ln in lines[1:]:
        parts = ln.split("\t")
        if len(parts) != len(header):
            continue
        out.append({header[i]: parts[i] for i in range(len(header))})
    return out


def _perf_rows_from_csv(perf_summary_csv: Path) -> dict[str, dict[str, str]]:
    lines = perf_summary_csv.read_text(encoding="utf-8").strip().splitlines()
    rows = [ln.split(",") for ln in lines]

    metrics: dict[str, dict[str, str]] = {}
    if rows and len(rows[0]) >= 3:
        for r in rows[1:]:
            if len(r) < 3:
                continue
            metrics[r[0]] = {"tco319": r[1], "tco399": r[2]}

    return metrics


def _fmt_float_1(x: str) -> str:
    if x == "":
        return ""
    try:
        return f"{float(x):.1f}"
    except Exception:
        return x


def _show_image(path: Path, caption: str, missing_msg: str) -> None:
    if path.exists():
        st.image(str(path), caption=caption, use_container_width=True)
    else:
        st.warning(missing_msg)


def main() -> None:
    st.set_page_config(page_title="Climate Model Dashboard — t2m", layout="wide")

    base_dir = Path(__file__).resolve().parents[1]

    with st.sidebar:
        st.header("Settings")

        page = st.radio(
            "Page",
            options=[
                "TCO319 vs TCO399 (1990–2014)",
                "3 experiments + ERA5 (1990–1996)",
            ],
        )

        mean_year_start = st.number_input("Mean year start", value=1995, step=1)
        mean_year_end = st.number_input("Mean year end", value=2014, step=1)
        simulation_years = st.number_input("Simulation (yrs)", value=25, step=1)
        grid_points_319 = st.number_input("Grid points tco319", value=654400, step=1)
        grid_points_399 = st.number_input("Grid points tco399", value=421120, step=1)
        timestep_319_s = st.number_input("Timestep tco319 (s)", value=1200, step=1)
        timestep_399_s = st.number_input("Timestep tco399 (s)", value=900, step=1)

        fig_dir = Path(
            st.text_input("Figures dir", value=str(base_dir / "figures"))
        ).expanduser()
        csv_dir = Path(st.text_input("CSV dir", value=str(base_dir / "csv"))).expanduser()

    if page == "TCO319 vs TCO399 (1990–2014)":
        st.title("CLIMATE MODEL DASHBOARD")
        st.caption("TCO319 vs TCO399 — Simulation: 1990–2014")

        perf_summary_csv = csv_dir / "t2m_perf_summary.csv"

        runtime_png = fig_dir / "t2m_perf_runtime_per_chunk.png"
        ts_png = fig_dir / "t2m_global_mean_monthly_1990_2014.png"

        mean_map_png = fig_dir / f"t2m_cmp_mean_{mean_year_start}_{mean_year_end}_robinson_0p5deg.png"
        sic_mean_map_png = fig_dir / f"sic_cmp_mean_{mean_year_start}_{mean_year_end}_robinson_0p5deg.png"
        pr_mean_map_png = fig_dir / f"pr_cmp_mean_{mean_year_start}_{mean_year_end}_robinson_0p5deg.png"

        era5_cmp_319_png = fig_dir / "t2m_obs_cmp_era5_tco319_1990_2014_robinson_0p5deg.png"
        era5_cmp_399_png = fig_dir / "t2m_obs_cmp_era5_tco399_1990_2014_robinson_0p5deg.png"

        st.header("Performance summary")

        if perf_summary_csv.exists():
            metrics = _perf_rows_from_csv(perf_summary_csv)

            avg_rt_319 = metrics.get("avg_run_time_hhmm", {}).get("tco319", "")
            avg_rt_399 = metrics.get("avg_run_time_hhmm", {}).get("tco399", "")
            med_rt_319 = metrics.get("median_run_time_hhmm", {}).get("tco319", "")
            med_rt_399 = metrics.get("median_run_time_hhmm", {}).get("tco399", "")
            sypd_319 = _fmt_float_1(metrics.get("mean_sypd", {}).get("tco319", ""))
            sypd_399 = _fmt_float_1(metrics.get("mean_sypd", {}).get("tco399", ""))
            asypd_319 = _fmt_float_1(metrics.get("mean_asypd", {}).get("tco319", ""))
            asypd_399 = _fmt_float_1(metrics.get("mean_asypd", {}).get("tco399", ""))

            perf_rows = [
                ("Simulation (yrs)", str(simulation_years), str(simulation_years)),
                ("avg run time", avg_rt_319, avg_rt_399),
                ("median run time", med_rt_319, med_rt_399),
                ("mean SYPD", sypd_319, sypd_399),
                ("mean ASYPD", asypd_319, asypd_399),
                ("No. of Grid pts.", f"{int(grid_points_319):,}", f"{int(grid_points_399):,}"),
                ("Timestep", f"{int(timestep_319_s)} s", f"{int(timestep_399_s)} s"),
            ]

            st.table(
                {
                    "Resolutions": [r[0] for r in perf_rows],
                    "tco319": [r[1] for r in perf_rows],
                    "tco399": [r[2] for r in perf_rows],
                }
            )
        else:
            st.warning("Performance summary not found. Run t2m_perf.py first.")

        st.divider()
        st.header("Runtime per chunk")
        _show_image(runtime_png, "Runtime per chunk", "Missing runtime plot. Run t2m_perf.py first.")

        st.divider()
        st.header("2m temperature")
        st.subheader("Global monthly mean")
        _show_image(ts_png, "Global mean monthly t2m", "Missing time series plot. Run t2m_ts.py first.")

        st.subheader("Spatial mean")
        _show_image(
            mean_map_png,
            "Mean comparison map",
            f"Mean map PNG not found in figures folder. Generate with: t2m_cmp.py --mean_year_start {mean_year_start} --mean_year_end {mean_year_end}",
        )

        st.subheader("ERA5 comparison")
        _show_image(
            era5_cmp_319_png,
            "TCO319 vs ERA5 mean and bias",
            "ERA5 comparison PNG not found in figures folder. Generate with: era5_t2m_cmp.py --start_year 1990 --end_year 2014",
        )
        _show_image(
            era5_cmp_399_png,
            "TCO399 vs ERA5 mean and bias",
            "ERA5 comparison PNG not found in figures folder. Generate with: era5_t2m_cmp.py --start_year 1990 --end_year 2014",
        )

        st.divider()
        st.header("Sea ice concentration")
        _show_image(
            sic_mean_map_png,
            "Sea ice concentration mean comparison map",
            f"Sea-ice mean map PNG not found in figures folder. Generate with: sic_cmp.py --mean_year_start {mean_year_start} --mean_year_end {mean_year_end}",
        )

        st.divider()
        st.header("Precipitation rate")
        _show_image(
            pr_mean_map_png,
            "Precipitation mean comparison map",
            f"Precipitation mean map PNG not found in figures folder. Generate with: pr_cmp.py --mean_year_start {mean_year_start} --mean_year_end {mean_year_end}",
        )
    else:
        st.title("CLIMATE MODEL DASHBOARD")
        st.caption("3 experiments + ERA5 — Simulation: 1990–1996")

        ts_png = fig_dir / "t2m_global_mean_monthly_3exp_plus_era5_1990_1996.png"
        fig1_png = fig_dir / "t2m_3exp_plus_era5_mean_1990_1996_robinson_0p5deg.png"
        fig2_png = fig_dir / "t2m_bias_complete_1990_1996_robinson_0p5deg.png"
        fig3_png = fig_dir / "t2m_era5_biasdiff_2exp_vs_a3df_1990_1996_robinson_0p5deg.png"
        csv_3exp = csv_dir / "t2m_global_mean_monthly_3exp_plus_era5_1990_1996.csv"

        perf_a3be = csv_dir / "TCO319_ConsideredPerformance_a3be.csv"
        perf_a3bo = csv_dir / "TCO399_ConsideredPerformance_a3bo.csv"
        perf_a3df = csv_dir / "TCO319_ConsideredPerformance_a3df.csv"

        st.header("Figures")
        st.subheader("Global monthly mean")
        _show_image(ts_png, "Global mean monthly t2m (3 exp + ERA5)", "Missing plot. Re-run the notebook to generate it.")

        st.subheader("Mean maps")
        _show_image(fig1_png, "Mean maps (3 exp + ERA5)", "Missing mean-map figure. Re-run the notebook to generate it.")

        st.subheader("Bias maps")
        _show_image(fig2_png, "Bias maps", "Missing bias figure. Re-run the notebook to generate it.")

        st.subheader("Bias-difference maps")
        _show_image(fig3_png, "Bias-difference maps", "Missing bias-difference figure. Re-run the notebook to generate it.")

        st.divider()
        st.header("Data")
        if csv_3exp.exists():
            st.caption(str(csv_3exp))
            df_3exp = pd.read_csv(csv_3exp, parse_dates=["time"])
            st.dataframe(df_3exp, use_container_width=True)
        else:
            st.warning("3-experiment CSV not found. Re-run the notebook to generate it.")

        st.divider()
        st.header("Performance tables")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("a3be")
            if perf_a3be.exists():
                st.dataframe(_read_tsv_table(perf_a3be), use_container_width=True)
            else:
                st.warning("Missing CSV: TCO319_ConsideredPerformance_a3be.csv")

        with col2:
            st.subheader("a3df")
            if perf_a3df.exists():
                st.dataframe(_read_tsv_table(perf_a3df), use_container_width=True)
            else:
                st.warning("Missing CSV: TCO319_ConsideredPerformance_a3df.csv")

        with col3:
            st.subheader("a3bo")
            if perf_a3bo.exists():
                st.dataframe(_read_tsv_table(perf_a3bo), use_container_width=True)
            else:
                st.warning("Missing CSV: TCO399_ConsideredPerformance_a3bo.csv")


if __name__ == "__main__":
    main()
