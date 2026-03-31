import argparse
from pathlib import Path

import healpy as hp
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

import cartopy.crs as ccrs
import cartopy.feature as cfeature


def _guess_data_var(ds: xr.Dataset) -> str:
    candidates = [
        v
        for v in ds.data_vars
        if v.lower() in {"mtpr", "tp", "pr"} or "precip" in v.lower() or "rain" in v.lower()
    ]
    if len(candidates) == 1:
        return candidates[0]
    if len(ds.data_vars) == 1:
        return next(iter(ds.data_vars))
    raise ValueError(f"Could not uniquely determine precipitation variable. data_vars={list(ds.data_vars)}")


def _get_monthly_healpix_values(path_grib: str, var_name: str | None = None) -> np.ndarray:
    ds = xr.open_dataset(path_grib, engine="cfgrib")
    if var_name is None:
        var_name = _guess_data_var(ds)

    da = ds[var_name]
    if "time" in da.dims:
        da = da.isel(time=0)

    vals = np.asarray(da.values, dtype=float)
    if vals.ndim != 1:
        raise ValueError(f"Expected 1D HEALPix values, got shape {vals.shape} from {path_grib}")

    fill = da.attrs.get("missingValue")
    if fill is not None:
        vals = np.where(vals == float(fill), np.nan, vals)

    return vals


def healpix_to_latlon_grid(
    healpix_vals: np.ndarray,
    lon_1d: np.ndarray,
    lat_1d: np.ndarray,
    nest: bool,
) -> np.ndarray:
    lon2d, lat2d = np.meshgrid(lon_1d, lat_1d)
    flat = hp.get_interp_val(healpix_vals, lon2d.ravel(), lat2d.ravel(), lonlat=True, nest=nest)
    return flat.reshape(lat2d.shape)


def _iter_months(start_year: int, end_year: int):
    for y in range(int(start_year), int(end_year) + 1):
        for m in range(1, 13):
            yield y, m


def _mean_over_period(path_tmpl: str, start_year: int, end_year: int, progress_every: int = 24) -> np.ndarray:
    total: np.ndarray | None = None
    n = 0
    missing = 0

    for i, (y, m) in enumerate(_iter_months(start_year, end_year), start=1):
        p = Path(path_tmpl.format(year=y, month=m))
        if not p.exists():
            missing += 1
            continue
        vals = _get_monthly_healpix_values(str(p))
        if total is None:
            total = np.zeros_like(vals, dtype=np.float64)
        total += vals
        n += 1
        if progress_every and (i % progress_every == 0):
            print(f"period-mean progress: {y}-{m:02d} (used={n}, missing={missing})")

    if total is None or n == 0:
        raise FileNotFoundError(
            f"No GRIB files found for template={path_tmpl!r} in {start_year}-{end_year}"
        )

    if missing:
        print(f"warning: missing {missing} files while computing mean over {start_year}-{end_year}")

    return (total / float(n)).astype(np.float64)


def _to_mm_per_day(vals: np.ndarray, units_hint: str | None) -> np.ndarray:
    # Most GRIB mtpr is kg m-2 s-1 (== mm/s). Convert to mm/day.
    if units_hint is None:
        return vals * 86400.0

    u = units_hint.lower().replace(" ", "")
    if "kgm-2s-1" in u or "kg/m^2/s" in u or "kgm**-2s**-1" in u or "mm/s" in u:
        return vals * 86400.0
    if "mm/day" in u or "mm/d" in u:
        return vals

    # Fallback assumption
    return vals * 86400.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mean_year_start", type=int, default=1995)
    ap.add_argument("--mean_year_end", type=int, default=2014)
    ap.add_argument(
        "--path319",
        default="/work/ab0995/a270135/MN5/projt319/a3be_grib/varyears/{year}/sfc_mtpr/mtpr_{year}{month:02d}.grib",
    )
    ap.add_argument(
        "--path399",
        default="/work/ab0995/a270135/MN5/projt399/a3bo_grib/varyears/{year}/sfc_mtpr/mtpr_{year}{month:02d}.grib",
    )
    ap.add_argument(
        "--out",
        default="/work/ab0995/a270135/Analysis/FESOM/MRana/figures/pr_cmp_mean_{start}_{end}_robinson_0p5deg.png",
    )
    ap.add_argument("--progress_every", type=int, default=24)

    ap.add_argument("--nest", action="store_true", default=True)
    ap.add_argument("--ring", dest="nest", action="store_false")
    ap.add_argument("--deg", type=float, default=0.5)

    ap.add_argument("--pr_vmax", type=float, default=16.0, help="Upper limit for precipitation (mm/day)")
    ap.add_argument("--pr_step", type=float, default=2.0)
    ap.add_argument("--diff_abs", type=float, default=2.0)
    ap.add_argument("--diff_step", type=float, default=0.5)

    args = ap.parse_args()

    pr319 = _mean_over_period(
        args.path319,
        start_year=args.mean_year_start,
        end_year=args.mean_year_end,
        progress_every=args.progress_every,
    )
    pr399 = _mean_over_period(
        args.path399,
        start_year=args.mean_year_start,
        end_year=args.mean_year_end,
        progress_every=args.progress_every,
    )

    # Units are not reliably available without re-opening a representative file.
    pr319_mm_day = _to_mm_per_day(pr319, None)
    pr399_mm_day = _to_mm_per_day(pr399, None)

    dlat = args.deg
    lat_1d = np.arange(-90.0, 90.0 + dlat, dlat)
    lon_1d = np.arange(-180.0, 180.0 + dlat, dlat)
    lon2d, lat2d = np.meshgrid(lon_1d, lat_1d)

    grid319 = healpix_to_latlon_grid(pr319_mm_day, lon_1d, lat_1d, nest=args.nest)
    grid399 = healpix_to_latlon_grid(pr399_mm_day, lon_1d, lat_1d, nest=args.nest)
    griddiff = grid399 - grid319

    vmax = float(args.pr_vmax)
    step = float(args.pr_step)
    pr_levels = np.arange(0.0, vmax + step, step)
    pr_cmap = plt.get_cmap("YlGnBu")
    pr_norm = mcolors.BoundaryNorm(pr_levels, ncolors=pr_cmap.N, clip=True)

    diff_abs = float(args.diff_abs)
    diff_step = float(args.diff_step)
    diff_levels = np.arange(-diff_abs, diff_abs + diff_step, diff_step)
    diff_cmap = plt.get_cmap("RdBu_r")
    diff_norm = mcolors.BoundaryNorm(diff_levels, ncolors=diff_cmap.N, clip=True)

    fig = plt.figure(figsize=(16, 11))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.08], hspace=0.18, wspace=0.02)

    ax1 = fig.add_subplot(gs[0, 0], projection=ccrs.Robinson())
    ax2 = fig.add_subplot(gs[0, 2], projection=ccrs.Robinson())
    ax3 = fig.add_subplot(gs[1, 1], projection=ccrs.Robinson())
    ax_blank_1 = fig.add_subplot(gs[0, 1])
    ax_blank_2 = fig.add_subplot(gs[1, 0])
    ax_blank_3 = fig.add_subplot(gs[1, 2])

    for ax in (ax_blank_1, ax_blank_2, ax_blank_3):
        ax.set_axis_off()

    ax1.set_position([0.04, 0.64, 0.44, 0.30])
    ax2.set_position([0.52, 0.64, 0.44, 0.30])
    ax3.set_position([0.18, 0.14, 0.64, 0.34])

    cax_top = fig.add_axes([0.20, 0.585, 0.60, 0.025])
    cax_bottom = fig.add_axes([0.28, 0.065, 0.44, 0.025])

    im1 = ax1.pcolormesh(
        lon2d,
        lat2d,
        grid319,
        transform=ccrs.PlateCarree(),
        cmap=pr_cmap,
        norm=pr_norm,
        shading="auto",
    )
    ax2.pcolormesh(
        lon2d,
        lat2d,
        grid399,
        transform=ccrs.PlateCarree(),
        cmap=pr_cmap,
        norm=pr_norm,
        shading="auto",
    )
    ax3.pcolormesh(
        lon2d,
        lat2d,
        griddiff,
        transform=ccrs.PlateCarree(),
        cmap=diff_cmap,
        norm=diff_norm,
        shading="auto",
    )

    for ax in (ax1, ax2, ax3):
        ax.coastlines(linewidth=0.7)
        ax.add_feature(cfeature.BORDERS, linewidth=0.4, color="gray", alpha=0.2)
        ax.gridlines(draw_labels=False, linewidth=0.3, color="gray", alpha=0.5)

    label = f"{args.mean_year_start}–{args.mean_year_end} mean"
    ax1.set_title(f"TCO319 mtpr (mm/day) — {label}", fontsize=22)
    ax2.set_title(f"TCO399 mtpr (mm/day) — {label}", fontsize=22)
    ax3.set_title(
        f"Difference (TCO399 − TCO319) (mm/day) — {label}",
        fontsize=22,
        pad=18,
    )

    cb1 = fig.colorbar(im1, cax=cax_top, orientation="horizontal")
    cb1.set_label("Precipitation (mm/day)", fontsize=18)
    cb1.ax.tick_params(labelsize=16)
    cb1.set_ticks(pr_levels)

    cb2 = fig.colorbar(plt.cm.ScalarMappable(norm=diff_norm, cmap=diff_cmap), cax=cax_bottom, orientation="horizontal")
    cb2.set_label("Δ precipitation (mm/day)", fontsize=18)
    cb2.ax.tick_params(labelsize=16)
    cb2.set_ticks(diff_levels)

    out_path = Path(args.out.format(start=args.mean_year_start, end=args.mean_year_end))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print("saved:", str(out_path))


if __name__ == "__main__":
    main()
