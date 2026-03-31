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
        if v.lower() in {"t2m", "2t", "mean2t"} or "2m" in v.lower() or "t2" in v.lower()
    ]
    if len(candidates) == 1:
        return candidates[0]
    if len(ds.data_vars) == 1:
        return next(iter(ds.data_vars))
    raise ValueError(f"Could not uniquely determine temperature variable. data_vars={list(ds.data_vars)}")


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


def _iter_months(start_year: int, end_year: int):
    for y in range(int(start_year), int(end_year) + 1):
        for m in range(1, 13):
            yield y, m


def _mean_over_period_k(path_tmpl: str, start_year: int, end_year: int, progress_every: int = 24) -> np.ndarray:
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
        raise FileNotFoundError(f"No GRIB files found for template={path_tmpl!r} in {start_year}-{end_year}")

    if missing:
        print(f"warning: missing {missing} files while computing mean over {start_year}-{end_year}")

    return (total / float(n)).astype(np.float64)


def _to_c(vals_k: np.ndarray) -> np.ndarray:
    return vals_k - 273.15


def _healpix_to_latlon_grid(
    healpix_vals: np.ndarray,
    lon_1d: np.ndarray,
    lat_1d: np.ndarray,
    nest: bool,
) -> np.ndarray:
    lon2d, lat2d = np.meshgrid(lon_1d, lat_1d)
    flat = hp.get_interp_val(healpix_vals, lon2d.ravel(), lat2d.ravel(), lonlat=True, nest=nest)
    return flat.reshape(lat2d.shape)


def _era5_mean_t2m_c(
    url: str,
    start_date: str,
    end_date: str,
    lat_1d: np.ndarray,
    lon_1d: np.ndarray,
    nest: bool,
) -> np.ndarray:
    ds = xr.open_zarr(url, chunks={"time": 12})

    if "time" not in ds:
        raise ValueError("ERA5 dataset does not contain 'time' coordinate")

    t2m_var = "2t" if "2t" in ds.data_vars else _guess_data_var(ds)
    da = ds[t2m_var]

    ds_period = da.sel(time=slice(start_date, end_date))
    if ds_period.time.size == 0:
        raise ValueError(f"No ERA5 data found in {start_date}..{end_date}")

    print(f"ERA5 time steps selected: {int(ds_period.time.size)}")
    t2m_mean_k = ds_period.mean(dim="time").compute()

    if ("cell" in t2m_mean_k.dims) and (t2m_mean_k.ndim == 1):
        vals_k = np.asarray(t2m_mean_k.values, dtype=float)
        vals_c = _to_c(vals_k)
        return _healpix_to_latlon_grid(vals_c, lon_1d, lat_1d, nest=nest)

    if ("lat" in t2m_mean_k.dims) and ("lon" in t2m_mean_k.dims):
        mean_c = t2m_mean_k - 273.15
        mean_c = mean_c.interp(lat=lat_1d, lon=lon_1d)
        return np.asarray(mean_c.values, dtype=float)

    raise ValueError(f"Unsupported ERA5 mean dimensions: dims={t2m_mean_k.dims}, shape={t2m_mean_k.shape}")


def _plot_three_panel(
    grid_model: np.ndarray,
    grid_obs: np.ndarray,
    lon2d: np.ndarray,
    lat2d: np.ndarray,
    out_path: Path,
    title_left: str,
    title_right: str,
    title_bottom: str,
    temp_min: float,
    temp_max: float,
    temp_step: float,
    bias_abs: float,
    bias_step: float,
) -> None:
    temp_levels = np.arange(temp_min, temp_max + temp_step, temp_step)
    temp_cmap = plt.get_cmap("RdBu_r")
    temp_norm = mcolors.BoundaryNorm(temp_levels, ncolors=temp_cmap.N, clip=True)

    bias_levels = np.arange(-bias_abs, bias_abs + bias_step, bias_step)
    bias_cmap = plt.get_cmap("RdBu_r")
    bias_norm = mcolors.BoundaryNorm(bias_levels, ncolors=bias_cmap.N, clip=True)

    grid_bias = grid_model - grid_obs

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
        grid_model,
        transform=ccrs.PlateCarree(),
        cmap=temp_cmap,
        norm=temp_norm,
        shading="auto",
    )
    ax2.pcolormesh(
        lon2d,
        lat2d,
        grid_obs,
        transform=ccrs.PlateCarree(),
        cmap=temp_cmap,
        norm=temp_norm,
        shading="auto",
    )
    ax3.pcolormesh(
        lon2d,
        lat2d,
        grid_bias,
        transform=ccrs.PlateCarree(),
        cmap=bias_cmap,
        norm=bias_norm,
        shading="auto",
    )

    for ax in (ax1, ax2, ax3):
        ax.coastlines(linewidth=0.7)
        ax.add_feature(cfeature.BORDERS, linewidth=0.4, color="gray", alpha=0.2)
        ax.gridlines(draw_labels=False, linewidth=0.3, color="gray", alpha=0.5)

    ax1.set_title(title_left, fontsize=22)
    ax2.set_title(title_right, fontsize=22)
    ax3.set_title(title_bottom, fontsize=22, pad=18)

    cb1 = fig.colorbar(im1, cax=cax_top, orientation="horizontal")
    cb1.set_label("t2m (°C)", fontsize=18)
    cb1.ax.tick_params(labelsize=16)
    cb1.set_ticks(np.arange(temp_min, temp_max + 0.001, 6.0))

    cb2 = fig.colorbar(plt.cm.ScalarMappable(norm=bias_norm, cmap=bias_cmap), cax=cax_bottom, orientation="horizontal")
    cb2.set_label("Bias (°C)", fontsize=18)
    cb2.ax.tick_params(labelsize=16)
    cb2.set_ticks(bias_levels)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start_year", type=int, default=1990)
    ap.add_argument("--end_year", type=int, default=2014)
    ap.add_argument("--deg", type=float, default=0.5)
    ap.add_argument("--progress_every", type=int, default=24)

    ap.add_argument(
        "--era5_url",
        default="https://swift.dkrz.de/v1/dkrz_41caca03ec414c2f95f52b23b775134f/reanalysis/v1/ERA5_P1M_6.zarr",
    )

    ap.add_argument(
        "--path319",
        default="/work/ab0995/a270135/MN5/projt319/a3be_grib/varyears/{year}/sfc_mean2t/mean2t_{year}{month:02d}.grib",
    )
    ap.add_argument(
        "--path399",
        default="/work/ab0995/a270135/MN5/projt399/a3bo_grib/varyears/{year}/sfc_mean2t/mean2t_{year}{month:02d}.grib",
    )

    ap.add_argument(
        "--out_dir",
        default="/work/ab0995/a270135/Analysis/FESOM/MRana/figures",
    )
    ap.add_argument(
        "--obs_dir",
        default="/work/ab0995/a270135/Analysis/FESOM/MRana/obs",
    )

    ap.add_argument("--nest", action="store_true", default=True)
    ap.add_argument("--ring", dest="nest", action="store_false")

    ap.add_argument("--temp_min", type=float, default=-36.0)
    ap.add_argument("--temp_max", type=float, default=36.0)
    ap.add_argument("--temp_step", type=float, default=3.0)
    ap.add_argument("--bias_abs", type=float, default=6.0)
    ap.add_argument("--bias_step", type=float, default=1.0)

    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    obs_dir = Path(args.obs_dir)
    obs_dir.mkdir(parents=True, exist_ok=True)

    lat_1d = np.arange(-90.0, 90.0 + args.deg, args.deg)
    lon_1d = np.arange(-180.0, 180.0 + args.deg, args.deg)
    lon2d, lat2d = np.meshgrid(lon_1d, lat_1d)

    start_date = f"{args.start_year:04d}-01-01"
    end_date = f"{args.end_year:04d}-12-31"

    obs_cache = obs_dir / f"ERA5_t2m_mean_{args.start_year}_{args.end_year}_lat_lon_{args.deg}deg_C.nc"
    if obs_cache.exists():
        print(f"loading cached ERA5 mean: {obs_cache}")
        ds_obs = xr.open_dataset(obs_cache)
        grid_era5 = np.asarray(ds_obs["t2m"].values, dtype=float)
    else:
        print(f"computing ERA5 mean t2m for {args.start_year}-{args.end_year}")
        grid_era5 = _era5_mean_t2m_c(
            url=args.era5_url,
            start_date=start_date,
            end_date=end_date,
            lat_1d=lat_1d,
            lon_1d=lon_1d,
            nest=args.nest,
        )

        ds_out = xr.Dataset(
            {"t2m": (("lat", "lon"), grid_era5)},
            coords={"lat": lat_1d, "lon": lon_1d},
        )
        ds_out["t2m"].attrs["units"] = "degC"
        ds_out.attrs["source"] = "ERA5"
        ds_out.attrs["period"] = f"{args.start_year}-{args.end_year}"
        ds_out.to_netcdf(obs_cache)
        print(f"saved ERA5 cache: {obs_cache}")

    print(f"computing model means for {args.start_year}-{args.end_year}")
    vals319_k = _mean_over_period_k(args.path319, args.start_year, args.end_year, progress_every=args.progress_every)
    vals399_k = _mean_over_period_k(args.path399, args.start_year, args.end_year, progress_every=args.progress_every)

    grid319 = _healpix_to_latlon_grid(_to_c(vals319_k), lon_1d, lat_1d, nest=args.nest)
    grid399 = _healpix_to_latlon_grid(_to_c(vals399_k), lon_1d, lat_1d, nest=args.nest)

    label = f"{args.start_year}–{args.end_year} mean"

    out_319 = out_dir / f"t2m_obs_cmp_era5_tco319_{args.start_year}_{args.end_year}_robinson_{args.deg}deg.png"
    _plot_three_panel(
        grid_model=grid319,
        grid_obs=grid_era5,
        lon2d=lon2d,
        lat2d=lat2d,
        out_path=out_319,
        title_left=f"TCO319 mean2t (°C) — {label}",
        title_right=f"ERA5 t2m (°C) — {label}",
        title_bottom=f"Bias (TCO319 − ERA5) (°C) — {label}",
        temp_min=args.temp_min,
        temp_max=args.temp_max,
        temp_step=args.temp_step,
        bias_abs=args.bias_abs,
        bias_step=args.bias_step,
    )
    print("saved:", str(out_319))

    out_399 = out_dir / f"t2m_obs_cmp_era5_tco399_{args.start_year}_{args.end_year}_robinson_{args.deg}deg.png"
    _plot_three_panel(
        grid_model=grid399,
        grid_obs=grid_era5,
        lon2d=lon2d,
        lat2d=lat2d,
        out_path=out_399,
        title_left=f"TCO399 mean2t (°C) — {label}",
        title_right=f"ERA5 t2m (°C) — {label}",
        title_bottom=f"Bias (TCO399 − ERA5) (°C) — {label}",
        temp_min=args.temp_min,
        temp_max=args.temp_max,
        temp_step=args.temp_step,
        bias_abs=args.bias_abs,
        bias_step=args.bias_step,
    )
    print("saved:", str(out_399))


if __name__ == "__main__":
    main()
