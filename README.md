# Climate Model Evaluation Dashboard

An interactive dashboard for exploring and comparing high-resolution climate model simulations against ERA5 reanalysis for different resolutions.

The dashboard demonstrates workflows developed for climate model evaluation using Python and HPC environments.

![Dashboard overview](figures/t2m_dashboard_preview.png)

## Live Dashboard

https://tunahim.github.io/MRana-dashboard/

How the data pipeline and the static site are built: [How it is built](https://tunahim.github.io/MRana-dashboard/engineering_overview.html).

## Features

- Interactive climate dashboard
- Multi-model comparison
- ERA5 evaluation
- Performance statistics
- Global temperature, sea ice and precipitation diagnostics

## Key technologies

- Python
- xarray
- Dask
- HEALPix
- GRIB
- ERA5
- Cartopy
- Matplotlib
- Climate model evaluation
- HPC workflows
- Windsurf AI

## Repository contents

- Dashboard HTML pages
- Python analysis scripts
- Jupyter notebooks
- Example figures

## Data sources and access

| Data | What it is | How it is accessed |
|---|---|---|
| Destination Earth Climate DT, IFS-FESOM experiments (TCo319 ~35 km, TCo399 ~28 km) | Coupled model output on HEALPix grids, GRIB/NetCDF | Read on the DKRZ HPC system with xarray, Dask and cfgrib; not public, so not included here |
| ERA5 reanalysis (Copernicus Climate Change Service) | 2 m temperature, sea ice, precipitation reference | Copy on the DKRZ HPC system |
| Workflow job reports | Simulated years per day (SYPD), core-hours per simulated year (CHSY) | Parsed from the run performance reports (`t2m_perf.py`); summaries in `csv/` |

The small derived tables in `csv/` and the HTML dashboards are what the live website uses, so it runs without access to the raw model data.

## Notes

- The original simulation data are not included because they are hosted on HPC systems and are not publicly distributable.
- The dashboard is intended as a demonstration of climate model evaluation workflows, visualisation, and software engineering practices.
- The analysis was developed and tested in HPC environments using Python, xarray, Dask, and related scientific libraries.

## Licence

Code: MIT, see [LICENSE](LICENSE). Data: see *Data sources and access*; third-party data keep their own terms.
