import argparse
from pathlib import Path


def _iter_months(start_ym: str, end_ym: str):
    sy, sm = (int(x) for x in start_ym.split("-"))
    ey, em = (int(x) for x in end_ym.split("-"))

    y, m = sy, sm
    while (y < ey) or (y == ey and m <= em):
        yield y, m
        m += 1
        if m == 13:
            m = 1
            y += 1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="1990-01")
    ap.add_argument("--end", default="2014-12")
    ap.add_argument(
        "--cmp_script",
        default="/work/ab0995/a270135/Analysis/FESOM/MRana/t2m_cmp.py",
        help="Path to the comparison plotting script",
    )
    ap.add_argument(
        "--out_dir",
        default="/work/ab0995/a270135/Analysis/FESOM/MRana/figures",
        help="Folder where map PNGs are stored",
    )
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--progress_every", type=int, default=12)

    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # We intentionally use subprocess so this script can remain a thin orchestrator.
    import subprocess

    done = 0
    for i, (year, month) in enumerate(_iter_months(args.start, args.end), start=1):
        out_png = out_dir / f"t2m_cmp_{year}{month:02d}_robinson_0p5deg.png"
        if out_png.exists() and not args.overwrite:
            done += 1
            continue

        cmd = [
            "python",
            str(args.cmp_script),
            "--year",
            str(year),
            "--month",
            str(month),
            "--out",
            str(out_png),
        ]
        subprocess.run(cmd, check=True)
        done += 1

        if args.progress_every and (i % args.progress_every == 0):
            print(f"generated up to {year}-{month:02d} ({done} files)")

    print(f"done. map pngs in: {out_dir}")


if __name__ == "__main__":
    main()
