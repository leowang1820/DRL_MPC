"""Generate a smooth double-lane-change reference path for CarSim and Python.

The generated manoeuvre is a configurable research/training path.  It is not
claimed to be an ISO 3888 homologation course.
"""

import argparse
import csv
from pathlib import Path

import numpy as np


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = BASE_DIR / "paths"


def smootherstep(u: np.ndarray) -> np.ndarray:
    """Quintic transition with zero first/second derivative at both ends."""
    return 10.0 * u**3 - 15.0 * u**4 + 6.0 * u**5


def build_dlc_path(
    sample_dx: float = 0.1,
    lane_offset: float = 3.5,
    entry_end: float = 30.0,
    outbound_end: float = 60.0,
    hold_end: float = 85.0,
    return_end: float = 115.0,
    total_x: float = 160.0,
):
    boundaries = [0.0, entry_end, outbound_end, hold_end, return_end, total_x]
    if sample_dx <= 0.0:
        raise ValueError("sample_dx must be positive")
    if lane_offset == 0.0:
        raise ValueError("lane_offset must be non-zero")
    if any(b <= a for a, b in zip(boundaries, boundaries[1:])):
        raise ValueError("DLC x boundaries must be strictly increasing")

    x = np.arange(0.0, total_x + sample_dx * 0.5, sample_dx)
    y = np.zeros_like(x)

    outbound = (x >= entry_end) & (x < outbound_end)
    u_out = (x[outbound] - entry_end) / (outbound_end - entry_end)
    y[outbound] = lane_offset * smootherstep(u_out)

    hold = (x >= outbound_end) & (x < hold_end)
    y[hold] = lane_offset

    returning = (x >= hold_end) & (x < return_end)
    u_return = (x[returning] - hold_end) / (return_end - hold_end)
    y[returning] = lane_offset * (1.0 - smootherstep(u_return))

    dx = np.diff(x)
    dy = np.diff(y)
    station = np.concatenate(([0.0], np.cumsum(np.hypot(dx, dy))))
    yaw_deg = np.degrees(np.arctan2(np.gradient(y, x), 1.0))
    curvature = np.gradient(np.radians(yaw_deg), station, edge_order=2)
    return x, y, station, yaw_deg, curvature


def write_parsfile(path: Path, x, y, station) -> None:
    lines = [
        "PARSFILE",
        "#FullDataName Path: X-Y Coordinates`DLC Smooth 40 kmh`tracking-stability",
        "#RingCtrl0 1",
        "SET_IPATH_FOR_ID 0",
        "DEFINE_XY_TABLES 1",
        "ITAB_XY = NTAB_XY",
        "XY_TABLE_ID = ITAB_XY",
        "SET_DESCRIPTION PATH_ID DLC Smooth 40 kmh",
        "SET_DESCRIPTION XY_TABLE_ID DLC Smooth 40 kmh",
        "IPATHSEG = 1",
        "PATH_ID_DM = PATH_ID",
        "SEGMENT_TYPE = 1",
        "XY_SEGMENT_ID = XY_TABLE_ID",
        "",
        "#CheckBox0 0",
        "OPT_PATH_LOOP 0",
        "#CheckBox1 0",
        "#CheckBox2 0",
        "#CheckBox4 0",
        "",
        "#RadioCtrl0 0",
        "",
        "SPATH_START 0",
        "#DiagramOne0",
        "SEGMENT_XY_TABLE",
    ]
    lines.extend(
        f"{px:.4f}, {py:.4f}, {ps:.4f}"
        for px, py, ps in zip(x, y, station)
    )
    lines.extend(
        [
            "ENDTABLE",
            "",
            "LOG_ENTRY Used Dataset: Path: X-Y Coordinates; "
            "{ tracking-stability } DLC Smooth 40 kmh",
            "#Library : Path: X-Y Coordinates",
            "#DataSet : DLC Smooth 40 kmh",
            "#Category: tracking-stability",
            "#FileID  : PathXY_DLC_Smooth_40kmh",
            "#Product : CarSim 2022.1",
            "#DataVer : 2022.1",
            "#VehCode X-Y Coordinates of Path",
            "",
            "END",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_csv(path: Path, x, y, station, yaw_deg, curvature) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow(
            ["x_m", "y_m", "station_m", "path_yaw_deg", "curvature_1_per_m"]
        )
        writer.writerows(zip(x, y, station, yaw_deg, curvature))


def write_paste_table(path: Path, x, y, station) -> None:
    text = "\n".join(
        f"{px:.4f}\t{py:.4f}\t{ps:.4f}"
        for px, py, ps in zip(x, y, station)
    )
    path.write_text(text + "\n", encoding="utf-8")


def write_preview_svg(path: Path, x, y) -> None:
    width, height = 1100, 420
    left, right, top, bottom = 75, 35, 45, 65
    xmin, xmax = float(x.min()), float(x.max())
    ymin = min(-1.0, float(y.min()) - 0.5)
    ymax = max(4.5, float(y.max()) + 0.5)

    def sx(value):
        return left + (value - xmin) / (xmax - xmin) * (width - left - right)

    def sy(value):
        return height - bottom - (value - ymin) / (ymax - ymin) * (height - top - bottom)

    points = " ".join(f"{sx(px):.2f},{sy(py):.2f}" for px, py in zip(x, y))
    lane0 = sy(0.0)
    lane1 = sy(3.5)
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="white"/>
<text x="{width / 2}" y="25" text-anchor="middle" font-family="Arial" font-size="20">Smooth Double Lane Change Reference</text>
<line x1="{left}" y1="{lane0:.2f}" x2="{width-right}" y2="{lane0:.2f}" stroke="#999" stroke-dasharray="8,8"/>
<line x1="{left}" y1="{lane1:.2f}" x2="{width-right}" y2="{lane1:.2f}" stroke="#999" stroke-dasharray="8,8"/>
<polyline points="{points}" fill="none" stroke="#1261a0" stroke-width="4"/>
<circle cx="{sx(x[0]):.2f}" cy="{sy(y[0]):.2f}" r="6" fill="#2ca02c"/>
<circle cx="{sx(x[-1]):.2f}" cy="{sy(y[-1]):.2f}" r="6" fill="#d62728"/>
<line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" stroke="black"/>
<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" stroke="black"/>
<text x="{width / 2}" y="{height-18}" text-anchor="middle" font-family="Arial">X / m</text>
<text x="18" y="{height / 2}" text-anchor="middle" transform="rotate(-90 18 {height / 2})" font-family="Arial">Y / m (magnified)</text>
<text x="{left+12}" y="{top+20}" font-family="Arial" font-size="13">3.5 m lane offset; quintic transitions</text>
</svg>
"""
    path.write_text(svg, encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description="Generate a smooth DLC path")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--sample-dx", type=float, default=0.1)
    parser.add_argument("--lane-offset", type=float, default=3.5)
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data = build_dlc_path(sample_dx=args.sample_dx, lane_offset=args.lane_offset)
    x, y, station, yaw_deg, curvature = data

    par_path = args.output_dir / "dlc_reference_path.par"
    csv_path = args.output_dir / "dlc_reference_path.csv"
    table_path = args.output_dir / "dlc_reference_table.txt"
    preview_path = args.output_dir / "dlc_reference_preview.svg"
    write_parsfile(par_path, x, y, station)
    write_csv(csv_path, x, y, station, yaw_deg, curvature)
    write_paste_table(table_path, x, y, station)
    write_preview_svg(preview_path, x, y)

    print("DLC points:", len(x))
    print(f"Path length: {station[-1]:.3f} m")
    print(f"Maximum curvature: {np.max(np.abs(curvature)):.5f} 1/m")
    print("CarSim parsfile:", par_path)
    print("CarSim paste table:", table_path)
    print("Preview:", preview_path)


if __name__ == "__main__":
    main()
