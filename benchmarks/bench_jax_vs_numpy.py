"""Time NumPy against JAX on CPU and GPU across problem sizes, and draw the README chart."""

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

import lexicase

SEED = 0
REPEATS = 5
SIZES = [(100, 50), (250, 100), (500, 200), (1000, 300), (2000, 500)]

METHODS = {
    "lexicase": lambda f, n: lexicase.lexicase_selection(f, n, seed=SEED),
    "epsilon (MAD)": lambda f, n: lexicase.epsilon_lexicase_selection(f, n, seed=SEED),
    "downsample 10%": lambda f, n: lexicase.downsample_lexicase_selection(
        f, n, max(1, f.shape[1] // 10), seed=SEED
    ),
    "dalex": lambda f, n: lexicase.dalex_selection(f, n, seed=SEED),
}

SERIES = [
    ("numpy", "cpu", "#2a78d6", "#3987e5"),
    ("jax (cpu)", "cpu", "#eb6834", "#d95926"),
    ("jax (gpu)", "gpu", "#1baf7a", "#199e70"),
]

LIGHT = {
    "surface": "#fcfcfb",
    "primary": "#0b0b0b",
    "secondary": "#52514e",
    "grid": "#e2e1dc",
    "index": 2,
}
DARK = {
    "surface": "#1a1a19",
    "primary": "#ffffff",
    "secondary": "#c3c2b7",
    "grid": "#383835",
    "index": 3,
}


def make_matrix(n_individuals, n_cases, label, device):
    """A population with real ties, which is what makes lexicase do work."""
    rng = np.random.default_rng(SEED)
    matrix = rng.integers(0, 4, size=(n_individuals, n_cases)).astype(np.float32)
    if label == "numpy":
        return matrix
    import jax
    import jax.numpy as jnp

    return jax.device_put(jnp.asarray(matrix), jax.devices(device)[0])


def time_call(call, matrix, num_selected, is_jax):
    """Median milliseconds per call, after one warmup call that pays for compilation."""
    result = call(matrix, num_selected)
    if is_jax:
        result.block_until_ready()

    timings = []
    for _ in range(REPEATS):
        start = time.perf_counter()
        result = call(matrix, num_selected)
        if is_jax:
            result.block_until_ready()
        timings.append(time.perf_counter() - start)
    return 1000 * float(np.median(timings))


def cpu_name():
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def available_series():
    if not lexicase.jax_is_available():
        return [SERIES[0]]
    import jax

    platforms = {device.platform for device in jax.devices()} | {"cpu"}
    return [s for s in SERIES if s[1] in platforms or s[0] == "numpy"]


def measure(series):
    """timings[method][series label] = list of milliseconds, one per size."""
    timings = {name: {label: [] for label, *_ in series} for name in METHODS}
    for n_individuals, n_cases in SIZES:
        print(f"=== {n_individuals} x {n_cases}")
        for label, device, *_ in series:
            matrix = make_matrix(n_individuals, n_cases, label, device)
            for name, call in METHODS.items():
                milliseconds = time_call(
                    call, matrix, n_individuals, label != "numpy"
                )
                timings[name][label].append(milliseconds)
                print(f"  {name:16s} {label:10s} {milliseconds:9.2f} ms")
    return timings


def draw(timings, series, theme, out_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [f"{n}x{m}" for n, m in SIZES]
    x = np.arange(len(SIZES))

    figure, axes = plt.subplots(2, 2, figsize=(9.5, 7.2), sharex=True)
    figure.patch.set_facecolor(theme["surface"])

    for panel, (name, per_series) in zip(axes.flat, timings.items()):
        panel.set_facecolor(theme["surface"])
        panel.set_yscale("log")
        panel.grid(True, which="major", color=theme["grid"], linewidth=1, zorder=0)
        panel.set_axisbelow(True)
        for spine in ("top", "right"):
            panel.spines[spine].set_visible(False)
        for spine in ("left", "bottom"):
            panel.spines[spine].set_color(theme["grid"])

        for label, _device, *colors in series:
            color = colors[theme["index"] - 2]
            values = per_series[label]
            panel.plot(
                x, values, color=color, linewidth=2, marker="o", markersize=8,
                markeredgecolor=theme["surface"], markeredgewidth=2, zorder=3,
                label=label, clip_on=False,
            )

        panel.set_title(name, color=theme["primary"], fontsize=11, loc="left", pad=8)
        panel.tick_params(colors=theme["secondary"], labelsize=9, length=0)
        panel.set_xticks(x)
        panel.set_xticklabels(labels, rotation=30, ha="right")
        panel.set_xlim(-0.3, len(SIZES) - 1 + 0.3)

    for panel in axes[:, 0]:
        panel.set_ylabel("milliseconds per call", color=theme["secondary"], fontsize=9)
    for panel in axes[1, :]:
        panel.set_xlabel("individuals x cases", color=theme["secondary"], fontsize=9)

    figure.suptitle(
        "Selection cost by backend, lower is better (log scale)",
        color=theme["primary"], fontsize=13, x=0.055, ha="left", y=0.985,
    )
    handles, legend_labels = axes.flat[0].get_legend_handles_labels()
    legend = figure.legend(
        handles, legend_labels, loc="upper right", bbox_to_anchor=(0.985, 0.995),
        frameon=False, ncol=len(series), fontsize=9,
    )
    for text in legend.get_texts():
        text.set_color(theme["secondary"])

    figure.tight_layout(rect=(0, 0, 0.93, 0.955), w_pad=4.0)
    figure.canvas.draw()
    for panel, per_series in zip(axes.flat, timings.values()):
        label_lines(panel, per_series, series, theme, x[-1])

    figure.savefig(out_path, dpi=160, facecolor=theme["surface"])
    plt.close(figure)
    print(f"wrote {out_path}")


def label_lines(panel, per_series, series, theme, last_x):
    """Direct-label each line at its right end, pushed apart so labels never collide."""
    minimum_gap = 13.0
    ends = sorted(
        ((panel.transData.transform((last_x, per_series[label][-1]))[1], label)
         for label, *_ in series),
        key=lambda pair: pair[0],
    )

    placed = []
    for pixel_y, label in ends:
        if placed and pixel_y - placed[-1][0] < minimum_gap:
            pixel_y = placed[-1][0] + minimum_gap
        placed.append((pixel_y, label))

    to_points = 72.0 / panel.figure.dpi
    for pixel_y, label in placed:
        original = panel.transData.transform((last_x, per_series[label][-1]))[1]
        panel.annotate(
            label,
            (last_x, per_series[label][-1]),
            textcoords="offset points",
            xytext=(10, (pixel_y - original) * to_points),
            va="center",
            fontsize=8.5,
            color=theme["secondary"],
            annotation_clip=False,
        )


def write_table(timings, series, out_path):
    labels = [label for label, *_ in series]
    lines = [
        "# NumPy against JAX",
        "",
        f"Milliseconds per call, median of {REPEATS} timed calls after one warmup call "
        "that pays for compilation. Each call selects as many parents as there are "
        "individuals, from a fitness matrix of integers in `{0, 1, 2, 3}`.",
        "",
        "```",
        f"cpu: {cpu_name()}",
        f"os: {platform.system()} {platform.machine()}",
        f"python: {platform.python_version()}",
        f"numpy: {np.__version__}",
    ]
    try:
        import jax

        devices = ", ".join(sorted({d.device_kind for d in jax.devices()}))
        lines.append(f"jax: {jax.__version__} on {devices}")
    except ImportError:
        lines.append("jax: not installed")
    lines += ["```", ""]

    for name, per_series in timings.items():
        lines += [
            f"## {name}",
            "",
            f"| individuals x cases | {' | '.join(labels)} | best JAX speedup |",
            "|" + "---|" * (len(labels) + 2),
        ]
        for index, (n_individuals, n_cases) in enumerate(SIZES):
            cells = [f"{per_series[label][index]:.2f}" for label in labels]
            jax_labels = [label for label in labels if label != "numpy"]
            if jax_labels:
                best = min(per_series[label][index] for label in jax_labels)
                speedup = f"{per_series['numpy'][index] / best:.2f}x"
            else:
                speedup = "n/a"
            lines.append(
                f"| {n_individuals} x {n_cases} | {' | '.join(cells)} | {speedup} |"
            )
        lines.append("")

    out_path.write_text("\n".join(lines) + "\n")
    print(f"wrote {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).parent)
    parser.add_argument(
        "--chart-dir", type=Path,
        default=Path(__file__).parent.parent / "docs" / "assets",
        help="where the PNGs go; the docs site and the README both read them from here",
    )
    parser.add_argument("--replot", action="store_true",
                        help="redraw from the last run's timings instead of measuring again")
    arguments = parser.parse_args()

    raw_path = arguments.out_dir / "results_jax_vs_numpy.json"

    if arguments.replot:
        payload = json.loads(raw_path.read_text())
        timings = payload["timings"]
        series = [tuple(entry) for entry in payload["series"]]
    else:
        series = available_series()
        if len(series) == 1:
            print("jax is not installed, nothing to compare against")
            return 1
        timings = measure(series)
        raw_path.write_text(
            json.dumps({"series": [list(s) for s in series], "timings": timings}, indent=1)
            + "\n"
        )
        write_table(timings, series, arguments.out_dir / "results_jax_vs_numpy.md")

    arguments.chart_dir.mkdir(parents=True, exist_ok=True)
    draw(timings, series, LIGHT, arguments.chart_dir / "jax_vs_numpy_light.png")
    draw(timings, series, DARK, arguments.chart_dir / "jax_vs_numpy_dark.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
