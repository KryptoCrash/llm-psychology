from __future__ import annotations

import argparse
import json
import math
import random
import struct
import zlib
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

import oracle_utils
import prompts
from layer_sweep import MODEL_IDS, collect_activations, load_model


MODES = [str(i) for i in range(1, 11)] + ["qd", "da"]
PROMPT_STYLES = ["base", "explain"]
DATASETS = ["mmlu", "bbh"]
PLOT_KEYS = ["dataset", "mode", "prompt_style", "point_type"]

PALETTES = {
    "dataset": {
        "mmlu": (33, 113, 181),
        "bbh": (230, 126, 34),
    },
    "prompt_style": {
        "base": (39, 174, 96),
        "explain": (142, 68, 173),
    },
    "point_type": {
        "condition": (44, 62, 80),
        "baseline": (192, 57, 43),
    },
    "mode": {
        "1": (31, 119, 180),
        "2": (255, 127, 14),
        "3": (44, 160, 44),
        "4": (214, 39, 40),
        "5": (148, 103, 189),
        "6": (140, 86, 75),
        "7": (227, 119, 194),
        "8": (127, 127, 127),
        "9": (188, 189, 34),
        "10": (23, 190, 207),
        "qd": (52, 73, 94),
        "da": (243, 156, 18),
    },
}

FONT_5X7 = {
    " ": ["00000", "00000", "00000", "00000", "00000", "00000", "00000"],
    "A": ["01110", "10001", "10001", "11111", "10001", "10001", "10001"],
    "B": ["11110", "10001", "10001", "11110", "10001", "10001", "11110"],
    "C": ["01111", "10000", "10000", "10000", "10000", "10000", "01111"],
    "D": ["11110", "10001", "10001", "10001", "10001", "10001", "11110"],
    "E": ["11111", "10000", "10000", "11110", "10000", "10000", "11111"],
    "F": ["11111", "10000", "10000", "11110", "10000", "10000", "10000"],
    "G": ["01111", "10000", "10000", "10011", "10001", "10001", "01111"],
    "H": ["10001", "10001", "10001", "11111", "10001", "10001", "10001"],
    "I": ["11111", "00100", "00100", "00100", "00100", "00100", "11111"],
    "J": ["00111", "00010", "00010", "00010", "00010", "10010", "01100"],
    "K": ["10001", "10010", "10100", "11000", "10100", "10010", "10001"],
    "L": ["10000", "10000", "10000", "10000", "10000", "10000", "11111"],
    "M": ["10001", "11011", "10101", "10101", "10001", "10001", "10001"],
    "N": ["10001", "11001", "10101", "10011", "10001", "10001", "10001"],
    "O": ["01110", "10001", "10001", "10001", "10001", "10001", "01110"],
    "P": ["11110", "10001", "10001", "11110", "10000", "10000", "10000"],
    "Q": ["01110", "10001", "10001", "10001", "10101", "10010", "01101"],
    "R": ["11110", "10001", "10001", "11110", "10100", "10010", "10001"],
    "S": ["01111", "10000", "10000", "01110", "00001", "00001", "11110"],
    "T": ["11111", "00100", "00100", "00100", "00100", "00100", "00100"],
    "U": ["10001", "10001", "10001", "10001", "10001", "10001", "01110"],
    "V": ["10001", "10001", "10001", "10001", "10001", "01010", "00100"],
    "W": ["10001", "10001", "10001", "10101", "10101", "10101", "01010"],
    "X": ["10001", "10001", "01010", "00100", "01010", "10001", "10001"],
    "Y": ["10001", "10001", "01010", "00100", "00100", "00100", "00100"],
    "Z": ["11111", "00001", "00010", "00100", "01000", "10000", "11111"],
    "0": ["01110", "10001", "10011", "10101", "11001", "10001", "01110"],
    "1": ["00100", "01100", "00100", "00100", "00100", "00100", "01110"],
    "2": ["01110", "10001", "00001", "00010", "00100", "01000", "11111"],
    "3": ["11110", "00001", "00001", "01110", "00001", "00001", "11110"],
    "4": ["00010", "00110", "01010", "10010", "11111", "00010", "00010"],
    "5": ["11111", "10000", "10000", "11110", "00001", "00001", "11110"],
    "6": ["01110", "10000", "10000", "11110", "10001", "10001", "01110"],
    "7": ["11111", "00001", "00010", "00100", "01000", "01000", "01000"],
    "8": ["01110", "10001", "10001", "01110", "10001", "10001", "01110"],
    "9": ["01110", "10001", "10001", "01111", "00001", "00001", "01110"],
    ".": ["00000", "00000", "00000", "00000", "00000", "01100", "01100"],
    ",": ["00000", "00000", "00000", "00000", "01100", "01100", "01000"],
    ":": ["00000", "01100", "01100", "00000", "01100", "01100", "00000"],
    "-": ["00000", "00000", "00000", "11111", "00000", "00000", "00000"],
    "_": ["00000", "00000", "00000", "00000", "00000", "00000", "11111"],
    "/": ["00001", "00010", "00010", "00100", "01000", "01000", "10000"],
    "%": ["11001", "11010", "00010", "00100", "01000", "01011", "10011"],
    "(": ["00010", "00100", "01000", "01000", "01000", "00100", "00010"],
    ")": ["01000", "00100", "00010", "00010", "00010", "00100", "01000"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run per-layer PCA over regenerated layer-sweep condition/baseline activations."
    )
    parser.add_argument("--model", choices=sorted(MODEL_IDS), default="qwen")
    parser.add_argument("--input-dir", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path, default=Path("layer_sweep_pca_results"))
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--layers", default="all", help='Comma-separated layer ids, or "all".')
    parser.add_argument("--activation-kind", choices=["residual", "delta"], default="residual")
    parser.add_argument("--position", choices=["assistant_start"], default="assistant_start")
    parser.add_argument("--baseline", choices=["random_answers", "same_prompt"], default="random_answers")
    parser.add_argument("--scopes", default="combined,mmlu,bbh")
    parser.add_argument("--plot-by", default="dataset,mode,prompt_style,point_type")
    parser.add_argument("--cache-dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument("--force-recompute", action="store_true")
    parser.add_argument("--skip-collection", action="store_true")
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument("--skip-static-plots", action="store_true")
    parser.add_argument("--skip-html", action="store_true")
    parser.add_argument("--pca-device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--pca-niter", type=int, default=4)
    parser.add_argument("--width", type=int, default=1200)
    parser.add_argument("--height", type=int, default=900)
    parser.add_argument("--point-radius", type=int, default=2)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def select_layers(model_name: str, layers_arg: str) -> list[int]:
    layer_count = oracle_utils.LAYER_COUNTS[model_name]
    if layers_arg == "all":
        return list(range(layer_count))
    layers = [int(layer.strip()) for layer in layers_arg.split(",") if layer.strip()]
    for layer in layers:
        if layer < 0 or layer >= layer_count:
            raise ValueError(f"Layer {layer} is out of range for {model_name} with {layer_count} layers")
    return layers


def experiment_file_name(model_key: str, dataset: str, mode: str, prompt_style: str) -> str:
    return f"{model_key}_{dataset}_{mode}_{prompt_style}.json"


def build_points(args: argparse.Namespace) -> list[dict[str, Any]]:
    points = []
    for dataset in DATASETS:
        rows = prompts.load_combined_questions(dataset, args.input_dir)
        if len(rows) < args.limit:
            raise ValueError(f"{dataset} only has {len(rows)} rows, cannot draw limit={args.limit}")
        rows = rows[: args.limit]
        for mode in MODES:
            n, use_qd, use_da = prompts.parse_mode(mode)
            for prompt_style in PROMPT_STYLES:
                explain = prompt_style == "explain"
                condition_rng = random.Random(args.seed)
                baseline_rng = random.Random(
                    f"{args.seed}:{experiment_file_name(args.model, dataset, mode, prompt_style)}"
                )
                experiment_id = f"{dataset}_{mode}_{prompt_style}"
                for question_idx, row in enumerate(rows):
                    prompt_info = prompts.build_experiment_prompt(
                        row,
                        dataset,
                        mode,
                        explain,
                        condition_rng,
                    )
                    condition_prompt = prompt_info["prompt"]
                    record = dict(row)
                    record.update(
                        {
                            "prompt": condition_prompt,
                            "main_wrong": prompt_info["main_wrong"],
                            "da_position": prompt_info["da_position"],
                            "da_answer": prompt_info["da_answer"],
                        }
                    )
                    random_baseline_answers = prompts.random_answers_for_record(
                        dataset, n, record, baseline_rng
                    )
                    if args.baseline == "same_prompt":
                        baseline_prompt = condition_prompt
                    else:
                        baseline_prompt = prompts.replace_prior_block(
                            condition_prompt,
                            dataset,
                            mode,
                            n,
                            random_baseline_answers,
                        )

                    base_metadata = {
                        "dataset": dataset,
                        "mode": mode,
                        "n": n,
                        "qd": use_qd,
                        "da": use_da,
                        "prompt_style": prompt_style,
                        "experiment_id": experiment_id,
                        "question_idx": question_idx,
                        "subject": row.get("subject"),
                        "ground_truth": prompt_info["ground_truth"],
                        "main_wrong": prompt_info["main_wrong"],
                        "da_position": prompt_info["da_position"],
                        "da_answer": prompt_info["da_answer"],
                        "random_baseline_answers": random_baseline_answers,
                    }
                    points.append(
                        {
                            **base_metadata,
                            "point_type": "condition",
                            "prompt": condition_prompt,
                        }
                    )
                    points.append(
                        {
                            **base_metadata,
                            "point_type": "baseline",
                            "prompt": baseline_prompt,
                        }
                    )
    return points


def metadata_without_prompt(point: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in point.items() if key != "prompt"}


def write_metadata(points: list[dict[str, Any]], output_dir: Path) -> None:
    metadata_file = output_dir / "metadata.jsonl"
    with metadata_file.open("w") as handle:
        for idx, point in enumerate(points):
            row = {"point_idx": idx, **metadata_without_prompt(point)}
            handle.write(json.dumps(row) + "\n")


def activation_file(activation_dir: Path, layer: int, dtype: str) -> Path:
    return activation_dir / f"layer_{layer:02d}.{dtype}.mmap"


def write_activation_manifest(
    manifest_file: Path,
    *,
    complete: bool,
    processed_count: int,
    model_key: str,
    model_name: str,
    layers: list[int],
    point_count: int,
    hidden_dim: int | None,
    dtype: str,
    args: argparse.Namespace,
) -> None:
    manifest = {
        "complete": complete,
        "processed_count": processed_count,
        "model": model_key,
        "model_name": model_name,
        "layers": layers,
        "point_count": point_count,
        "hidden_dim": hidden_dim,
        "dtype": dtype,
        "position": args.position,
        "activation_kind": args.activation_kind,
        "baseline": args.baseline,
        "seed": args.seed,
        "limit": args.limit,
    }
    manifest_file.write_text(json.dumps(manifest, indent=2))


def read_activation_manifest(manifest_file: Path) -> dict[str, Any] | None:
    if not manifest_file.exists():
        return None
    return json.loads(manifest_file.read_text())


def open_activation_maps(
    activation_dir: Path,
    layers: list[int],
    point_count: int,
    hidden_dim: int,
    dtype: str,
    mode: str,
) -> dict[int, np.memmap]:
    np_dtype = np.float16 if dtype == "float16" else np.float32
    return {
        layer: np.memmap(
            activation_file(activation_dir, layer, dtype),
            dtype=np_dtype,
            mode=mode,
            shape=(point_count, hidden_dim),
        )
        for layer in layers
    }


def collect_activation_cache(
    points: list[dict[str, Any]],
    model_key: str,
    model_name: str,
    layers: list[int],
    args: argparse.Namespace,
    output_dir: Path,
) -> dict[str, Any]:
    activation_dir = output_dir / "activations"
    activation_dir.mkdir(parents=True, exist_ok=True)
    manifest_file = activation_dir / "activation_manifest.json"
    manifest = None if args.force_recompute else read_activation_manifest(manifest_file)
    if manifest and manifest.get("complete"):
        return manifest
    if args.skip_collection:
        if manifest is None:
            raise ValueError("--skip-collection was set but no activation cache manifest exists")
        return manifest

    print(f"Loading {model_key}: {model_name}")
    model, tokenizer, device = load_model(model_key)
    processed_count = int(manifest.get("processed_count", 0)) if manifest else 0
    hidden_dim = int(manifest["hidden_dim"]) if manifest and manifest.get("hidden_dim") else None
    maps = None

    if processed_count > 0:
        maps = open_activation_maps(
            activation_dir,
            layers,
            len(points),
            hidden_dim,
            args.cache_dtype,
            "r+",
        )

    try:
        start_idx = processed_count
        if start_idx == 0:
            first_acts, _, _ = collect_activations(
                points[0]["prompt"],
                model,
                tokenizer,
                device,
                layers,
                args.activation_kind,
                args.position,
            )
            hidden_dim = int(first_acts[layers[0]].numel())
            maps = open_activation_maps(
                activation_dir,
                layers,
                len(points),
                hidden_dim,
                args.cache_dtype,
                "w+",
            )
            for layer in layers:
                maps[layer][0] = first_acts[layer].to(torch.float32).numpy()
            processed_count = 1
            write_activation_manifest(
                manifest_file,
                complete=False,
                processed_count=processed_count,
                model_key=model_key,
                model_name=model_name,
                layers=layers,
                point_count=len(points),
                hidden_dim=hidden_dim,
                dtype=args.cache_dtype,
                args=args,
            )
            start_idx = 1

        for point_idx in tqdm(range(start_idx, len(points)), desc="Collecting assistant-start activations"):
            acts, _, _ = collect_activations(
                points[point_idx]["prompt"],
                model,
                tokenizer,
                device,
                layers,
                args.activation_kind,
                args.position,
            )
            for layer in layers:
                maps[layer][point_idx] = acts[layer].to(torch.float32).numpy()
            if (point_idx + 1) % 100 == 0:
                for mmap in maps.values():
                    mmap.flush()
                write_activation_manifest(
                    manifest_file,
                    complete=False,
                    processed_count=point_idx + 1,
                    model_key=model_key,
                    model_name=model_name,
                    layers=layers,
                    point_count=len(points),
                    hidden_dim=hidden_dim,
                    dtype=args.cache_dtype,
                    args=args,
                )
    finally:
        if maps:
            for mmap in maps.values():
                mmap.flush()
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    write_activation_manifest(
        manifest_file,
        complete=True,
        processed_count=len(points),
        model_key=model_key,
        model_name=model_name,
        layers=layers,
        point_count=len(points),
        hidden_dim=hidden_dim,
        dtype=args.cache_dtype,
        args=args,
    )
    return json.loads(manifest_file.read_text())


def pca_device(args: argparse.Namespace) -> torch.device:
    if args.pca_device == "cuda":
        return torch.device("cuda")
    if args.pca_device == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def fit_pca_scores(
    layer_matrix: np.ndarray,
    indices: list[int],
    device: torch.device,
    niter: int,
) -> tuple[np.ndarray, list[float]]:
    if len(indices) == layer_matrix.shape[0] and indices[0] == 0 and indices[-1] == len(indices) - 1:
        data = np.array(layer_matrix, copy=True)
    else:
        data = np.array(layer_matrix[indices], copy=True)
    x = torch.from_numpy(data).to(device=device, dtype=torch.float32)
    x_mean = x.mean(dim=0, keepdim=True)
    x_centered = x - x_mean
    _, singular_values, components = torch.pca_lowrank(x, q=2, center=True, niter=niter)
    scores = x_centered @ components[:, :2]
    total_variance = torch.sum(x_centered * x_centered)
    explained = ((singular_values[:2] ** 2) / total_variance).detach().cpu().tolist()
    scores_np = scores.detach().cpu().numpy()
    del x, x_mean, x_centered, scores
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return scores_np, [float(value) for value in explained]


def scope_indices(points: list[dict[str, Any]], scope: str) -> list[int]:
    if scope == "combined":
        return list(range(len(points)))
    return [idx for idx, point in enumerate(points) if point["dataset"] == scope]


def category_values(points: list[dict[str, Any]], indices: list[int], key: str) -> list[str]:
    return [str(points[idx][key]) for idx in indices]


def draw_rect(image: np.ndarray, x0: int, y0: int, x1: int, y1: int, color: tuple[int, int, int]) -> None:
    h, w, _ = image.shape
    x0 = max(0, min(w, x0))
    x1 = max(0, min(w, x1))
    y0 = max(0, min(h, y0))
    y1 = max(0, min(h, y1))
    if x1 > x0 and y1 > y0:
        image[y0:y1, x0:x1] = color


def draw_line(
    image: np.ndarray,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    color: tuple[int, int, int],
) -> None:
    steps = max(abs(x1 - x0), abs(y1 - y0), 1)
    for step in range(steps + 1):
        t = step / steps
        x = int(round(x0 + (x1 - x0) * t))
        y = int(round(y0 + (y1 - y0) * t))
        if 0 <= y < image.shape[0] and 0 <= x < image.shape[1]:
            image[y, x] = color


def draw_text(
    image: np.ndarray,
    x: int,
    y: int,
    text: str,
    color: tuple[int, int, int] = (30, 30, 30),
    scale: int = 2,
) -> None:
    cursor_x = x
    for char in text.upper():
        pattern = FONT_5X7.get(char, FONT_5X7[" "])
        for row_idx, row in enumerate(pattern):
            for col_idx, value in enumerate(row):
                if value == "1":
                    draw_rect(
                        image,
                        cursor_x + col_idx * scale,
                        y + row_idx * scale,
                        cursor_x + (col_idx + 1) * scale,
                        y + (row_idx + 1) * scale,
                        color,
                    )
        cursor_x += 6 * scale


def blend_point(
    image: np.ndarray,
    x: int,
    y: int,
    color: tuple[int, int, int],
    radius: int,
    alpha: float,
) -> None:
    h, w, _ = image.shape
    color_arr = np.asarray(color, dtype=np.float32)
    for yy in range(max(0, y - radius), min(h, y + radius + 1)):
        for xx in range(max(0, x - radius), min(w, x + radius + 1)):
            if (xx - x) * (xx - x) + (yy - y) * (yy - y) <= radius * radius:
                old = image[yy, xx].astype(np.float32)
                image[yy, xx] = np.clip(old * (1.0 - alpha) + color_arr * alpha, 0, 255)


def render_plot(
    scores: np.ndarray,
    labels: list[str],
    palette: dict[str, tuple[int, int, int]],
    title: str,
    explained: list[float],
    width: int,
    height: int,
    point_radius: int,
) -> np.ndarray:
    image = np.full((height, width, 3), 255, dtype=np.uint8)
    plot_left = 85
    plot_right = width - 270
    plot_top = 90
    plot_bottom = height - 95
    axis_color = (90, 90, 90)
    grid_color = (225, 225, 225)

    draw_text(image, 40, 26, title[:95], (25, 25, 25), 2)
    draw_text(
        image,
        plot_left,
        height - 48,
        f"PC1 {explained[0] * 100:.1f}%",
        (55, 55, 55),
        2,
    )
    draw_text(
        image,
        15,
        plot_top - 34,
        f"PC2 {explained[1] * 100:.1f}%",
        (55, 55, 55),
        2,
    )

    draw_rect(image, plot_left, plot_top, plot_right, plot_bottom, (250, 250, 250))
    for frac in [0.25, 0.5, 0.75]:
        x = int(plot_left + (plot_right - plot_left) * frac)
        y = int(plot_top + (plot_bottom - plot_top) * frac)
        draw_line(image, x, plot_top, x, plot_bottom, grid_color)
        draw_line(image, plot_left, y, plot_right, y, grid_color)
    draw_line(image, plot_left, plot_bottom, plot_right, plot_bottom, axis_color)
    draw_line(image, plot_left, plot_top, plot_left, plot_bottom, axis_color)
    draw_line(image, plot_right, plot_top, plot_right, plot_bottom, axis_color)
    draw_line(image, plot_left, plot_top, plot_right, plot_top, axis_color)

    x_values = scores[:, 0]
    y_values = scores[:, 1]
    x_min, x_max = np.percentile(x_values, [0.5, 99.5])
    y_min, y_max = np.percentile(y_values, [0.5, 99.5])
    if math.isclose(float(x_min), float(x_max)):
        x_min -= 1.0
        x_max += 1.0
    if math.isclose(float(y_min), float(y_max)):
        y_min -= 1.0
        y_max += 1.0
    x_pad = (x_max - x_min) * 0.06
    y_pad = (y_max - y_min) * 0.06
    x_min -= x_pad
    x_max += x_pad
    y_min -= y_pad
    y_max += y_pad

    xs = np.clip((x_values - x_min) / (x_max - x_min), 0, 1)
    ys = np.clip((y_values - y_min) / (y_max - y_min), 0, 1)
    pixel_x = (plot_left + xs * (plot_right - plot_left)).astype(np.int32)
    pixel_y = (plot_bottom - ys * (plot_bottom - plot_top)).astype(np.int32)
    for x, y, label in zip(pixel_x, pixel_y, labels):
        blend_point(image, int(x), int(y), palette.get(label, (80, 80, 80)), point_radius, 0.65)

    legend_x = plot_right + 25
    legend_y = plot_top
    draw_text(image, legend_x, legend_y - 35, "LEGEND", (35, 35, 35), 2)
    for idx, label in enumerate(sorted(set(labels), key=lambda item: list(palette).index(item) if item in palette else item)):
        y = legend_y + idx * 28
        color = palette.get(label, (80, 80, 80))
        draw_rect(image, legend_x, y, legend_x + 18, y + 18, color)
        draw_text(image, legend_x + 28, y + 2, label, (35, 35, 35), 2)

    return image


def save_png(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width, _ = image.shape

    def chunk(kind: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + kind
            + data
            + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
        )

    raw = b"".join(b"\x00" + image[row].tobytes() for row in range(height))
    png = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw, level=6))
        + chunk(b"IEND", b"")
    )
    path.write_bytes(png)


def write_pdf(path: Path, pages: list[np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    objects: dict[int, bytes] = {}
    objects[1] = b"<< /Type /Catalog /Pages 2 0 R >>"
    next_id = 3
    page_ids = []
    page_width = 792
    page_height = 612

    for image in pages:
        height, width, _ = image.shape
        image_id = next_id
        next_id += 1
        stream = zlib.compress(image.tobytes(), level=6)
        objects[image_id] = (
            f"<< /Type /XObject /Subtype /Image /Width {width} /Height {height} "
            f"/ColorSpace /DeviceRGB /BitsPerComponent 8 /Filter /FlateDecode "
            f"/Length {len(stream)} >>\nstream\n".encode()
            + stream
            + b"\nendstream"
        )

        scale = min(page_width / width, page_height / height)
        draw_w = width * scale
        draw_h = height * scale
        offset_x = (page_width - draw_w) / 2
        offset_y = (page_height - draw_h) / 2
        content = f"q\n{draw_w:.3f} 0 0 {draw_h:.3f} {offset_x:.3f} {offset_y:.3f} cm\n/Im0 Do\nQ\n".encode()
        content_id = next_id
        next_id += 1
        objects[content_id] = (
            f"<< /Length {len(content)} >>\nstream\n".encode()
            + content
            + b"endstream"
        )

        page_id = next_id
        next_id += 1
        page_ids.append(page_id)
        objects[page_id] = (
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {page_width} {page_height}] "
            f"/Resources << /XObject << /Im0 {image_id} 0 R >> >> "
            f"/Contents {content_id} 0 R >>"
        ).encode()

    kids = " ".join(f"{page_id} 0 R" for page_id in page_ids)
    objects[2] = f"<< /Type /Pages /Kids [{kids}] /Count {len(page_ids)} >>".encode()

    with path.open("wb") as handle:
        handle.write(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
        offsets = {}
        for obj_id in sorted(objects):
            offsets[obj_id] = handle.tell()
            handle.write(f"{obj_id} 0 obj\n".encode())
            handle.write(objects[obj_id])
            handle.write(b"\nendobj\n")
        xref_offset = handle.tell()
        max_id = max(objects)
        handle.write(f"xref\n0 {max_id + 1}\n".encode())
        handle.write(b"0000000000 65535 f \n")
        for obj_id in range(1, max_id + 1):
            handle.write(f"{offsets[obj_id]:010d} 00000 n \n".encode())
        handle.write(
            f"trailer\n<< /Size {max_id + 1} /Root 1 0 R >>\nstartxref\n{xref_offset}\n%%EOF\n".encode()
        )


def rgb_to_hex(color: tuple[int, int, int]) -> str:
    return "#" + "".join(f"{channel:02x}" for channel in color)


def score_bounds(scores: np.ndarray) -> dict[str, float]:
    x_values = scores[:, 0]
    y_values = scores[:, 1]
    x_min, x_max = np.percentile(x_values, [0.5, 99.5])
    y_min, y_max = np.percentile(y_values, [0.5, 99.5])
    if math.isclose(float(x_min), float(x_max)):
        x_min -= 1.0
        x_max += 1.0
    if math.isclose(float(y_min), float(y_max)):
        y_min -= 1.0
        y_max += 1.0
    x_pad = (x_max - x_min) * 0.06
    y_pad = (y_max - y_min) * 0.06
    return {
        "x_min": round(float(x_min - x_pad), 6),
        "x_max": round(float(x_max + x_pad), 6),
        "y_min": round(float(y_min - y_pad), 6),
        "y_max": round(float(y_max + y_pad), 6),
    }


def compact_scores(scores: np.ndarray) -> list[list[float]]:
    return [
        [round(float(row[0]), 5), round(float(row[1]), 5)]
        for row in scores
    ]


def html_metadata(points: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keys = [
        "dataset",
        "mode",
        "n",
        "qd",
        "da",
        "prompt_style",
        "experiment_id",
        "question_idx",
        "subject",
        "ground_truth",
        "main_wrong",
        "da_position",
        "da_answer",
        "point_type",
    ]
    return [
        {"point_idx": idx, **{key: point.get(key) for key in keys}}
        for idx, point in enumerate(points)
    ]


def write_html_viewer(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Layer Sweep PCA Viewer</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f6f7f9;
      --panel: #ffffff;
      --ink: #1f2933;
      --muted: #6b7280;
      --line: #d7dce2;
      --strong: #111827;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    header {{
      display: flex;
      align-items: baseline;
      justify-content: space-between;
      gap: 24px;
      padding: 18px 24px 12px;
      border-bottom: 1px solid var(--line);
      background: var(--panel);
    }}
    h1 {{
      margin: 0;
      font-size: 20px;
      font-weight: 650;
      letter-spacing: 0;
    }}
    .meta {{
      color: var(--muted);
      font-size: 13px;
      white-space: nowrap;
    }}
    main {{
      padding: 16px 20px 22px;
      display: grid;
      gap: 14px;
    }}
    .toolbar {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px 16px;
      align-items: end;
      padding: 12px 14px;
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
    }}
    label {{
      display: grid;
      gap: 5px;
      color: var(--muted);
      font-size: 12px;
      font-weight: 600;
    }}
    select, button {{
      font: inherit;
    }}
    select {{
      min-width: 140px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #fff;
      color: var(--ink);
      padding: 7px 30px 7px 9px;
    }}
    .viewer {{
      display: grid;
      grid-template-columns: minmax(620px, 1fr) 300px;
      gap: 14px;
      align-items: stretch;
      min-height: 720px;
    }}
    .plot-shell {{
      position: relative;
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      min-height: 720px;
      overflow: hidden;
    }}
    canvas {{
      display: block;
      width: 100%;
      height: 100%;
      min-height: 720px;
    }}
    aside {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
      overflow: auto;
      max-height: 720px;
    }}
    h2 {{
      margin: 0 0 10px;
      font-size: 14px;
      letter-spacing: 0;
    }}
    .legend {{
      display: grid;
      gap: 7px;
    }}
    .legend-button {{
      display: grid;
      grid-template-columns: 14px 1fr auto;
      align-items: center;
      gap: 8px;
      width: 100%;
      padding: 7px 8px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #fff;
      color: var(--ink);
      cursor: default;
      text-align: left;
    }}
    .legend-button.is-muted {{
      color: #9ca3af;
      background: #f4f5f7;
    }}
    .swatch {{
      width: 12px;
      height: 12px;
      border-radius: 50%;
      box-shadow: inset 0 0 0 1px rgba(0, 0, 0, 0.18);
    }}
    .count {{
      color: var(--muted);
      font-variant-numeric: tabular-nums;
    }}
    .summary {{
      margin-top: 14px;
      padding-top: 12px;
      border-top: 1px solid var(--line);
      color: var(--muted);
      display: grid;
      gap: 6px;
      font-size: 12px;
    }}
    .tooltip {{
      position: absolute;
      pointer-events: none;
      display: none;
      width: max-content;
      max-width: 320px;
      padding: 8px 9px;
      border-radius: 6px;
      border: 1px solid #cbd5e1;
      background: rgba(255, 255, 255, 0.96);
      box-shadow: 0 8px 22px rgba(15, 23, 42, 0.14);
      font-size: 12px;
      color: var(--ink);
      z-index: 5;
    }}
    @media (max-width: 980px) {{
      .viewer {{
        grid-template-columns: 1fr;
      }}
      aside {{
        max-height: none;
      }}
      .plot-shell, canvas {{
        min-height: 560px;
      }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>Layer Sweep PCA Viewer</h1>
    <div class="meta" id="headerMeta"></div>
  </header>
  <main>
    <section class="toolbar">
      <label>Layer<select id="layerSelect"></select></label>
      <label>Scope<select id="scopeSelect"></select></label>
      <label>Color By<select id="colorSelect"></select></label>
    </section>
    <section class="viewer">
      <div class="plot-shell">
        <canvas id="plotCanvas"></canvas>
        <div class="tooltip" id="tooltip"></div>
      </div>
      <aside>
        <h2 id="legendTitle">Labels</h2>
        <div class="legend" id="legend"></div>
        <div class="summary" id="summary"></div>
      </aside>
    </section>
  </main>
  <script>
    const DATA = {payload};
    const canvas = document.getElementById("plotCanvas");
    const ctx = canvas.getContext("2d");
    const tooltip = document.getElementById("tooltip");
    const layerSelect = document.getElementById("layerSelect");
    const scopeSelect = document.getElementById("scopeSelect");
    const colorSelect = document.getElementById("colorSelect");
    const legend = document.getElementById("legend");
    const summary = document.getElementById("summary");
    const legendTitle = document.getElementById("legendTitle");
    const headerMeta = document.getElementById("headerMeta");
    const state = {{
      layer: DATA.layers[0],
      scope: DATA.scopes[0],
      colorBy: DATA.plot_by.includes("mode") ? "mode" : DATA.plot_by[0],
      highlight: null,
    }};
    let screenPoints = [];

    function titleCase(value) {{
      return String(value).replace(/_/g, " ").replace(/\\b\\w/g, match => match.toUpperCase());
    }}

    function colorFor(key, value) {{
      return (DATA.palettes[key] && DATA.palettes[key][String(value)]) || "#4b5563";
    }}

    function hexToRgb(hex) {{
      const clean = hex.replace("#", "");
      return [
        parseInt(clean.slice(0, 2), 16),
        parseInt(clean.slice(2, 4), 16),
        parseInt(clean.slice(4, 6), 16),
      ];
    }}

    function rgba(hex, alpha) {{
      const [r, g, b] = hexToRgb(hex);
      return `rgba(${{r}}, ${{g}}, ${{b}}, ${{alpha}})`;
    }}

    function scopeIndices(scope) {{
      return DATA.scope_indices[scope];
    }}

    function layerData() {{
      return DATA.layer_scores[state.layer][state.scope];
    }}

    function labelValue(meta, key) {{
      const value = meta[key];
      return value === null || value === undefined ? "none" : String(value);
    }}

    function paletteOrder(key) {{
      return Object.keys(DATA.palettes[key] || {{}});
    }}

    function sortedValues(key, counts) {{
      const order = paletteOrder(key);
      return Object.keys(counts).sort((a, b) => {{
        const ai = order.indexOf(a);
        const bi = order.indexOf(b);
        if (ai !== -1 || bi !== -1) {{
          if (ai === -1) return 1;
          if (bi === -1) return -1;
          return ai - bi;
        }}
        return a.localeCompare(b, undefined, {{ numeric: true }});
      }});
    }}

    function currentCounts() {{
      const counts = {{}};
      for (const pointIdx of scopeIndices(state.scope)) {{
        const label = labelValue(DATA.metadata[pointIdx], state.colorBy);
        counts[label] = (counts[label] || 0) + 1;
      }}
      return counts;
    }}

    function configureControls() {{
      for (const layer of DATA.layers) {{
        const option = document.createElement("option");
        option.value = layer;
        option.textContent = `Layer ${{String(layer).padStart(2, "0")}}`;
        layerSelect.appendChild(option);
      }}
      for (const scope of DATA.scopes) {{
        const option = document.createElement("option");
        option.value = scope;
        option.textContent = titleCase(scope);
        scopeSelect.appendChild(option);
      }}
      for (const key of DATA.plot_by) {{
        const option = document.createElement("option");
        option.value = key;
        option.textContent = titleCase(key);
        colorSelect.appendChild(option);
      }}
      layerSelect.value = state.layer;
      scopeSelect.value = state.scope;
      colorSelect.value = state.colorBy;
    }}

    function renderLegend() {{
      const counts = currentCounts();
      const values = sortedValues(state.colorBy, counts);
      legendTitle.textContent = titleCase(state.colorBy);
      legend.innerHTML = "";
      for (const value of values) {{
        const button = document.createElement("button");
        button.type = "button";
        button.className = "legend-button";
        button.dataset.label = value;
        const swatch = document.createElement("span");
        swatch.className = "swatch";
        swatch.style.background = colorFor(state.colorBy, value);
        const label = document.createElement("span");
        label.textContent = value;
        const count = document.createElement("span");
        count.className = "count";
        count.textContent = counts[value].toLocaleString();
        button.append(swatch, label, count);
        button.addEventListener("mouseenter", () => {{
          state.highlight = value;
          updateLegendState();
          draw();
        }});
        button.addEventListener("mouseleave", () => {{
          state.highlight = null;
          updateLegendState();
          draw();
        }});
        legend.appendChild(button);
      }}
      updateLegendState();
    }}

    function updateLegendState() {{
      for (const button of legend.querySelectorAll(".legend-button")) {{
        button.classList.toggle(
          "is-muted",
          state.highlight !== null && button.dataset.label !== state.highlight
        );
      }}
    }}

    function updateSummary() {{
      const layer = layerData();
      const ev = layer.explained_variance_ratio;
      const indices = scopeIndices(state.scope);
      headerMeta.textContent = `${{DATA.model.toUpperCase()}} | ${{DATA.point_count.toLocaleString()}} points`;
      summary.innerHTML = `
        <div>Scope: <strong>${{titleCase(state.scope)}}</strong></div>
        <div>Layer: <strong>${{String(state.layer).padStart(2, "0")}}</strong></div>
        <div>Points: <strong>${{indices.length.toLocaleString()}}</strong></div>
        <div>PC1: <strong>${{(ev[0] * 100).toFixed(2)}}%</strong></div>
        <div>PC2: <strong>${{(ev[1] * 100).toFixed(2)}}%</strong></div>
      `;
    }}

    function resizeCanvas() {{
      const rect = canvas.getBoundingClientRect();
      const dpr = window.devicePixelRatio || 1;
      const width = Math.max(640, Math.floor(rect.width));
      const height = Math.max(480, Math.floor(rect.height));
      if (canvas.width !== Math.floor(width * dpr) || canvas.height !== Math.floor(height * dpr)) {{
        canvas.width = Math.floor(width * dpr);
        canvas.height = Math.floor(height * dpr);
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      }}
      return {{ width, height }};
    }}

    function drawPoint(x, y, radius, fill) {{
      ctx.beginPath();
      ctx.arc(x, y, radius, 0, Math.PI * 2);
      ctx.fillStyle = fill;
      ctx.fill();
    }}

    function draw() {{
      const {{ width, height }} = resizeCanvas();
      const plot = layerData();
      const bounds = plot.bounds;
      const scores = plot.scores;
      const indices = scopeIndices(state.scope);
      const left = 72;
      const right = width - 34;
      const top = 44;
      const bottom = height - 62;
      screenPoints = [];

      ctx.clearRect(0, 0, width, height);
      ctx.fillStyle = "#ffffff";
      ctx.fillRect(0, 0, width, height);
      ctx.fillStyle = "#f8fafc";
      ctx.fillRect(left, top, right - left, bottom - top);
      ctx.strokeStyle = "#d7dce2";
      ctx.lineWidth = 1;
      for (const frac of [0.25, 0.5, 0.75]) {{
        const x = left + (right - left) * frac;
        const y = top + (bottom - top) * frac;
        ctx.beginPath();
        ctx.moveTo(x, top);
        ctx.lineTo(x, bottom);
        ctx.moveTo(left, y);
        ctx.lineTo(right, y);
        ctx.stroke();
      }}
      ctx.strokeStyle = "#6b7280";
      ctx.strokeRect(left, top, right - left, bottom - top);

      ctx.fillStyle = "#111827";
      ctx.font = "600 15px Inter, system-ui, sans-serif";
      ctx.fillText(
        `${{DATA.model.toUpperCase()}} ${{state.scope.toUpperCase()}} layer ${{String(state.layer).padStart(2, "0")}} by ${{titleCase(state.colorBy)}}`,
        left,
        24
      );
      ctx.fillStyle = "#6b7280";
      ctx.font = "12px Inter, system-ui, sans-serif";
      ctx.fillText(`PC1 ${{(plot.explained_variance_ratio[0] * 100).toFixed(1)}}%`, left, height - 24);
      ctx.save();
      ctx.translate(20, top + 86);
      ctx.rotate(-Math.PI / 2);
      ctx.fillText(`PC2 ${{(plot.explained_variance_ratio[1] * 100).toFixed(1)}}%`, 0, 0);
      ctx.restore();

      const denominatorX = bounds.x_max - bounds.x_min || 1;
      const denominatorY = bounds.y_max - bounds.y_min || 1;
      function project(score) {{
        const x = left + Math.min(1, Math.max(0, (score[0] - bounds.x_min) / denominatorX)) * (right - left);
        const y = bottom - Math.min(1, Math.max(0, (score[1] - bounds.y_min) / denominatorY)) * (bottom - top);
        return [x, y];
      }}

      const highlighted = [];
      for (let i = 0; i < scores.length; i += 1) {{
        const pointIdx = indices[i];
        const meta = DATA.metadata[pointIdx];
        const label = labelValue(meta, state.colorBy);
        const [x, y] = project(scores[i]);
        screenPoints.push({{ x, y, pointIdx, label }});
        const isHighlighted = state.highlight !== null && label === state.highlight;
        if (isHighlighted) {{
          highlighted.push([x, y, label]);
        }} else {{
          const color = state.highlight === null ? colorFor(state.colorBy, label) : "#9ca3af";
          drawPoint(x, y, state.highlight === null ? 2.1 : 1.7, rgba(color, state.highlight === null ? 0.68 : 0.22));
        }}
      }}
      for (const [x, y, label] of highlighted) {{
        drawPoint(x, y, 3.6, rgba(colorFor(state.colorBy, label), 0.92));
        ctx.strokeStyle = "rgba(17, 24, 39, 0.28)";
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.arc(x, y, 4.2, 0, Math.PI * 2);
        ctx.stroke();
      }}
    }}

    function refresh() {{
      state.highlight = null;
      renderLegend();
      updateSummary();
      draw();
    }}

    function nearestPoint(clientX, clientY) {{
      const rect = canvas.getBoundingClientRect();
      const x = clientX - rect.left;
      const y = clientY - rect.top;
      let best = null;
      let bestDistance = 64;
      for (const point of screenPoints) {{
        const dx = point.x - x;
        const dy = point.y - y;
        const distance = dx * dx + dy * dy;
        if (distance < bestDistance) {{
          bestDistance = distance;
          best = point;
        }}
      }}
      return best;
    }}

    canvas.addEventListener("mousemove", event => {{
      const point = nearestPoint(event.clientX, event.clientY);
      if (!point) {{
        tooltip.style.display = "none";
        return;
      }}
      const meta = DATA.metadata[point.pointIdx];
      tooltip.innerHTML = `
        <strong>${{meta.experiment_id}}</strong><br>
        point ${{meta.point_idx}} | question ${{meta.question_idx}} | ${{meta.point_type}}<br>
        dataset ${{meta.dataset}} | mode ${{meta.mode}} | ${{meta.prompt_style}}
      `;
      const shellRect = canvas.parentElement.getBoundingClientRect();
      tooltip.style.left = `${{event.clientX - shellRect.left + 12}}px`;
      tooltip.style.top = `${{event.clientY - shellRect.top + 12}}px`;
      tooltip.style.display = "block";
    }});
    canvas.addEventListener("mouseleave", () => {{
      tooltip.style.display = "none";
    }});

    layerSelect.addEventListener("change", () => {{
      state.layer = layerSelect.value;
      refresh();
    }});
    scopeSelect.addEventListener("change", () => {{
      state.scope = scopeSelect.value;
      refresh();
    }});
    colorSelect.addEventListener("change", () => {{
      state.colorBy = colorSelect.value;
      refresh();
    }});
    window.addEventListener("resize", draw);

    configureControls();
    refresh();
  </script>
</body>
</html>
"""
    path.write_text(html)


def run_pca_and_plots(
    points: list[dict[str, Any]],
    layers: list[int],
    manifest: dict[str, Any],
    args: argparse.Namespace,
    output_dir: Path,
) -> dict[str, Any]:
    activation_dir = output_dir / "activations"
    plot_dir = output_dir / "pngs"
    pdf_pages = []
    summary: dict[str, Any] = {
        "model": args.model,
        "model_name": MODEL_IDS[args.model],
        "point_count": len(points),
        "layers": {},
        "scopes": args.scopes.split(","),
        "plot_by": args.plot_by.split(","),
        "activation_manifest": str(activation_dir / "activation_manifest.json"),
    }
    maps = open_activation_maps(
        activation_dir,
        layers,
        len(points),
        int(manifest["hidden_dim"]),
        manifest["dtype"],
        "r",
    )
    device = pca_device(args)
    scopes = [scope.strip() for scope in args.scopes.split(",") if scope.strip()]
    plot_keys = [key.strip() for key in args.plot_by.split(",") if key.strip()]
    scope_index_map = {scope: scope_indices(points, scope) for scope in scopes}
    html_data: dict[str, Any] | None = None
    if not args.skip_html:
        html_data = {
            "model": args.model,
            "model_name": MODEL_IDS[args.model],
            "point_count": len(points),
            "layers": [str(layer) for layer in layers],
            "scopes": scopes,
            "plot_by": plot_keys,
            "metadata": html_metadata(points),
            "scope_indices": scope_index_map,
            "palettes": {
                key: {label: rgb_to_hex(color) for label, color in palette.items()}
                for key, palette in PALETTES.items()
            },
            "layer_scores": {},
        }

    for layer in tqdm(layers, desc="Running per-layer PCA"):
        layer_summary = {}
        for scope in scopes:
            indices = scope_index_map[scope]
            scores, explained = fit_pca_scores(maps[layer], indices, device, args.pca_niter)
            scope_summary = {
                "point_count": len(indices),
                "explained_variance_ratio": explained,
                "plots": {},
            }
            if html_data is not None:
                html_data["layer_scores"].setdefault(str(layer), {})[scope] = {
                    "scores": compact_scores(scores),
                    "bounds": score_bounds(scores),
                    "explained_variance_ratio": [round(float(value), 8) for value in explained],
                }
            for plot_key in plot_keys:
                png_file = (
                    plot_dir
                    / scope
                    / f"by_{plot_key}"
                    / f"layer_{layer:02d}_{scope}_by_{plot_key}.png"
                )
                if not args.skip_static_plots:
                    labels = category_values(points, indices, plot_key)
                    title = (
                        f"{args.model.upper()} {scope.upper()} LAYER {layer:02d} "
                        f"BY {plot_key.upper().replace('_', ' ')}"
                    )
                    image = render_plot(
                        scores,
                        labels,
                        PALETTES[plot_key],
                        title,
                        explained,
                        args.width,
                        args.height,
                        args.point_radius,
                    )
                    save_png(png_file, image)
                    pdf_pages.append(image)
                scope_summary["plots"][plot_key] = str(png_file)
            layer_summary[scope] = scope_summary
        summary["layers"][str(layer)] = layer_summary

    pdf_file = output_dir / "layer_sweep_pca_all_plots.pdf"
    if not args.skip_static_plots:
        write_pdf(pdf_file, pdf_pages)
    summary["pdf_file"] = str(pdf_file)
    if html_data is not None:
        html_file = output_dir / "layer_sweep_pca_viewer.html"
        write_html_viewer(html_file, html_data)
        summary["html_file"] = str(html_file)
    summary_file = output_dir / "pca_summary.json"
    summary_file.write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    args = parse_args()
    model_name = MODEL_IDS[args.model]
    output_dir = args.output_dir / args.model
    output_dir.mkdir(parents=True, exist_ok=True)
    layers = select_layers(model_name, args.layers)
    points = build_points(args)
    write_metadata(points, output_dir)

    manifest = {
        "model": args.model,
        "model_name": model_name,
        "point_count": len(points),
        "layers": layers,
        "limit": args.limit,
        "datasets": DATASETS,
        "modes": MODES,
        "prompt_styles": PROMPT_STYLES,
        "baseline": args.baseline,
        "position": args.position,
        "activation_kind": args.activation_kind,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    if args.dry_run:
        print(json.dumps(manifest, indent=2))
        return

    activation_manifest = collect_activation_cache(points, args.model, model_name, layers, args, output_dir)
    if args.skip_plots:
        return
    summary = run_pca_and_plots(points, layers, activation_manifest, args, output_dir)
    print(f"PCA summary saved to {output_dir / 'pca_summary.json'}")
    print(f"Multi-page PDF saved to {summary['pdf_file']}")
    if summary.get("html_file"):
        print(f"Interactive HTML saved to {summary['html_file']}")


if __name__ == "__main__":
    main()
