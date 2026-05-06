"""Build an HTML report of response-token projection curves by conformity group."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm


GROUPS = ["true", "false", "none"]
GROUP_LABELS = {
    "true": "conforms=True",
    "false": "conforms=False",
    "none": "conforms=None",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=Path("activation_cache"),
        help="Root containing activation cache manifests.",
    )
    parser.add_argument(
        "--diffmeans-path",
        type=Path,
        default=Path(
            "layer_sweep_results/qwen/"
            "layer_sweep_qwen_mmlu_mode_8_base_residual_assistant_start_"
            "all_wrong_minus_random_answers_n8_100_diffmeans.pt"
        ),
        help="Layer-wise diffmean direction to project onto.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("activation_cache/projection_report.html"),
    )
    return parser.parse_args()


def group_key(value: Any) -> str:
    if value is True:
        return "true"
    if value is False:
        return "false"
    return "none"


def init_series(max_len: int) -> dict[str, list[float]]:
    return {
        "count": [0] * max_len,
        "sum": [0.0] * max_len,
        "sum_sq": [0.0] * max_len,
    }


def ensure_len(series: dict[str, list[float]], length: int) -> None:
    current = len(series["count"])
    if length <= current:
        return
    extra = length - current
    series["count"].extend([0] * extra)
    series["sum"].extend([0.0] * extra)
    series["sum_sq"].extend([0.0] * extra)


def add_values(series: dict[str, list[float]], values: torch.Tensor) -> None:
    vals = values.detach().float().cpu().tolist()
    ensure_len(series, len(vals))
    for i, value in enumerate(vals):
        series["count"][i] += 1
        series["sum"][i] += float(value)
        series["sum_sq"][i] += float(value) * float(value)


def finish_series(series: dict[str, list[float]]) -> dict[str, list[float | int | None]]:
    mean: list[float | None] = []
    std: list[float | None] = []
    for count, total, total_sq in zip(series["count"], series["sum"], series["sum_sq"]):
        if count == 0:
            mean.append(None)
            std.append(None)
        else:
            m = total / count
            var = max(0.0, total_sq / count - m * m)
            mean.append(m)
            std.append(math.sqrt(var))
    return {
        "count": series["count"],
        "mean": mean,
        "std": std,
    }


def load_diffmeans(path: Path) -> tuple[dict[int, torch.Tensor], dict[str, Any]]:
    obj = torch.load(path, map_location="cpu", weights_only=False)
    units = {}
    for layer, vector in obj["diffmeans"].items():
        vec = vector.detach().float().cpu().squeeze()
        norm = vec.norm()
        if norm > 0:
            units[int(layer)] = vec / norm
    meta = {
        key: obj.get(key)
        for key in [
            "dataset",
            "mode",
            "prompt_style",
            "position",
            "activation_kind",
            "baseline",
            "direction",
            "sample_count",
            "experiment_file",
        ]
    }
    return units, meta


def build_data(cache_root: Path, diffmeans_path: Path) -> dict[str, Any]:
    units, diff_meta = load_diffmeans(diffmeans_path)
    experiments = []

    manifests = sorted(cache_root.rglob("manifest.json"))
    for manifest_path in manifests:
        cache_dir = manifest_path.parent
        manifest = json.loads(manifest_path.read_text())
        layers = [int(layer) for layer in manifest["layers"] if int(layer) in units]
        if not layers:
            continue
        accum = {
            str(layer): {group: init_series(0) for group in GROUPS}
            for layer in layers
        }
        record_counts = {group: 0 for group in GROUPS}
        token_counts = {group: 0 for group in GROUPS}

        for record in tqdm(manifest["records"], desc=str(cache_dir)):
            shard = torch.load(cache_dir / record["shard"], map_location="cpu", weights_only=False)
            group = group_key(record.get("conforms"))
            record_counts[group] += 1
            response_len = int(shard["response_len"])
            token_counts[group] += response_len
            for layer in layers:
                acts = shard["activations_by_layer"][layer].float()
                projections = acts @ units[layer]
                add_values(accum[str(layer)][group], projections)

        layer_data = {}
        layer_summary = []
        for layer in layers:
            layer_key = str(layer)
            groups = {
                group: finish_series(accum[layer_key][group])
                for group in GROUPS
            }
            layer_data[layer_key] = groups
            summary_row = {"layer": layer}
            for group in GROUPS:
                counts = groups[group]["count"]
                means = groups[group]["mean"]
                values = [
                    mean * count
                    for mean, count in zip(means, counts)
                    if mean is not None and count
                ]
                total_count = sum(int(c) for c in counts)
                summary_row[f"{group}_mean"] = (
                    sum(values) / total_count if total_count else None
                )
            t = summary_row.get("true_mean")
            f = summary_row.get("false_mean")
            summary_row["true_minus_false"] = (
                t - f if t is not None and f is not None else None
            )
            layer_summary.append(summary_row)

        experiments.append(
            {
                "id": "/".join(cache_dir.parts[-3:]),
                "cache_dir": str(cache_dir),
                "source_file": manifest.get("source_file"),
                "model": manifest.get("model"),
                "dataset": manifest.get("dataset"),
                "mode": manifest.get("mode"),
                "prompt_style": manifest.get("prompt_style"),
                "alignment": manifest.get("alignment"),
                "activation_kind": manifest.get("activation_kind"),
                "record_count": manifest.get("record_count"),
                "layers": layers,
                "record_counts": record_counts,
                "token_counts": token_counts,
                "layer_summary": layer_summary,
                "layer_data": layer_data,
            }
        )

    return {
        "diffmeans_path": str(diffmeans_path),
        "diffmeans_metadata": diff_meta,
        "groups": GROUPS,
        "group_labels": GROUP_LABELS,
        "experiments": experiments,
    }


def html_template(data: dict[str, Any]) -> str:
    data_json = json.dumps(data)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Conformity Projection Report</title>
  <style>
    :root {{
      --bg: #f7f7f4;
      --panel: #ffffff;
      --ink: #202124;
      --muted: #6b6f76;
      --line: #d8dadd;
      --true: #0f766e;
      --false: #b42318;
      --none: #5f6368;
    }}
    body {{
      margin: 0;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: var(--bg);
      color: var(--ink);
    }}
    header {{
      padding: 20px 28px 14px;
      border-bottom: 1px solid var(--line);
      background: var(--panel);
    }}
    h1 {{ margin: 0 0 8px; font-size: 24px; }}
    .meta {{ color: var(--muted); font-size: 13px; line-height: 1.5; }}
    main {{ max-width: 1260px; margin: 0 auto; padding: 18px 20px 36px; }}
    .controls {{
      display: flex;
      gap: 12px;
      flex-wrap: wrap;
      align-items: end;
      padding: 14px;
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      margin-bottom: 16px;
    }}
    label {{ display: grid; gap: 5px; font-size: 12px; color: var(--muted); }}
    select, input {{
      font: inherit;
      padding: 7px 9px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #fff;
      min-width: 120px;
    }}
    .grid {{
      display: grid;
      grid-template-columns: minmax(0, 1fr);
      gap: 16px;
    }}
    section {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px;
    }}
    h2 {{ margin: 0 0 10px; font-size: 16px; }}
    .legend {{ display: flex; gap: 16px; color: var(--muted); font-size: 13px; margin: 6px 0 12px; }}
    .legend span {{ display: inline-flex; align-items: center; gap: 6px; }}
    .swatch {{ width: 18px; height: 3px; display: inline-block; }}
    svg {{ width: 100%; height: auto; display: block; overflow: visible; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
    th, td {{ padding: 7px 8px; border-bottom: 1px solid var(--line); text-align: right; }}
    th:first-child, td:first-child {{ text-align: left; }}
    th {{ color: var(--muted); font-weight: 600; }}
    .note {{ color: var(--muted); font-size: 13px; margin-top: 8px; }}
    .tooltip {{
      position: fixed;
      pointer-events: none;
      background: rgba(32,33,36,0.94);
      color: white;
      padding: 7px 9px;
      border-radius: 6px;
      font-size: 12px;
      display: none;
      z-index: 10;
      max-width: 320px;
    }}
  </style>
</head>
<body>
  <header>
    <h1>Conformity Projection Over Response Tokens</h1>
    <div class="meta" id="meta"></div>
  </header>
  <main>
    <div class="controls">
      <label>Experiment<select id="experimentSelect"></select></label>
      <label>Layer<select id="layerSelect"></select></label>
      <label>Smooth Window<input id="smoothInput" type="number" min="1" max="51" step="2" value="7"></label>
    </div>
    <div class="grid">
      <section>
        <h2>Mean Projection By Token Index</h2>
        <div class="legend">
          <span><i class="swatch" style="background:var(--true)"></i> conforms=True</span>
          <span><i class="swatch" style="background:var(--false)"></i> conforms=False</span>
          <span><i class="swatch" style="background:var(--none)"></i> conforms=None</span>
        </div>
        <svg id="lineChart" viewBox="0 0 1040 430"></svg>
        <div class="note">Y values are projections onto the normalized layer-wise diffmean direction. Lines are smoothed only for display.</div>
      </section>
      <section>
        <h2>Layer x Token Heatmap: conforms=True minus conforms=False</h2>
        <svg id="heatmap" viewBox="0 0 1040 520"></svg>
        <div class="note">Red means conforming responses project higher than non-conforming responses; blue means lower.</div>
      </section>
      <section>
        <h2>Layer Summary</h2>
        <table id="summaryTable"></table>
      </section>
    </div>
  </main>
  <div id="tooltip" class="tooltip"></div>
  <script>
    const DATA = {data_json};
    const colors = {{true: '#0f766e', false: '#b42318', none: '#5f6368'}};
    const $ = id => document.getElementById(id);
    const fmt = v => v === null || Number.isNaN(v) ? 'n/a' : Number(v).toFixed(3);
    function meanAt(group, i) {{ return group.mean[i] === undefined ? null : group.mean[i]; }}
    function smooth(arr, win) {{
      if (win <= 1) return arr.slice();
      const half = Math.floor(win / 2);
      return arr.map((_, i) => {{
        let n = 0, s = 0;
        for (let j = Math.max(0, i-half); j <= Math.min(arr.length-1, i+half); j++) {{
          if (arr[j] !== null && arr[j] !== undefined) {{ s += arr[j]; n++; }}
        }}
        return n ? s/n : null;
      }});
    }}
    function scales(seriesList, width, height, pad) {{
      let maxX = 1, minY = Infinity, maxY = -Infinity;
      for (const s of seriesList) {{
        maxX = Math.max(maxX, s.length - 1);
        for (const v of s) if (v !== null && v !== undefined) {{ minY = Math.min(minY, v); maxY = Math.max(maxY, v); }}
      }}
      if (!isFinite(minY)) {{ minY = -1; maxY = 1; }}
      if (minY === maxY) {{ minY -= 1; maxY += 1; }}
      const x = i => pad.l + (i / maxX) * (width - pad.l - pad.r);
      const y = v => pad.t + (1 - (v - minY) / (maxY - minY)) * (height - pad.t - pad.b);
      return {{x, y, maxX, minY, maxY}};
    }}
    function pathFor(values, x, y) {{
      let d = '', open = false;
      values.forEach((v, i) => {{
        if (v === null || v === undefined) {{ open = false; return; }}
        d += (open ? 'L' : 'M') + x(i).toFixed(1) + ',' + y(v).toFixed(1);
        open = true;
      }});
      return d;
    }}
    function drawLineChart() {{
      const exp = DATA.experiments[$('experimentSelect').value];
      const layer = $('layerSelect').value;
      const win = Number($('smoothInput').value) || 1;
      const svg = $('lineChart');
      const W = 1040, H = 430, pad = {{l: 58, r: 18, t: 18, b: 42}};
      const layerData = exp.layer_data[layer];
      const series = DATA.groups.map(g => smooth(layerData[g].mean, win));
      const sc = scales(series, W, H, pad);
      let html = `<rect x="0" y="0" width="${{W}}" height="${{H}}" fill="white"/>`;
      for (let k=0; k<=4; k++) {{
        const yy = pad.t + k*(H-pad.t-pad.b)/4;
        const val = sc.maxY - k*(sc.maxY-sc.minY)/4;
        html += `<line x1="${{pad.l}}" y1="${{yy}}" x2="${{W-pad.r}}" y2="${{yy}}" stroke="#e8eaed"/>`;
        html += `<text x="${{pad.l-8}}" y="${{yy+4}}" text-anchor="end" font-size="12" fill="#6b6f76">${{fmt(val)}}</text>`;
      }}
      const zeroY = sc.y(0);
      if (zeroY >= pad.t && zeroY <= H-pad.b) html += `<line x1="${{pad.l}}" y1="${{zeroY}}" x2="${{W-pad.r}}" y2="${{zeroY}}" stroke="#9aa0a6" stroke-dasharray="4 4"/>`;
      for (let k=0; k<=6; k++) {{
        const x = pad.l + k*(W-pad.l-pad.r)/6;
        const idx = Math.round(k*sc.maxX/6);
        html += `<text x="${{x}}" y="${{H-14}}" text-anchor="middle" font-size="12" fill="#6b6f76">${{idx}}</text>`;
      }}
      DATA.groups.forEach((g, idx) => {{
        html += `<path d="${{pathFor(series[idx], sc.x, sc.y)}}" fill="none" stroke="${{colors[g]}}" stroke-width="2.4"/>`;
      }});
      html += `<text x="${{W/2}}" y="${{H-2}}" text-anchor="middle" font-size="12" fill="#6b6f76">response token index</text>`;
      svg.innerHTML = html;
    }}
    function color(v, maxAbs) {{
      if (v === null || maxAbs === 0) return '#f1f3f4';
      const t = Math.min(1, Math.abs(v) / maxAbs);
      const a = Math.round(245 - 120*t);
      if (v > 0) return `rgb(${{255}},${{a}},${{a}})`;
      return `rgb(${{a}},${{a+20}},${{255}})`;
    }}
    function drawHeatmap() {{
      const exp = DATA.experiments[$('experimentSelect').value];
      const svg = $('heatmap');
      const W = 1040, H = 520, pad = {{l: 48, r: 18, t: 12, b: 34}};
      const layers = exp.layers;
      const maxLen = Math.max(...layers.map(l => exp.layer_data[String(l)].true.mean.length));
      const diffs = [];
      let maxAbs = 0;
      for (const layer of layers) {{
        const ld = exp.layer_data[String(layer)];
        const row = [];
        for (let i=0; i<maxLen; i++) {{
          const t = meanAt(ld.true, i), f = meanAt(ld.false, i);
          const v = (t === null || f === null || t === undefined || f === undefined) ? null : t - f;
          row.push(v);
          if (v !== null) maxAbs = Math.max(maxAbs, Math.abs(v));
        }}
        diffs.push(row);
      }}
      const cw = (W-pad.l-pad.r)/maxLen, ch = (H-pad.t-pad.b)/layers.length;
      let html = `<rect x="0" y="0" width="${{W}}" height="${{H}}" fill="white"/>`;
      for (let r=0; r<layers.length; r++) {{
        const y = pad.t + r*ch;
        html += `<text x="${{pad.l-8}}" y="${{y+ch*0.72}}" text-anchor="end" font-size="10" fill="#6b6f76">${{layers[r]}}</text>`;
        for (let c=0; c<maxLen; c++) {{
          const v = diffs[r][c];
          html += `<rect data-tip="layer ${{layers[r]}}, token ${{c}}, true-false ${{fmt(v)}}" x="${{pad.l+c*cw}}" y="${{y}}" width="${{Math.max(0.5,cw)}}" height="${{Math.max(0.5,ch)}}" fill="${{color(v,maxAbs)}}"/>`;
        }}
      }}
      for (let k=0; k<=6; k++) {{
        const x = pad.l + k*(W-pad.l-pad.r)/6;
        const idx = Math.round(k*(maxLen-1)/6);
        html += `<text x="${{x}}" y="${{H-10}}" text-anchor="middle" font-size="12" fill="#6b6f76">${{idx}}</text>`;
      }}
      svg.innerHTML = html;
    }}
    function drawTable() {{
      const exp = DATA.experiments[$('experimentSelect').value];
      const rows = exp.layer_summary.slice().sort((a,b) => Math.abs(b.true_minus_false ?? 0) - Math.abs(a.true_minus_false ?? 0));
      let html = '<thead><tr><th>layer</th><th>true mean</th><th>false mean</th><th>true - false</th><th>none mean</th></tr></thead><tbody>';
      for (const r of rows) {{
        html += `<tr><td>${{r.layer}}</td><td>${{fmt(r.true_mean)}}</td><td>${{fmt(r.false_mean)}}</td><td>${{fmt(r.true_minus_false)}}</td><td>${{fmt(r.none_mean)}}</td></tr>`;
      }}
      html += '</tbody>';
      $('summaryTable').innerHTML = html;
    }}
    function populate() {{
      $('meta').innerHTML = `Direction: <code>${{DATA.diffmeans_metadata.direction}}</code>; position: <code>${{DATA.diffmeans_metadata.position}}</code>; diffmeans: <code>${{DATA.diffmeans_path}}</code>`;
      const exSel = $('experimentSelect');
      DATA.experiments.forEach((e, i) => {{
        const opt = document.createElement('option');
        opt.value = i;
        opt.textContent = `${{e.id}} · records ${{e.record_count}} · T/F/N ${{e.record_counts.true}}/${{e.record_counts.false}}/${{e.record_counts.none}}`;
        exSel.appendChild(opt);
      }});
      updateLayers();
    }}
    function updateLayers() {{
      const exp = DATA.experiments[$('experimentSelect').value];
      const layerSel = $('layerSelect');
      layerSel.innerHTML = '';
      const preferred = exp.layers.includes(35) ? 35 : exp.layers[exp.layers.length-1];
      for (const layer of exp.layers) {{
        const opt = document.createElement('option');
        opt.value = String(layer);
        opt.textContent = `layer ${{layer}}`;
        if (layer === preferred) opt.selected = true;
        layerSel.appendChild(opt);
      }}
      render();
    }}
    function render() {{ drawLineChart(); drawHeatmap(); drawTable(); }}
    document.addEventListener('mousemove', e => {{
      const tip = e.target?.dataset?.tip;
      const tt = $('tooltip');
      if (!tip) {{ tt.style.display = 'none'; return; }}
      tt.textContent = tip;
      tt.style.left = (e.clientX + 12) + 'px';
      tt.style.top = (e.clientY + 12) + 'px';
      tt.style.display = 'block';
    }});
    $('experimentSelect').addEventListener('change', updateLayers);
    $('layerSelect').addEventListener('change', render);
    $('smoothInput').addEventListener('input', drawLineChart);
    populate();
  </script>
</body>
</html>
"""


def main() -> None:
    args = parse_args()
    data = build_data(args.cache_root, args.diffmeans_path)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(html_template(data))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
