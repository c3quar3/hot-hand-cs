"""Exploratory per-player hot-hand degree distribution figure.

Reads only frozen inputs, rebuilds the participant-event state with the frozen
construction, fits the fixed model per player (one-way match clustering), and
plots the per-player beta distribution for E-A and E-B with the frozen pooled
beta and zero as references.

Exploratory figure for author review; the manuscript supplement would carry a
versioned result package and a figure legend before inclusion.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SKILL_SCRIPTS = Path(r"C:\Users\Admin\.dsh\skills\nature-figure\scripts")
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(SKILL_SCRIPTS))

import matplotlib as mpl
import matplotlib.pyplot as plt
from audit_panel_alignment import require_matplotlib_panel_alignment

from analyze_exponential_history_state_sequence_v1 import build_state, read_json, digest, require

ASSET = ROOT / "results" / "conditional_residual_asset_e1_v1" / "E1-v2" / "residual_asset.parquet"
ASSIGNMENTS = ROOT / "data" / "unified_evidence_two_partitions_v3" / "match_assignments.parquet"
ASSIGNMENT_MANIFEST = ROOT / "data" / "unified_evidence_two_partitions_v3" / "manifest.json"
CONFIRMATION_SUMMARY = ROOT / "results" / "unified_evidence_two_partition_confirmation_v1" / "checkpoint_05_summary.json"
OUT_DIR = ROOT / "docs" / "稿件" / "figures" / "per_player_hot_hand_degree_v1"
PARTITIONS = ("E-A", "E-B")
Z95 = 1.959963984540054
MIN_ROWS = 100
MIN_MATCHES = 2

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "font.size": 8,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.linewidth": 0.8,
    "legend.frameon": False,
})


def fit_player(frame: pd.DataFrame, reference: dict[str, float]) -> dict[str, Any]:
    n = len(frame)
    z = np.log2(1.0 + frame["d_tick"].to_numpy(dtype=float)) - float(reference["mu_z"])
    s = (frame["S_h60"].to_numpy(dtype=float) - float(reference["mu_S"])) / float(reference["sigma_S_ddof1"])
    design = np.column_stack([np.ones(n), z, s, s * z])
    k = design.shape[1]
    match_ids = frame["match_id"].astype(str).to_numpy()
    groups = int(len(np.unique(match_ids)))
    if n <= k or np.linalg.matrix_rank(design) < k or groups < MIN_MATCHES:
        return {"n": int(n), "match_clusters": groups, "rank_fail": True}
    inverse = np.linalg.inv(design.T @ design)
    response = frame["e_t"].to_numpy(dtype=float)
    coefficient = inverse @ design.T @ response
    residual = response - design @ coefficient
    _, codes = np.unique(match_ids, return_inverse=True)
    summed = np.zeros((int(codes.max()) + 1, k))
    np.add.at(summed, codes, design * residual[:, None])
    correction = (groups / (groups - 1)) * ((n - 1) / (n - k))
    covariance = inverse @ (correction * (summed.T @ summed)) @ inverse
    covariance = (covariance + covariance.T) / 2
    se = math.sqrt(max(float(covariance[2, 2]), 0.0))
    beta = float(coefficient[2])
    return {
        "n": int(n), "match_clusters": groups, "rank_fail": False,
        "beta": beta, "se": se,
        "ci95_low": beta - Z95 * se, "ci95_high": beta + Z95 * se,
    }


def main() -> None:
    manifest = read_json(ASSIGNMENT_MANIFEST)
    require(digest(ASSIGNMENTS) == manifest["files"]["match_assignments.parquet"], "assignment_hash_mismatch")
    assignments = pd.read_parquet(ASSIGNMENTS)
    assignments["match_id"] = assignments["match_id"].astype(str)
    require(assignments["match_id"].is_unique, "assignment_nonunique")
    asset = pd.read_parquet(ASSET)
    require(asset["physical_event_id"].is_unique, "asset_event_nonunique")
    asset["match_id"] = asset["match_id"].astype(str)
    merged = asset.merge(assignments[["match_id", "partition"]], on="match_id", how="left")
    require(merged["partition"].notna().all(), "assignment_coverage_incomplete")
    confirmation = read_json(CONFIRMATION_SUMMARY)
    reference = {key: float(confirmation["reference"][key]) for key in ("mu_z", "mu_S", "sigma_S_ddof1")}

    panels = {}
    for part in PARTITIONS:
        part_matches = sorted(assignments.loc[assignments["partition"].eq(part), "match_id"].tolist())
        sub = merged.loc[merged["match_id"].isin(part_matches)].copy()
        state, audit = build_state(sub, part)
        eligible = state.loc[state["analysis_eligible"]].copy()
        eligible["d_tick"] = eligible["event_tick"] - eligible["prior_tick"]
        eligible["match_id"] = eligible["match_id"].astype(str)
        eligible["player_id"] = eligible["player_id"].astype(str)
        betas, ses, cexcl, n_ok = [], [], 0, 0
        for _, rows in eligible.groupby("player_id"):
            if len(rows) >= MIN_ROWS:
                fit = fit_player(rows, reference)
                if not fit.get("rank_fail"):
                    betas.append(fit["beta"]); ses.append(fit["se"]); n_ok += 1
                    if fit["ci95_low"] > 0 or fit["ci95_high"] < 0:
                        cexcl += 1
        panels[part] = {
            "betas": np.asarray(betas), "n": n_ok, "cexcl": cexcl,
            "pooled": float(confirmation["models"][part]["beta"]["estimate"]),
        }

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9), sharey=True)
    all_min = min(float(p["betas"].min()) for p in panels.values())
    all_max = max(float(p["betas"].max()) for p in panels.values())
    pad = 0.04
    x_lo, x_hi = all_min - pad, all_max + pad
    bins = np.arange(x_lo, x_hi + 0.02, 0.02)

    for ax, part in zip(axes, PARTITIONS):
        p = panels[part]
        ax.hist(p["betas"], bins=bins, color="#8FA9B9", edgecolor="white", linewidth=0.4)
        ax.axvline(0.0, color="#555555", linewidth=0.8, linestyle="--")
        ax.axvline(p["pooled"], color="#B03A2E", linewidth=1.1)
        ax.set_xlim(x_lo, x_hi)
        ax.set_xlabel("Per-player state association $\\beta_j$\n(residual units per D1-A reference SD)")
        ax.set_ylabel("Players")
        ax.set_title(part, loc="left", fontsize=9)
        ax.text(0.985, 0.96, f"$n$ = {p['n']} players\nCI excludes 0: {p['cexcl']}/{p['n']}",
                transform=ax.transAxes, ha="right", va="top", fontsize=7)
        ax.text(p["pooled"], 1.02, f"pooled $\\beta$ = {p['pooled']:.3f}",
                transform=ax.get_xaxis_transform(), ha="center", va="bottom",
                color="#B03A2E", fontsize=7)
    axes[0].set_ylabel("Players")
    fig.suptitle("", y=1.0)

    fig.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = OUT_DIR / "figure_per_player_hot_hand_degree_v1"
    require_matplotlib_panel_alignment(
        fig, json_out=str(stem) + ".alignment.json", overlay_svg=str(stem) + ".alignment.svg",
        tolerance_pt=1.5, gutter_tolerance_pt=1.5, strict=True,
    )
    fig.savefig(str(stem) + ".svg", bbox_inches="tight")
    fig.savefig(str(stem) + ".pdf", bbox_inches="tight")
    fig.savefig(str(stem) + ".png", dpi=300, bbox_inches="tight")
    print("saved", stem)


if __name__ == "__main__":
    main()
