"""Day-7 assembly: collect every stage's JSON into the go/no-go memo + overlay figure.

Tolerant to missing stages (marked PENDING) so it can be run at any point to see where
the week stands. Gates are evaluated against constants.py / the amended plan doc §4.

Usage: python -m posthoc_ci.report
"""

import json

import numpy as np

from posthoc_ci import constants, glib, paths
from posthoc_ci.figstyle import SERIES, apply_style, save

GATE_C = {
    "deterministic_ce_nats": 0.05,
    "stochastic_mean_ratio": 1.3,
    "stochastic_p99_ratio": 2.0,
    "adversarial_ratio": 1.5,
    "induced_l0_max": 2.0,
}


def _load(path):
    return json.loads(path.read_text()) if path.exists() else None


def _overlay_figure(summary, controls, mdl) -> None:
    apply_style()
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    recs = [r for r in summary["records"] if r["arm"] == "sym"]
    ks = sorted({r["k"] for r in recs})
    for ax, field, ylabel in (
        (axes[0], "recall_weighted", "weighted recall@0.1 (held-out)"),
        (axes[1], "induced_l0_ratio", "induced-L0 ratio (held-out)"),
    ):
        means = [np.mean([r[field] for r in recs if r["k"] == k]) for k in ks]
        ax.plot(ks, means, "o-", color=SERIES[0], label="NMF (sym)")
        asym = [r for r in summary["records"] if r["arm"].startswith("asym")]
        if asym:
            ks_a = sorted({r["k"] for r in asym})
            means_a = [np.mean([r[field] for r in asym if r["k"] == k]) for k in ks_a]
            ax.plot(ks_a, means_a, "s-", color=SERIES[1], label=f"NMF ({asym[0]['arm']})")
        if controls:
            for name, marker, si in (("shuffled", "x", 3), ("random", "^", 4)):
                if name in controls:
                    ax.scatter([controls["k"]], [controls[name][field]], marker=marker,
                               color=SERIES[si], zorder=5, label=f"{name} B")
        if mdl and field in ("recall_weighted", "induced_l0_ratio"):
            key = {"recall_weighted": "recall_at_tau_eval",
                   "induced_l0_ratio": "induced_l0_ratio"}[field]
            ax.scatter([mdl["k_groups"]], [mdl["or_broadcast"][key]], marker="D",
                       color=SERIES[2], zorder=5, label="MDL OR-broadcast")
            ax.scatter([mdl["k_groups"]], [mdl["partition_constrained_fit"][
                "recall_weighted" if field == "recall_weighted" else "induced_l0_ratio"]],
                marker="v", color=SERIES[2], zorder=5, label="MDL partition+fit")
        ax.set(xscale="log", xlabel="K", ylabel=ylabel)
        ax.legend(fontsize=8)
    axes[1].axhline(GATE_C["induced_l0_max"], color=SERIES[3], lw=1, ls="--")
    fig.suptitle("E1.4 baseline overlay", fontsize=11)
    save(fig, glib.fig_path("e14_overlay.png"))


def main() -> None:
    lines = ["# Rung 1 & 2 — results memo (auto-collected)", ""]

    sanity = _load(paths.SANITY_REPORT_JSON)
    if sanity:
        lines += [
            "## E1.0 harvest sanity — PASS" if abs(sanity["mean_l0_at_tau_store"] - 205) < 52
            else "## E1.0 harvest sanity — CHECK",
            f"- positions {sanity['n_positions']}, alive {sanity['n_alive']}, "
            f"L0@0.01 = {sanity['mean_l0_at_tau_store']} (paper ~205)",
            f"- per-layer alive {sanity['per_layer_alive']} (paper 848/1943/3472)", "",
        ]

    gate_a = _load(paths.HARVEST_DIR / "e12_gate_a.json")
    if gate_a:
        lines += [f"## E1.2 — {gate_a['gate_a_verdict']}",
                  f"- graded EV@512 {gate_a['graded']['ev_at_512']}, "
                  f"rank for 90% ≈ {gate_a['graded']['rank_for_90pct']}"
                  f" (extrapolated: {gate_a['graded']['rank_90_extrapolated']})", ""]

    summary = _load(paths.FITS_DIR / "summary.json")
    controls = None
    mdl = None
    if summary:
        lines += [f"## E1.3 — knee: {summary['knee']}, Gate B pass: {summary['gate_b_pass']}", ""]
        lines += ["| K | arm | EV | w-recall@0.1 | induced-L0 | code-L0@0.01 |",
                  "|---|-----|----|--------------|------------|--------------|"]
        by_cell: dict[tuple, list] = {}
        for r in summary["records"]:
            by_cell.setdefault((r["k"], r["arm"]), []).append(r)
        for (k, arm), rs in sorted(by_cell.items()):
            ev = np.mean([r["ev"] for r in rs])
            wr = np.mean([r["recall_weighted"] for r in rs])
            il = np.mean([r["induced_l0_ratio"] for r in rs])
            cl = np.mean([r["code_l0"]["0.01"] for r in rs])
            lines.append(f"| {k} | {arm} | {ev:.3f} | {wr:.3f} | {il:.2f} | {cl:.1f} |")
        lines.append("")

        ctrl_files = sorted(paths.BASELINES_DIR.glob("controls_K*.json"))
        if ctrl_files:
            controls = _load(ctrl_files[-1])
            controls["k"] = int(ctrl_files[-1].stem.split("_")[1][1:])
            lines += [f"## E1.4c controls (K={controls['k']})",
                      f"- trained EV {controls['trained']['ev']:.3f} vs shuffled "
                      f"{controls['shuffled']['ev']:.3f} vs random {controls['random']['ev']:.3f}", ""]
        mdl_files = sorted(paths.BASELINES_DIR.glob("mdl_K*.json"))
        if mdl_files:
            mdl = _load(mdl_files[-1])
            lines += [f"## E1.4b MDL baseline (k_groups={mdl['k_groups']})",
                      f"- OR-broadcast: recall {mdl['or_broadcast']['recall_at_tau_eval']:.3f}, "
                      f"induced-L0 {mdl['or_broadcast']['induced_l0_ratio']:.2f}",
                      f"- partition+fit: EV {mdl['partition_constrained_fit']['ev']:.3f}, "
                      f"w-recall {mdl['partition_constrained_fit']['recall_weighted']:.3f}", ""]
        _overlay_figure(summary, controls, mdl)

    latent = _load(paths.LATENTS_DIR / "latent_report.json")
    if latent:
        h = latent["headline"]
        lines += ["## E1.5 latents",
                  f"- {h['frac_nonhub_latents_spanning_2_matrices']:.0%} of non-hub latents span "
                  f"≥2 matrices; {h['frac_nonhub_latents_spanning_attn_and_mlp']:.0%} span attn+mlp; "
                  f"stability median cos {h['stability_median_cos']:.2f}", ""]

    lines += ["## Rung 2", ""]
    for f in sorted(paths.SUBST_DIR.glob("*.json")):
        rec = _load(f)
        lines.append(f"### {f.stem}")
        if f.stem.startswith("encode"):
            lines += [f"- recall {rec['weighted_recall_at_tau_eval']:.3f} (weighted), induced-L0 "
                      f"{rec['induced_l0_ratio_store']:.2f}, code-L0 {rec['mean_code_l0_0.01']:.1f}, "
                      f"clip {rec['clip_rate']:.3%}, identity test "
                      f"{'PASS' if rec['identity_self_test_pass'] else 'FAIL'}"]
        elif f.stem.startswith("deterministic"):
            for cond, rows in rec["table"].items():
                ci_row = rows.get("ci", {})
                lines.append(f"- {cond}: CI-masked KL {ci_row.get('kl', float('nan')):.4f} "
                             f"(CI95 {ci_row.get('kl_ci95')}), floors L0 "
                             f"{rows['floors_l0_at_tau_store']:.1f}")
        elif f.stem.startswith("stochastic"):
            lines.append(f"- mean ratio {rec['ratio_mean']:.2f} (gate ≤ "
                         f"{GATE_C['stochastic_mean_ratio']}), p99 ratio {rec['ratio_p99']:.2f} "
                         f"(gate ≤ {GATE_C['stochastic_p99_ratio']})")
        elif f.stem.startswith("adversarial"):
            for cond, r in rec["results"].items():
                ratio = r.get("ratio_vs_g_20steps")
                lines.append(f"- {cond}: adv KL {r.get('adv_kl_20steps', float('nan')):.4f}"
                             + (f", ratio vs g {ratio:.2f} (gate ≤ "
                                f"{GATE_C['adversarial_ratio']})" if ratio else "")
                             + f", floors L0 {r['floors_l0_at_tau_store']:.1f}")
        lines.append("")

    lines += ["## Pre-committed gate thresholds (§4, amended)",
              f"```\n{json.dumps(GATE_C, indent=2)}\n```", ""]

    out = paths.POSTHOC_ROOT / "memo.md"
    out.write_text("\n".join(lines))
    print("\n".join(lines))
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
