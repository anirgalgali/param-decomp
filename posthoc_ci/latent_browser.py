"""E1.5: static, self-contained HTML browser over latent_report.json.

One file, no server, no external assets — scp it back and open locally. Left rail lists
featured latents; each latent page shows its top contexts (firing token highlighted)
and its top member atoms with their own contexts, side by side, so member-consistency
labeling is a scroll, not a join.

Usage: python -m posthoc_ci.latent_browser
"""

import html
import json

from posthoc_ci import paths

CSS = """
body { font: 14px/1.45 -apple-system, system-ui, sans-serif; margin: 0;
       background: #fcfcfb; color: #0b0b0b; }
.layout { display: flex; }
nav { width: 220px; padding: 12px; border-right: 1px solid #e6e5e0; position: sticky;
      top: 0; height: 100vh; overflow-y: auto; box-sizing: border-box; }
nav a { display: block; padding: 3px 6px; color: #2a78d6; text-decoration: none;
        border-radius: 4px; font-variant-numeric: tabular-nums; }
nav a:hover { background: #eef4fc; }
main { flex: 1; padding: 18px 28px; max-width: 1100px; }
.latent { border-bottom: 2px solid #e6e5e0; padding: 18px 0; }
h2 { margin: 0 0 2px; } .meta { color: #52514e; font-size: 12px; margin-bottom: 10px; }
.ctx { margin: 2px 0; white-space: nowrap; overflow-x: auto; }
.ctx .tok { background: #cde2fb; border-radius: 3px; padding: 0 2px; font-weight: 600; }
.ctx .val { display: inline-block; width: 52px; color: #52514e;
            font-variant-numeric: tabular-nums; }
.members { display: grid; grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
           gap: 12px; margin-top: 10px; }
.member { border: 1px solid #e6e5e0; border-radius: 6px; padding: 8px; }
.member h4 { margin: 0 0 4px; font-size: 13px; }
.flag { color: #eb6834; font-size: 11px; font-weight: 600; margin-left: 6px; }
.xmat { font-size: 12px; color: #52514e; }
.ctxlabel { font-size: 11px; color: #52514e; margin: 6px 0 2px; font-style: italic; }
details > summary { cursor: pointer; color: #2a78d6; font-size: 13px; margin: 6px 0; }
"""


def _ctx_html(c: dict, val_key: str) -> str:
    return (
        f'<div class="ctx"><span class="val">{c[val_key]:.3f}</span>'
        f"{html.escape(c['before'])}"
        f'<span class="tok">{html.escape(c["token"])}</span>'
        f"{html.escape(c['after'])}</div>"
    )


def main() -> None:
    report = json.loads((paths.LATENTS_DIR / "latent_report.json").read_text())
    parts = [
        "<!doctype html><meta charset='utf-8'>",
        f"<title>Latent browser K={report['k']}</title>",
        f"<style>{CSS}</style><div class='layout'><nav><b>latents</b>",
    ]
    for rec in report["latents"]:
        x = rec["cross_matrix"]["n_matrices_ge_20pct"]
        parts.append(f"<a href='#L{rec['latent']}'>#{rec['latent']} · z̄ {rec['mean_top_z']} · ⊞{x}</a>")
    parts.append("</nav><main>")
    h = report["headline"]
    parts.append(
        f"<h1>Latent browser — K={report['k']}, seed {report['seed']}, arm {report['arm']}</h1>"
        f"<p class='meta'>non-hub latents spanning ≥2 matrices: "
        f"{h['frac_nonhub_latents_spanning_2_matrices']:.0%} · spanning attn+mlp: "
        f"{h['frac_nonhub_latents_spanning_attn_and_mlp']:.0%} · seed-stability median cos: "
        f"{h['stability_median_cos']:.2f} · hub latents flagged: {h['n_hub_latents_flagged']}</p>"
    )
    for rec in report["latents"]:
        xm = ", ".join(f"{m} {v:.0%}" for m, v in rec["cross_matrix"]["top_matrices"])
        parts.append(
            f"<div class='latent' id='L{rec['latent']}'><h2>latent #{rec['latent']}</h2>"
            f"<div class='meta'>mean top-z {rec['mean_top_z']} · density {rec['density']} · "
            f"mass: {xm}</div>"
        )
        parts.extend(_ctx_html(c, "z") for c in rec["contexts"][:20])
        parts.append("<details><summary>member atoms</summary><div class='members'>")
        for m in rec["members"]:
            flags = "".join(
                f"<span class='flag'>{f}</span>"
                for f, on in (("HUB", m["is_hub"]), ("POSITIONAL", m["is_positional"]))
                if on
            )
            share = m.get("row_share")
            share_txt = f" · {share:.0%} of atom's mass in this latent" if share is not None else ""
            parts.append(
                f"<div class='member'><h4>{m['module']}:{m['c']} "
                f"<span class='xmat'>B={m['b_weight']}{share_txt}</span>{flags}</h4>"
            )
            cond = m.get("conditioned_contexts", [])
            parts.append("<div class='ctxlabel'>while this latent is engaged (by z):</div>")
            if cond:
                parts.extend(_ctx_html(c, "z") for c in cond[:6])
            else:
                parts.append(
                    "<div class='ctxlabel'>— never fires above 0.1 while latent engaged</div>"
                )
            parts.append("<div class='ctxlabel'>global top firings (by g, all uses mixed):</div>")
            parts.extend(_ctx_html(c, "g") for c in m["contexts"][:4])
            parts.append("</div>")
        parts.append("</div></details></div>")
    parts.append("</main></div>")

    out = paths.LATENTS_DIR / "latent_browser.html"
    out.write_text("".join(parts))
    print(f"-> {out} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
