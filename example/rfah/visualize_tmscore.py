"""Marimo notebook: visualize TM-score scatter for RfaH β-grid sampling.

Reads predicted structures from `output_boltz2/beta_*/` (Boltz-2) or
`output_af3/beta_*/` (AlphaFold 3), computes TM-score to each reference
(2OUG_C α-helix, 6C6S_D β-barrel) using tmtools, and plots them.

Run after `bash run_boltz2.sh` (or `run_af3.sh`):
    pip install marimo tmtools matplotlib
    marimo edit example/rfah/visualize_tmscore.py
"""
import marimo

__generated_with = "0.17.7"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    from pathlib import Path
    import matplotlib.pyplot as plt
    import pandas as pd
    return Path, pd, plt


@app.cell
def _(Path):
    SCRIPT = Path(__file__).parent
    REF1 = SCRIPT / "data" / "ref" / "2OUG_C.pdb"   # α-helix CTD
    REF2 = SCRIPT / "data" / "ref" / "6C6S_D.pdb"   # β-barrel CTD
    OUT_BOLTZ = SCRIPT / "output_boltz2"
    OUT_AF3   = SCRIPT / "output_af3"
    return REF1, REF2, OUT_BOLTZ, OUT_AF3


@app.cell
def _():
    import re

    def parse_beta(dirname: str) -> float:
        """beta_neg0p45 -> -0.45, beta_pos0p30 -> 0.30, beta_zero -> 0.0"""
        m = re.match(r"beta_(neg|pos|zero)(?:(\d+)p(\d+))?", dirname)
        if m is None:
            return float("nan")
        sign, integer, frac = m.groups()
        if sign == "zero":
            return 0.0
        magnitude = float(f"{integer}.{frac}")
        return -magnitude if sign == "neg" else magnitude
    return (parse_beta,)


@app.cell
def _(OUT_BOLTZ, OUT_AF3, REF1, REF2, parse_beta, pd):
    from tmtools.io import get_structure, get_residue_data
    from tmtools import tm_align

    def tm_to_ref(sample_path, ref_path):
        s_chain = next(get_structure(str(sample_path)).get_chains())
        r_chain = next(get_structure(str(ref_path)).get_chains())
        coords1, seq1 = get_residue_data(s_chain)
        coords2, seq2 = get_residue_data(r_chain)
        result = tm_align(coords1, coords2, seq1, seq2)
        return result.tm_norm_chain2

    rows = []
    for out_root, model in [(OUT_BOLTZ, "boltz2"), (OUT_AF3, "af3")]:
        if not out_root.exists():
            continue
        for beta_dir in sorted(out_root.iterdir()):
            if not beta_dir.is_dir():
                continue
            beta_value = parse_beta(beta_dir.name)
            samples = list(beta_dir.rglob("*_model_*.pdb")) + list(
                beta_dir.rglob("*_model.cif")
            )
            for s in samples:
                rows.append({
                    "model": model,
                    "beta": beta_value,
                    "sample": s.name,
                    "tm_2OUG_C": tm_to_ref(s, REF1),
                    "tm_6C6S_D": tm_to_ref(s, REF2),
                })
    df = pd.DataFrame(rows)
    return (df,)


@app.cell
def _(df, plt):
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.set_facecolor("#f0f0f0")
    cmap = plt.cm.jet_r
    betas = sorted(df["beta"].unique())
    norm = plt.Normalize(vmin=min(betas), vmax=max(betas))
    for b in betas:
        sub = df[df["beta"] == b]
        ax.scatter(
            sub["tm_2OUG_C"], sub["tm_6C6S_D"],
            c=[cmap(norm(b))],
            s=14, edgecolors="black", linewidths=0.4,
            alpha=0.85,
            label=f"beta={b:+.2f}",
        )
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, linewidth=1)
    ax.set_xlabel("TM-score to 2OUG_C\n(alpha-helix CTD)", fontsize=10)
    ax.set_ylabel("TM-score to 6C6S_D\n(beta-barrel CTD)", fontsize=10)
    ax.set_title("RfaH beta-grid sampling", fontsize=11)
    ax.grid(True, linestyle="--", linewidth=0.3, color="gray", alpha=0.7)
    ax.legend(
        loc="center", bbox_to_anchor=(0.5, -0.4),
        ncol=5, fontsize=7, frameon=False,
        title="beta (uniform scaling)",
    )
    ax.set_aspect("equal")
    plt.tight_layout()
    fig
    return


if __name__ == "__main__":
    app.run()
