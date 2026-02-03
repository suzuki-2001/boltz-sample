import marimo

__generated_with = "0.17.7"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    import pandas as pd
    import matplotlib.pyplot as plt
    from pathlib import Path
    return Path, pd, plt


@app.cell
def _(Path, pd, plt):
    # Load TM-score data (auto-detect path)
    tsv_path = Path("output/tm_scores.tsv")
    if not tsv_path.exists():
        _candidates = list(Path("output").glob("boltz_results_*/tm_scores.tsv"))
        if _candidates:
            tsv_path = _candidates[0]

    df = pd.read_csv(tsv_path, sep="\t")

    # Auto-detect TM column names from TSV header
    tm_cols = [c for c in df.columns if c.startswith("tm_")]
    col_x, col_y = tm_cols[0], tm_cols[1]
    ref_x = col_x.replace("tm_", "")
    ref_y = col_y.replace("tm_", "")

    # Create scatter plot
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.set_facecolor("#f0f0f0")
    cmap = plt.cm.jet_r

    # Get unique beta values
    betas = sorted(df["beta"].unique())
    beta_min = min(betas)
    beta_max = max(betas)
    norm = plt.Normalize(vmin=beta_min, vmax=beta_max)

    # Plot each beta value with its own color
    for beta in betas:
        subset = df[df["beta"] == beta]
        color = cmap(norm(beta))
        label = f"beta={beta:+.2f}"
        ax.scatter(
            subset[col_x],
            subset[col_y],
            c=[color],
            s=10,
            edgecolors="black",
            linewidths=0.5,
            alpha=0.8,
            label=label,
        )

    # Set equal axis limits for square plot
    lim_min = min(df[col_x].min(), df[col_y].min()) - 0.02
    lim_max = max(df[col_x].max(), df[col_y].max()) + 0.02
    ax.set_xlim(lim_min, lim_max)
    ax.set_ylim(lim_min, lim_max)

    # Reduce tick density
    ax.xaxis.set_major_locator(plt.MaxNLocator(5))
    ax.yaxis.set_major_locator(plt.MaxNLocator(5))

    # Add diagonal line
    ax.plot([lim_min, lim_max], [lim_min, lim_max], "k--", alpha=0.3, linewidth=1)

    # Labels (wrapped to 2 lines)
    ax.set_xlabel(f"TM-score to {ref_x}\n(inactive)", fontsize=10)
    ax.set_ylabel(f"TM-score to {ref_y}\n(active)", fontsize=10)
    ax.set_title("mu-OR Structure Prediction", fontsize=11)
    ax.grid(True, linestyle="--", linewidth=0.3, color="gray", alpha=0.8)

    # Legend with markers (no frame, centered)
    ax.legend(
        loc="center",
        bbox_to_anchor=(0.5, -0.5),
        ncol=5,
        fontsize=7,
        frameon=False,
        title="beta (uniform scaling)",
        title_fontsize=9,
        handletextpad=0.3,
        columnspacing=0.8,
    )

    # Make plot square
    ax.set_aspect("equal")
    plt.tight_layout()
    plt.savefig("output/tmscore_scatter.png", dpi=150, bbox_inches="tight")
    fig
    return col_x, col_y, df, ref_x, ref_y


@app.cell
def _(col_x, col_y, df, mo, ref_x, ref_y):
    # Find best predictions for each reference structure
    # Use model_idx if available, fall back to sample for legacy TSVs
    idx_col = "model_idx" if "model_idx" in df.columns else "sample"

    best_ref1 = df.loc[df[col_x].idxmax()]
    best_ref2 = df.loc[df[col_y].idxmax()]

    mo.md(f"""
    **Best prediction for {ref_x} (inactive state):**
    - beta={best_ref1['beta']:+.2f}, {idx_col}={int(best_ref1[idx_col])}, TM-score={best_ref1[col_x]:.3f}

    **Best prediction for {ref_y} (active state):**
    - beta={best_ref2['beta']:+.2f}, {idx_col}={int(best_ref2[idx_col])}, TM-score={best_ref2[col_y]:.3f}
    """)
    return best_ref1, best_ref2, idx_col


@app.cell
def _(Path, best_ref1, best_ref2, idx_col, mo, ref_x, ref_y):
    # Auto-detect predictions directory
    predictions_dir = Path("output/predictions")
    if not predictions_dir.is_dir():
        _candidates = list(Path("output").glob("boltz_results_*/predictions"))
        if _candidates:
            predictions_dir = _candidates[0]

    script_dir = Path(".")

    def beta_to_dirname(beta: float) -> str:
        """Convert beta value to directory name."""
        return f"beta_{beta:+.2f}".replace(".", "p").replace("+", "pos").replace("-", "neg")

    def get_prediction_path(beta: float, model_idx: int) -> Path:
        """Get path to prediction CIF file."""
        dirname = beta_to_dirname(beta)
        beta_dir = predictions_dir / dirname
        if beta_dir.is_dir():
            targets = [d for d in beta_dir.iterdir() if d.is_dir()]
            if targets:
                target = targets[0].name
                return beta_dir / target / f"{target}_model_{model_idx}.cif"
        return beta_dir / "muor" / f"muor_model_{model_idx}.cif"

    best_ref1_path = get_prediction_path(best_ref1["beta"], int(best_ref1[idx_col]))
    best_ref2_path = get_prediction_path(best_ref2["beta"], int(best_ref2[idx_col]))

    mo.md(f"""
    **Prediction paths:**
    - Best {ref_x} prediction: `{best_ref1_path}`
    - Best {ref_y} prediction: `{best_ref2_path}`
    """)
    return best_ref1_path, best_ref2_path, script_dir


@app.cell
def _(Path, mo):
    def show_structure(
        structure_path: Path,
        title: str = "",
        width: int = 500,
        height: int = 400,
        color_by_confidence: bool = False,
    ):
        """Display protein structure using 3Dmol.js via mo.iframe().

        Parameters
        ----------
        structure_path : Path
            Path to structure file (PDB or CIF).
        title : str
            Title to display above the viewer.
        width : int
            Viewer width in pixels.
        height : int
            Viewer height in pixels.
        color_by_confidence : bool
            If True, color by B-factor/pLDDT (blue=high, red=low).
            Recommended for predicted structures.
        """
        with open(structure_path, "r") as f:
            content = f.read()

        # Determine format from extension
        ext = structure_path.suffix.lower()
        fmt = "cif" if ext == ".cif" else "pdb"

        # Escape content for JavaScript template literal
        content_escaped = (
            content.replace("\\", "\\\\").replace("`", "\\`").replace("$", "\\$")
        )

        # Color scheme: confidence (pLDDT) or spectrum
        if color_by_confidence:
            # AlphaFold-style: blue (high confidence) to red (low confidence)
            style_js = """{cartoon: {colorscheme: {prop: 'b', gradient: 'rwb', min: 50, max: 90}}}"""
        else:
            style_js = """{cartoon: {color: "spectrum"}}"""

        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <script src="https://3Dmol.org/build/3Dmol-min.js"></script>
            <style>
                body {{ margin: 0; padding: 0; }}
                #viewer {{ width: {width}px; height: {height}px; }}
            </style>
        </head>
        <body>
            <div id="viewer"></div>
            <script>
                var viewer = $3Dmol.createViewer("viewer", {{backgroundColor: "white"}});
                viewer.addModel(`{content_escaped}`, "{fmt}");
                viewer.setStyle({{}}, {style_js});
                viewer.zoomTo();
                viewer.render();
            </script>
        </body>
        </html>
        """

        title_md = mo.md(f"**{title}**") if title else mo.md("")
        viewer = mo.iframe(html, width=f"{width}px", height=f"{height}px")

        return mo.vstack([title_md, viewer])
    return (show_structure,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Reference Structures (Experimental)
    """)
    return


@app.cell
def _(script_dir, show_structure):
    # Reference: 4DKL (inactive state)
    show_structure(
        script_dir / "data" / "ref" / "4DKL_7tm.pdb", "Reference: 4DKL (inactive state)"
    )


@app.cell
def _(script_dir, show_structure):
    # Reference: 5C1M (active state)
    show_structure(
        script_dir / "data" / "ref" / "5C1M_7tm.pdb", "Reference: 5C1M (active state)"
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Best Predicted Structures
    """)
    return


@app.cell
def _(best_ref1, best_ref1_path, col_x, ref_x, show_structure):
    # Best prediction matching inactive state (colored by pLDDT confidence)
    show_structure(
        best_ref1_path,
        f"Best prediction for inactive state ({ref_x}): beta={best_ref1['beta']:+.2f}, TM={best_ref1[col_x]:.3f}",
        color_by_confidence=True,
    )


@app.cell
def _(best_ref2, best_ref2_path, col_y, ref_y, show_structure):
    # Best prediction matching active state (colored by pLDDT confidence)
    show_structure(
        best_ref2_path,
        f"Best prediction for active state ({ref_y}): beta={best_ref2['beta']:+.2f}, TM={best_ref2[col_y]:.3f}",
        color_by_confidence=True,
    )


if __name__ == "__main__":
    app.run()
