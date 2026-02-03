# Examples

## RfaH (Fold-switching protein)

RfaH C-terminal domain switches between α-helix and β-barrel conformations.

```bash
cd example/rfah && bash run.sh
```

### Visualization

```bash
marimo edit example/rfah/visualize_tmscore.py
```

<table>
<tr>
<td><img src="../md/rfah_alpha_helix.png" width="350"><br><sub>α-helix CTD (beta=+0.60)</sub></td>
<td><img src="../md/rfah_beta_barrel.png" width="350"><br><sub>β-barrel CTD (beta=-0.45)</sub></td>
</tr>
</table>

---

## μ-Opioid Receptor (GPCR)

μ-OR adopts inactive and active conformations.

```bash
cd example/muor && bash run.sh
```

### Visualization

```bash
marimo edit example/muor/visualize_tmscore.py
```

<table>
<tr>
<td><img src="../md/muor_inactive.png" width="350"><br><sub>Inactive state (beta=-0.45)</sub></td>
<td><img src="../md/muor_active.png" width="350"><br><sub>Active state (beta=+0.45)</sub></td>
</tr>
</table>

---

## Color Scheme

Predicted structures are colored by pLDDT confidence:
- **Blue**: High confidence (pLDDT ≥ 90)
- **White**: Medium confidence
- **Red**: Low confidence (pLDDT ≤ 50)
