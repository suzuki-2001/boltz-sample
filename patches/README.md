# Patches

The pair representation scaling change for each backend, as a diff against a
pinned upstream revision. A scalar β multiplies the pair representation,
`z := (1 + β)·z`, just before the Pairformer stack.

| Patch | Upstream | Revision | License |
|---|---|---|---|
| `boltz-2.2.1.patch` | https://github.com/jwohlwend/boltz | tag `v2.2.1` (`cb04aec`) | MIT |
| `alphafold3-97639ff.patch` | https://github.com/google-deepmind/alphafold3 | `97639ff` (2026-05-06) | CC BY-NC-SA 4.0 |

## Boltz-2

The `prs` CLI applies the same scaling at runtime ([`src/prs/boltz_hook.py`](../src/prs/boltz_hook.py)),
so a stock `pip install boltz==2.2.1` is enough to run a β sweep. Use the patch
to put the change in a Boltz-2 checkout instead:

```bash
git clone --branch v2.2.1 https://github.com/jwohlwend/boltz
cd boltz
git apply -p1 ../patches/boltz-2.2.1.patch
```

It adds a `--beta` option to `boltz predict` and applies the scaling in
`PairformerModule.forward`. The confidence head builds its own Pairformer stack
from the checkpoint hyper-parameters and stays unscaled.

## AlphaFold 3

AlphaFold 3 is built from its own source, so the patch goes into your checkout:

```bash
prs patch-af3 /path/to/alphafold3
```

It adds a `--beta` flag to `run_alphafold.py`, which passes the value to the
Evoformer through the `PRS_BETA` environment variable.
