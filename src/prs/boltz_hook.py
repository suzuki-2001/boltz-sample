"""Pair representation scaling for an unmodified Boltz installation.

Multiplies the pair representation by (1 + beta) once before the trunk
Pairformer stack, which is the change in patches/boltz-2.2.1.patch applied at
runtime instead of in the source.

The trunk models import `PairformerModule` into their own module namespace, so
rebinding that name to a scaled subclass reaches the trunk alone. The confidence
head imports the same class separately and stays unscaled.
"""

from __future__ import annotations

_SCALED = "_prs_pair_scaling"


def scaled_pairformer(base: type, beta: float) -> type:
    """Return a Pairformer subclass that scales the pair representation."""

    class ScaledPairformerModule(base):
        def forward(self, s, z, *args, **kwargs):
            return super().forward(s, z * (1.0 + beta), *args, **kwargs)

    ScaledPairformerModule.__name__ = base.__name__
    ScaledPairformerModule.__qualname__ = base.__qualname__
    setattr(ScaledPairformerModule, _SCALED, True)
    return ScaledPairformerModule


def patch_module(namespace, beta: float) -> None:
    """Point `namespace.PairformerModule` at a scaled subclass."""
    base = getattr(namespace, "PairformerModule", None)
    if base is None:
        raise RuntimeError(
            f"{getattr(namespace, '__name__', namespace)} has no "
            "'PairformerModule'. Pair representation scaling supports boltz 2.2.1."
        )
    if getattr(base, _SCALED, False):
        return
    namespace.PairformerModule = scaled_pairformer(base, beta)


def install(beta: float) -> None:
    """Scale the pair representation of every Boltz model built from now on.

    beta = 0 leaves Boltz untouched and reproduces stock inference.
    """
    if beta == 0.0:
        return

    try:
        from boltz.model.models import boltz1, boltz2
    except ImportError as exc:
        raise RuntimeError(
            "boltz is not installed. Install it with: pip install boltz==2.2.1"
        ) from exc

    patch_module(boltz1, beta)
    patch_module(boltz2, beta)
