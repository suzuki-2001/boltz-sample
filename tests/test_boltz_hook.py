"""Tests for the runtime pair representation scaling hook."""

import types

import pytest

from prs import boltz_hook


class FakePairformer:
    """Stand-in for boltz PairformerModule.forward(s, z, ...)."""

    def __init__(self):
        self.seen = []

    def forward(self, s, z, **kwargs):
        self.seen.append(z)
        return s, z


def fake_trunk_namespace():
    return types.SimpleNamespace(__name__="fake.trunk", PairformerModule=FakePairformer)


def test_trunk_input_is_scaled():
    namespace = fake_trunk_namespace()
    boltz_hook.patch_module(namespace, 0.5)

    module = namespace.PairformerModule()
    module.forward(s=None, z=2.0)

    assert module.seen == [3.0]


def test_other_namespaces_are_left_alone():
    trunk = fake_trunk_namespace()
    confidence = fake_trunk_namespace()
    boltz_hook.patch_module(trunk, 0.5)

    module = confidence.PairformerModule()
    module.forward(s=None, z=2.0)

    assert module.seen == [2.0]
    assert confidence.PairformerModule is FakePairformer


def test_patching_twice_scales_once():
    namespace = fake_trunk_namespace()
    boltz_hook.patch_module(namespace, 0.5)
    boltz_hook.patch_module(namespace, 0.5)

    module = namespace.PairformerModule()
    module.forward(s=None, z=2.0)

    assert module.seen == [3.0]


def test_subclass_keeps_the_original_name():
    namespace = fake_trunk_namespace()
    boltz_hook.patch_module(namespace, 0.5)

    assert namespace.PairformerModule.__name__ == "FakePairformer"
    assert issubclass(namespace.PairformerModule, FakePairformer)


def test_missing_pairformer_raises():
    namespace = types.SimpleNamespace(__name__="fake.trunk")

    with pytest.raises(RuntimeError, match="PairformerModule"):
        boltz_hook.patch_module(namespace, 0.5)


def test_zero_beta_installs_nothing():
    assert boltz_hook.install(0.0) is None
