# SPDX-License-Identifier: Apache-2.0
"""Behavior tests for the gamma-Ada online boost-parameter controller.

gamma-Ada (UniBoost phase 4) adapts the boost parameter ``gamma`` from the
observed end-to-end response-time tail: a heavy tail (p99 >> p95) pushes gamma
DOWN toward ``gamma_min`` (more shortest-job-first), while a light/flat tail
pushes it UP toward ``gamma_max`` (more FCFS). gamma stays strictly positive so
the ``b_gamma`` divide-by-gamma is always safe. The controller is pure -- no
scheduler or GPU required.
"""

from __future__ import annotations

import random

import pytest

pytest.importorskip("sglang")

from sglang_omni.scheduling.omni_scheduler import GammaAdaController  # noqa: E402


def _drive(ctl: GammaAdaController, ticks: int = 30) -> None:
    for k in range(ticks):
        ctl.maybe_update(now=k + 1)


def test_no_update_before_interval() -> None:
    ctl = GammaAdaController(gamma_init=50.0, interval_s=5.0, min_samples=100)
    for _ in range(200):
        ctl.record(1.0)
    assert ctl.maybe_update(now=1.0) is None  # interval not elapsed
    assert ctl.maybe_update(now=10.0) is not None  # elapsed + enough samples


def test_no_update_below_min_samples() -> None:
    ctl = GammaAdaController(gamma_init=50.0, interval_s=0.0, min_samples=200)
    for _ in range(199):
        ctl.record(1.0)
    assert ctl.maybe_update(now=1.0) is None


def test_heavy_tail_lowers_gamma_toward_sjf() -> None:
    ctl = GammaAdaController(
        gamma_init=100.0, gamma_min=1.0, gamma_max=200.0,
        interval_s=0.0, min_samples=100,
    )
    rng = random.Random(0)
    # ~3% of requests ~100x slower => the [p95, p99] band sits in the spike.
    for i in range(1000):
        ctl.record(1.0 + (100.0 if i % 33 == 0 else rng.uniform(0, 0.05)))
    _drive(ctl)
    assert ctl.gamma < 20.0  # pushed well toward gamma_min
    assert ctl.gamma >= ctl.gamma_min


def test_light_tail_raises_gamma_toward_fcfs() -> None:
    ctl = GammaAdaController(
        gamma_init=20.0, gamma_min=1.0, gamma_max=200.0,
        interval_s=0.0, min_samples=100,
    )
    rng = random.Random(1)
    for _ in range(1000):
        ctl.record(1.0 + rng.uniform(0, 0.02))  # p99 ~= p95
    _drive(ctl)
    assert ctl.gamma > 150.0  # pushed toward gamma_max
    assert ctl.gamma <= ctl.gamma_max


def test_gamma_stays_positive_and_bounded() -> None:
    ctl = GammaAdaController(
        gamma_init=1.0, gamma_min=1.0, gamma_max=200.0,
        interval_s=0.0, min_samples=10,
    )
    for _ in range(100):
        ctl.record(1e6)  # huge but uniform => flat tail
    for k in range(50):
        g = ctl.maybe_update(now=k + 1)
        assert g is None or (ctl.gamma_min <= g <= ctl.gamma_max and g > 0.0)


def test_gamma_min_is_strictly_positive() -> None:
    # Even if a caller passes 0, gamma_min is clamped > 0 (b_gamma divides by it).
    ctl = GammaAdaController(gamma_init=0.0, gamma_min=0.0, gamma_max=10.0)
    assert ctl.gamma_min > 0.0
    assert ctl.gamma > 0.0
