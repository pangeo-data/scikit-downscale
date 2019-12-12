import numpy as np
import pandas as pd
import xarray as xr
import pytest
from xsd import qm


class TestQM:
    def test_simple(self, tas_series):
        n = 10000
        r = np.random.rand(n)
        ref = tas_series(r)
        fut_add = tas_series(r + 2)
        d = qm.train(ref, fut_add, 20, "time.month", "+", detrend_order=None)
        np.testing.assert_array_almost_equal(d, 2)

        out = qm.predict(ref, d, detrend_order=None)
        np.testing.assert_array_almost_equal(out - ref, 2)

        fut_mul = tas_series(r * 2)
        d = qm.train(ref, fut_mul, 20, "time.month", "*")
        np.testing.assert_array_almost_equal(d, 2)

        out = qm.predict(ref, d, detrend_order=None)
        np.testing.assert_array_almost_equal(out / ref, 2)

    @pytest.mark.parametrize(
        "freq,l,phi",
        [
            ("time.season", 4, -np.pi / 6),
            ("time.month", 12, 0),
            ("time.dayofyear", 366, 0),
        ],
    )
    def test_interp(self, tas_series, freq, l, phi):
        n = 10000
        r = np.random.rand(n)
        ref = tas_series(r)
        m = np.sin(ref.time.dt.dayofyear / 365.25 * 2 * np.pi) * 20
        fut_add = tas_series(r + m)
        d = qm.train(ref, fut_add, 1, freq, "+")
        a = d.sel(quantile=0.4, method="nearest")
        b = 20 * np.sin(2 * np.pi * (np.arange(l) + 0.5) / l + phi)
        if freq == "time.season":
            a = a[np.array([0, 2, 1, 3])]
        np.testing.assert_array_almost_equal(a, b, 0)

        if freq != "time.season":
            out = qm.predict(ref, d, interp=True)
            # The later half of December and beginning of January won't match due to the interpolation
            np.testing.assert_array_almost_equal(out, fut_add, 0)

    def test_detrend(self, tas_series, deg=0):
        # Create synthetic series with simple index (not datetime)
        n = 2000
        i1 = 10
        coefs = np.random.rand(deg + 1)
        t = np.arange(i1, i1+n)

        xcoefs = xr.DataArray(coefs, dims=('degree', ), coords={'degree': range(deg, -1, -1)})

        ref = xr.DataArray(np.random.rand(n), dims=('t',),  coords={'t': t})

        # Ensure ref is centered on 0 so the detrended version is comparable.
        ref -= ref.mean()

        fut = ref.where(ref > 0, ref + 1)

        # Quantile mapping on original values
        qf = qm.train(ref, fut, 10, 't', '+')

        # Add trend
        xtrend = qm.polyval(xcoefs, ref.t)
        pref = xtrend + ref
        detrended, est_trend, ecoefs = qm.detrend(pref, deg=deg, dim='t')

        np.testing.assert_array_almost_equal(ecoefs, coefs, 5)
        np.testing.assert_array_almost_equal(detrended, ref, 2)
        np.testing.assert_array_almost_equal(est_trend / xtrend, 1, 2)

        # Quantile on trended values
        pqf = qm.train(pref, fut, 10, 't', '+', detrend_order=deg)

        # Confirm that the trend does not affect the quantile mapping
        np.testing.assert_array_almost_equal(qf, pqf)


class TestPolyfit():
    def test_simple(self, tas_series):
        da = tas_series(np.arange(100))
        coefs = qm.polyfit(da, dim="time")
        assert len(coefs) == 2
        assert coefs.sel(degree=1) > 0

        y = qm.polyval(coefs, da.time)
        np.testing.assert_array_almost_equal(y, da)


class TestDetrend:
    def test_simple(self, tas_series):
        da = tas_series(np.arange(100))
        detrended, trend, coefs = qm.detrend(da)

        np.testing.assert_almost_equal(coefs.sel(degree=1).values, 1)
        np.testing.assert_array_almost_equal(detrended, 0)

