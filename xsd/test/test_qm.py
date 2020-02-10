import numpy as np
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
        r = np.random.rand(n) * 10
        ref = tas_series(r)
        m = np.sin(ref.time.dt.dayofyear / 365.25 * 2 * np.pi) * 5
        fut_add = tas_series(r + m)
        d = qm.train(ref, fut_add, 1, freq, "+")
        a = d.sel(quantile=0.4, method="nearest")
        b = 5 * np.sin(2 * np.pi * (np.arange(l) + 0.5) / l + phi)
        if freq == "time.season":
            a = a[np.array([0, 2, 1, 3])]
        np.testing.assert_array_almost_equal(a, b, 0)

        if freq == "time.dayofyear":
            out = qm.predict(ref, d, interp=True, detrend_order=None)
            np.testing.assert_array_almost_equal(out, fut_add, 0)

        elif freq == "time.month":
            # Note that there is a lot of intra-month variability within the correction.
            out = qm.predict(ref, d, interp=True, detrend_order=None)
            np.testing.assert_array_almost_equal(out.resample(time="M").mean(), fut_add.resample(time="M").mean(), 0)

        elif freq == "time.season":
            with pytest.raises(NotImplementedError):
                out = qm.predict(ref, d, interp=True, detrend_order=None)

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

    def test_3d(self, tas_series):
        nt, nx, ny = 1000, 3, 2
        r = np.random.rand(nt, nx, ny)
        ref = tas_series(r)
        delta = xr.DataArray([[2, 3, 4], [4, 5, 6]], dims=("lat", "lon"))
        fut_add = ref + delta
        d = qm.train(ref, fut_add, 20, "time.month", "+", detrend_order=None)
        xr.testing.assert_allclose(d, delta.broadcast_like(d))

        out = qm.predict(ref, d, detrend_order=None)
        xr.testing.assert_allclose(out - ref, delta.broadcast_like(ref))

        out = qm.predict(ref, d, interp=True, detrend_order=None)
        xr.testing.assert_allclose(out - ref, delta.broadcast_like(ref))

    def test_window(self, tas_series):
        nt, nx, ny = 10000, 3, 2
        r = np.random.rand(nt, nx, ny)
        ref = tas_series(r)
        fut = ref.copy()
        fut[fut.time.dt.dayofyear.isin([9, 10, 11])] += 10
        d = qm.train(ref, fut, 15, "time.dayofyear", "+", window=3, detrend_order=None)
        np.testing.assert_allclose(d.sel(dayofyear=30), 0)
        np.testing.assert_allclose(d.sel(dayofyear=10), 10)
        np.testing.assert_allclose(d.sel(dayofyear=[9, 11]).mean(), 6.7, 2)
        np.testing.assert_allclose(d.sel(dayofyear=[8, 12]).mean(), 3.3, 2)


class TestPolyfit():
    def test_simple(self, tas_series):
        da = tas_series(np.arange(100))
        coefs = qm.polyfit(da, dim="time")
        assert len(coefs) == 2
        assert coefs.sel(degree=1) > 0

        y = qm.polyval(coefs, da.time)
        np.testing.assert_array_almost_equal(y, da)


class TestPolyval():
    def test_simple(self, tas_series):
        da = tas_series(np.arange(100))
        coefs = qm.polyfit(da, dim="time")
        assert len(coefs) == 2
        assert coefs.sel(degree=1) > 0

        y = qm.polyval(coefs, da.time)
        np.testing.assert_array_almost_equal(y, da)


class TestDetrend:
    def test_simple(self, tas_series):
        da = tas_series(np.arange(100)+10)
        detrended, trend, coefs = qm.detrend(da)

        #np.testing.assert_almost_equal(coefs.sel(degree=1).values, 1)
        np.testing.assert_array_almost_equal(detrended, 0)


class TestGetIndex:
    def test_simple(self, timeda):
        x = qm.get_index(timeda)
        assert hasattr(x, "dims")

    def test_encoding(self, calendar):
        dt = xr.cftime_range("1970-01-01", periods=10, calendar=calendar)
        time = xr.DataArray(dt, dims=("time", ))
        time.encoding['calendar'] = calendar
        x = qm.get_index(time)
        assert hasattr(x, "dims")

