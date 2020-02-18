import numpy as np
import xarray as xr
import pytest
from xsd import qm, dqm


class TestQM:
    @pytest.mark.parametrize("group", ["time.month", "time.dayofyear", "time.season"])
    def test_train(self, tas_series, group):
        n = 10000
        r = np.random.rand(n)
        ref = tas_series(r)
        fut_add = tas_series(r + 2)
        qs = qm.train(ref, fut_add, 20, group)
        d = qs['target'] - qs['source']
        np.testing.assert_array_almost_equal(d, 2)
        assert qs.group == group

    def test_add_cyclic(self, qds_month):
        q = qm.add_cyclic(qds_month, "month")
        assert len(q['month']) == 14

    def test_q_bounds(self, qds_month):
        q = qm.add_q_bounds(qds_month)
        assert len(q['quantile']) == len(qds_month['quantile']) + 2


class TestDQM:
    def test_simple(self, tas_series):
        n = 10000
        r = np.random.rand(n)
        ref = tas_series(r)
        fut_add = tas_series(r + 2)
        d = dqm.train(ref, fut_add, 20, "time.month", "+", detrend_order=None)
        np.testing.assert_array_almost_equal(d, 2)

        out = dqm.predict(ref, d, detrend_order=None)
        np.testing.assert_array_almost_equal(out - ref, 2)

        fut_mul = tas_series(r * 2)
        d = dqm.train(ref, fut_mul, 20, "time.month", "*")
        np.testing.assert_array_almost_equal(d, 2)

        out = dqm.predict(ref, d, detrend_order=None)
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
        d = dqm.train(ref, fut_add, 1, freq, "+")
        a = d.sel(quantile=0.4, method="nearest")
        b = 5 * np.sin(2 * np.pi * (np.arange(l) + 0.5) / l + phi)
        if freq == "time.season":
            a = a[np.array([0, 2, 1, 3])]
        np.testing.assert_array_almost_equal(a, b, 0)

        if freq == "time.dayofyear":
            out = dqm.predict(ref, d, interp=True, detrend_order=None)
            np.testing.assert_array_almost_equal(out, fut_add, 0)

        elif freq == "time.month":
            # Note that there is a lot of intra-month variability within the correction.
            out = dqm.predict(ref, d, interp=True, detrend_order=None)
            np.testing.assert_array_almost_equal(out.resample(time="M").mean(), fut_add.resample(time="M").mean(), 0)

        elif freq == "time.season":
            with pytest.raises(NotImplementedError):
                out = dqm.predict(ref, d, interp=True, detrend_order=None)

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
        qf = dqm.train(ref, fut, 10, 't', '+')

        # Add trend
        xtrend = dqm.polyval(xcoefs, ref.t)
        pref = xtrend + ref
        detrended, est_trend, ecoefs = dqm.detrend(pref, deg=deg, dim='t')

        np.testing.assert_array_almost_equal(ecoefs, coefs, 5)
        np.testing.assert_array_almost_equal(detrended, ref, 2)
        np.testing.assert_array_almost_equal(est_trend / xtrend, 1, 2)

        # Quantile on trended values
        pqf = dqm.train(pref, fut, 10, 't', '+', detrend_order=deg)

        # Confirm that the trend does not affect the quantile mapping
        np.testing.assert_array_almost_equal(qf, pqf)

    def test_3d(self, tas_series):
        nt, nx, ny = 1000, 3, 2
        r = np.random.rand(nt, nx, ny)
        ref = tas_series(r)
        delta = xr.DataArray([[2, 3, 4], [4, 5, 6]], dims=("lat", "lon"))
        fut_add = ref + delta
        d = dqm.train(ref, fut_add, 20, "time.month", "+", detrend_order=None)
        xr.testing.assert_allclose(d, delta.broadcast_like(d))

        out = dqm.predict(ref, d, detrend_order=None)
        xr.testing.assert_allclose(out - ref, delta.broadcast_like(ref))

        out = dqm.predict(ref, d, interp=True, detrend_order=None)
        xr.testing.assert_allclose(out - ref, delta.broadcast_like(ref))

    def test_window(self, tas_series):
        nt, nx, ny = 10000, 3, 2
        r = np.random.rand(nt, nx, ny)
        ref = tas_series(r)
        fut = ref.copy()
        fut[fut.time.dt.dayofyear.isin([9, 10, 11])] += 10
        d = dqm.train(ref, fut, 15, "time.dayofyear", "+", window=3, detrend_order=None)
        np.testing.assert_allclose(d.sel(dayofyear=30), 0)
        np.testing.assert_allclose(d.sel(dayofyear=10), 10)
        np.testing.assert_allclose(d.sel(dayofyear=[9, 11]).mean(), 6.7, 2)
        np.testing.assert_allclose(d.sel(dayofyear=[8, 12]).mean(), 3.3, 2)


class TestPolyfit():
    def test_simple(self, tas_series):
        da = tas_series(np.arange(100))
        coefs = dqm.polyfit(da, dim="time")
        assert len(coefs) == 2
        assert coefs.sel(degree=1) > 0

        y = dqm.polyval(coefs, da.time)
        np.testing.assert_array_almost_equal(y, da)


class TestPolyval():
    def test_simple(self, tas_series):
        da = tas_series(np.arange(100))
        coefs = dqm.polyfit(da, dim="time")
        assert len(coefs) == 2
        assert coefs.sel(degree=1) > 0

        y = dqm.polyval(coefs, da.time)
        np.testing.assert_array_almost_equal(y, da)


class TestDetrend:
    def test_simple(self, tas_series):
        da = tas_series(np.arange(100)+10)
        detrended, trend, coefs = dqm.detrend(da)

        #np.testing.assert_almost_equal(coefs.sel(degree=1).values, 1)
        np.testing.assert_array_almost_equal(detrended, 0)


class TestGetIndex:
    def test_simple(self, timeda):
        x = dqm.get_index(timeda)
        assert hasattr(x, "dims")

    def test_encoding(self, calendar):
        dt = xr.cftime_range("1970-01-01", periods=10, calendar=calendar)
        time = xr.DataArray(dt, dims=("time", ))
        time.encoding['calendar'] = calendar
        x = dqm.get_index(time)
        assert hasattr(x, "dims")

