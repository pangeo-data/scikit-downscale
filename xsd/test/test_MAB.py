import xsd.qm
import xarray as xr
from pathlib import Path

path = Path('/home/david/projects/test_xsd_MAB/')
name_fut = path / "tasmax_RACMO22T_AFR-44_MOHC-HadGEM2-ES_rcp26_4qqmap.nc"
name_ref = path / "tasmax_RACMO22T_AFR-44_MOHC-HadGEM2-ES_rcp26_ref_4qqmap.nc"
name_obs = path / "tasmax_farakoba.nc"

v = "tasmax"

fut = xr.open_dataset(name_fut).squeeze()[v]
ref = xr.open_dataset(name_ref).squeeze()[v]
obs = xr.open_dataset(name_obs).squeeze()[v]

if v == "pr":

    kind = "*"
    obs.interpolate_na(dim="time")

else:
    obs = obs + 273.15
    obs = obs.interpolate_na(dim="time")
    kind = "+"

nq = 20
group = 'time.dayofyear'  # 'time.dayofyear'
detrend_order = 4

qmf = xsd.qm.train(ref, obs, nq, group, kind=kind, detrend_order=detrend_order)

pp_fut = xsd.qm.predict(fut, qmf, interp=True, detrend_order=detrend_order)
