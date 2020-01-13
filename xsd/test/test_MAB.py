import xsd.xsd.qm
import xarray as xr
from pathlib import Path

path = Path('/home/david/projects/test_xsd_MAB/')
path = Path("/exec/marcbour/sim_climat/burkina/pcci/leo/regrid/tasmin/")
name_fut = path / "tasmin_RACMO22T_AFR-44_MOHC-HadGEM2-ES_rcp26_4qqmap.nc"
name_ref = path / "tasmin_RACMO22T_AFR-44_MOHC-HadGEM2-ES_rcp26_ref_4qqmap.nc"
name_obs = path / "tasmin_leo.nc"
name_obs = "/exec/marcbour/sim_climat/burkina/pcci/leo/obs/"+"tasmin_leo.nc"

v = "tasmin"

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

nq = 50
group = 'time.month'  # 'time.dayofyear'
detrend_order = 1

qmf = train(ref, obs, nq, group, kind=kind, detrend_order=detrend_order)
pp_fut = predict(fut, qmf, interp=True, detrend_order=detrend_order)

pp_fut.plot()
ref.plot(alpha=0.5)
obs.plot(alpha=0.5)

plt.legend(["qqmap","ref","obs"])