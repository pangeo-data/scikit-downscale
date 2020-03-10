import os
import glob
import scipy
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import pandas as pd
from pathlib import Path


##### CREATE FILE FORM SHELL######
import xclim
from xclim import subset


# QUANTILE MAPPING FUNCTION
from xsd.dqm import train, predict


############################
# PLOT FUT,QQMAP,OBS FOR A LOCALISATION AND VARIABLE

#SET PARAMETERS

v = "pr"

#fut_path = '/exec/marcbour/sim_climat/senegal/fao/all/regrid/pr/pr_RACMO22T_AFR-44_ICHEC-EC-EARTH_rcp26_4qqmap.nc'
#ref_path = '/exec/marcbour/sim_climat/senegal/fao/all/regrid/pr/pr_RACMO22T_AFR-44_ICHEC-EC-EARTH_rcp26_ref_4qqmap.nc'
#obs_path = "/exec/marcbour/sim_climat/senegal/fao/all/obs/" + "pr_all_4qqmap.nc"

base = Path("/home/david/projects/test_xsd_MAB")
fut_path = base / "pr_RACMO22T_AFR-44_ICHEC-EC-EARTH_rcp26_4qqmap.nc"
ref_path = base / "pr_RACMO22T_AFR-44_ICHEC-EC-EARTH_rcp26_ref_4qqmap.nc"
obs_path = base / "pr_all_4qqmap.nc"


fut = xr.open_dataset(fut_path)[v]
ref = xr.open_dataset(ref_path)[v]
obs = xr.open_dataset(obs_path)[v]

if v == "pr":

    kind = "*"

else:
    obs = obs + 273.15
    obs = obs.interpolate_na(dim="time")
    kind = "+"

#up_qmf = 5
nq = 40
#timeint = 1
group = "time.month"
detrend_order=None

qmf = train(ref, obs, nq, group, kind=kind, detrend_order=detrend_order)
pp_fut = predict(fut, qmf, interp=False, detrend_order=detrend_order) #THIS IS NOT WORKING
