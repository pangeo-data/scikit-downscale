import xesmf as xe

def apply_weights(regridder, input_data):
    regridder._grid_in = None
    regridder._grid_out = None
    result = regridder(input_data)
    return result
