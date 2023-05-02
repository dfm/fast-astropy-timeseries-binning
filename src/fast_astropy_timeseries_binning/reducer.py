import sys

import numpy as np

from fast_astropy_timeseries_binning import _fast_astropy_timeseries_binning


sys_is_le = sys.byteorder == "little"
native_code = sys_is_le and "<" or ">"
swapped_code = sys_is_le and ">" or "<"


class reducer:
    def __init__(self, function):
        if function == "mean":
            self.func32 = _fast_astropy_timeseries_binning.reducemean_f32
            self.func64 = _fast_astropy_timeseries_binning.reducemean_f64
        elif function == "rms":
            self.func32 = _fast_astropy_timeseries_binning.reducerms_f32
            self.func64 = _fast_astropy_timeseries_binning.reducerms_f64
        elif function == "ivar":
            self.func32 = _fast_astropy_timeseries_binning.reduceivar_f32
            self.func64 = _fast_astropy_timeseries_binning.reduceivar_f64
        else:
            raise ValueError(f"Unknown function name {function}")

    def reduceat(self, array, indices):
        if hasattr(array, "mask"):
            array = array.filled(np.nan)
        if array.dtype.name == "float32":
            out = np.empty(indices.shape, dtype=np.float32)
            self.func32(array.astype(np.float32), indices, out)
        elif array.dtype.name == "float64":
            out = np.empty(indices.shape, dtype=np.float64)
            self.func64(array, indices, out)
        else:
            raise ValueError(f"Unsupported dtype {array.dtype}")
        return out
