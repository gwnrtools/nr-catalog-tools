from sxs import time_series as sxs_time_series
from waveformtools.waveformtools import interp_resam_wfs

class TimeSeries(sxs_time_series):
	''' Extended SXS time series '''
	def __init__(self):
		super().__init__()

	def interpolate_in_amp_phase(self, new_time, k=3, kind=None):
		''' Interpolate in amplitude and phase
		using a variety of interpolation methods.
		
		Paramters
		---------
		new_time: array_like
			  The new time axis to interpolate onto.

        k: int, optional
           The order of interpolation when 
            `scipy.interpolated.InterpolatedUnivariateSpline` is used.
            This gets preference over `kind` parameter when both are 
            specified. The default is 3.

        kind: str, optional
              The interpolation kind parameter of `scipy.interpolate.interp1d`
               is used. Default is None i.e. the parameter `k` will be used
                instead.
		See Also
		--------
		waveformtools.waveformtools.interp_resam_wfs :
			The function that interpolates in amplitude
			and phases using scipy interpolators.

		scipy.interpolate.CubicSpline:
			One of the possible methods that can
			be used for interpolation.
		scipy.interpolate.interp1d:
			Can be used in linear, quadratic and cubic mode.
		scipy.interpolate.InterpolatedUnivariateSpline:
			Can be used with orders k from 1 to 5.
        
        Notes
        -----
        These interpolation methods ensure that the
        interpolated function passes through all the
        data points.

		'''
        
        resam_data = TimeSeries(interp_resam_wfs(np.array(self), self.time, new_time_axis, k=k, kind=kind), )

        metadata = self._metadata.copy()
        metadata["time"] = new_time
        metadata["time_axis"] = self.time_axis

        return type(self)(resam_data, **metadata)
