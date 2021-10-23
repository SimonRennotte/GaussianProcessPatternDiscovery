import math

import torch
from tqdm import tqdm

from gp import Gp

torch.set_default_tensor_type(torch.DoubleTensor)


class GpSe(Gp):
	"""
	Class that inherits from the base Gp class, and defines the function init_params, compute_kernel and 
	compute_spectral_density for the squared exponential kernel.
	The parameters of the kernel are:
		"l": characteristic lengthscales
		"sf": signal std
	"""
	def __init__(self, x, y, l_range_limits=((0., math.inf),), sf_range_limits=(0, math.inf), sn_range_limits=(1e-3, math.inf),
				 l_range_init=((0, 10.),), sf_range_init=(0., 1), sn_range_init=(1e-3, 1)):
		"""
		Init the object
		Args:
			x (numpy.array or torch.Tensor): feature of the data that will be stored in the gp memory
			y (numpy.array or torch.Tensor): target values of the data that will be stored in the gp memory
			sn_range_init (tuple): min and max value of the signal noise used for initialization (random uniform). Shape=(2,)
			sn_range_limits (tuple): min and max value allowed for the signal noise. Shape=(2,)
			sf_range_init (tuple): min and max value of the signal std used for initialization (random uniform). Shape=(2,)
			sf_range_limits (tuple): min and max value allowed for the signal std. Shape=(2,)
			l_range_init (tuple or tuple of tuple or numpy.array): min and max value of the signal noise used for
										initialization (random uniform).
										Shape=(Nl, 2) to set lengthscales init differently for each dimension or (2,) otherwise
			l_range_limits (tuple or tuple of tuple or numpy.array): min and max value allowed for the lengthscales.
										Shape=(Nl, 2) to set lengthscales limits differently for each dimension or (2,) otherwise
		"""
		super(GpSe, self).__init__(x, y, sn_range_limits, sn_range_init)
		# l_range_limits can be specified for all dimensions separately or for all dimensions at the same time
		# but in either can, the parameter is set to be of shape (Nl, 2)
		l_range_limits = torch.Tensor(l_range_limits)
		if l_range_limits.dim == 1:
			l_range_limits = l_range_limits[None, :].repeat(1, x.shape[1])

		l_range_init = torch.Tensor(l_range_init)
		if l_range_init.dim == 1:
			l_range_init = l_range_init[None, :].repeat(1, x.shape[1])

		# the keys in the dict kernel_params_limits and kernel_params_init must correspond to the keys in kernel_params
		# l_range can be specified for each dimension separately or for all dimensions at the same time
		self.kernel_params_limits = {"l": l_range_limits, "sf": torch.Tensor(sf_range_limits)}
		self.kernel_params_init = {"l": l_range_init, "sf": torch.Tensor(sf_range_init)}

	def init_params(self):
		"""
		Define and return (random) initial values for the parameters of the kernel.
		Returns:
			kernel_params (dict): Contains the parameters name and value. The values must be torch.Tensor
									The keys are:
										"l": characteristic lengthscales
										"sf": signal std
			sn (torch.Tensor): Std of the likelihood
		"""
		kernel_params = {}
		kernel_params["l"] = torch.Tensor(self.kernel_params_init["l"][:, 0] +
						 torch.rand(self.kernel_params_init["l"].shape[0]) *
										  (self.kernel_params_init["l"][:, 1] - self.kernel_params_init["l"][:, 0]))

		kernel_params["sf"] = torch.Tensor(self.kernel_params_init["sf"][0] + torch.rand(1) *
										   (self.kernel_params_init["sf"][1] - self.kernel_params_init["sf"][0]))

		sn = torch.Tensor(self.sn_range_init[0] + torch.rand(1) * (self.sn_range_init[1] - self.sn_range_init[0]))

		return kernel_params, sn

	@staticmethod
	def compute_kernel(dist_mat, kernel_params):
		"""
		Compute the covariance matrix given the distance matrix between points
		distance matrix shape is (N, N, p)
		where N is the number of points, p the number of features
		"""
		return kernel_params["sf"].pow(2) * torch.exp(- dist_mat.pow(2) / (2 * kernel_params["l"].pow(2))).prod(-1)

	@staticmethod
	def compute_spectral_density(max_f, idx_feat, n_pts, kernel_params):
		"""
		Compute and return the spectral density of the kernel
		Args:
			max_f (float): maximum frequency for which to compute the spectral density
			idx_feat (int): feature index for which to represent the density function.
							max=fn where fn is the number of features of x
			n_pts (int): number of points between 0 and max_f for which to compute the spectral density
			kernel_params (dict): Contains the kernel parameters.
									The keys are:
										"l": lengthscales
										"sf": signal std
		Returns:
			f_axis (torch.Tensor): axis containing the value of the points at which the spectral density have been computed
			spectre (torch.Tensor): log of the spectral density, computed at the points of f_axis
		"""
		l2 = kernel_params['l'].pow(2)
		sf2 = kernel_params['sf'].pow(2)
		with torch.no_grad():
			try:
				max_f_feat = max_f[idx_feat]
			except:
				max_f_feat = max_f
			f_axis = torch.arange(0, max_f_feat, max_f_feat / n_pts)
			spectral_density = (2 * math.pi * l2[idx_feat]).pow(l2.shape[0] / 2) * \
							   torch.exp(-2 * math.pow(math.pi, 2) * f_axis.pow(2) * l2[idx_feat])
			spectral_density = torch.log(spectral_density)
		return f_axis, spectral_density

