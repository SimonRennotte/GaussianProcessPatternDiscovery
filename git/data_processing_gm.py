import math

import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

torch.set_default_tensor_type(torch.DoubleTensor)


class GMSDGP:
	def __init__(self, n_gm=10, n_pts=4096, sn_range=(1e-3, 1)):
		self.n_gm = n_gm
		self.n_pts = n_pts

		self.sn_range = torch.Tensor(sn_range)

		self.mixtures_weight = None
		self.mixtures_scale = None
		self.mixtures_mean = None
		self.sn = None

		self.y_min = None
		self.y_max = None

		self.min_sn = sn_range[0]

	def init_params(self, x_train, y_train):
		"""

		"""
		dists = (x_train[None, ...].transpose(1, 0) - x_train[None, ...]).flatten(0, 1)
		dist_min = torch.where(dists.eq(0.0), torch.tensor(1.0e10, dtype=x_train.dtype, device=x_train.device), dists).abs().min(0).values
		dist_max = dists.max(0).values
		# Inverse of lengthscales should be drawn from truncated Gaussian | N(0, max_dist^2) |
		mixtures_scale = torch.randn(self.n_gm, self.x.shape[1]).mul_(dist_max).abs_().reciprocal_()
		# Draw means from Unif(0, 0.5 / minimum distance between two points)
		mixtures_mean = torch.rand(self.n_gm, self.x.shape[1]).mul_(0.5).div(dist_min)
		mixtures_weight = y_train.std().div(self.n_gm)[None].repeat(self.n_gm)
		sn = torch.Tensor(self.sn_range[0] + torch.rand(1) * (self.sn_range[1] - self.sn_range[0]))
		return mixtures_weight, mixtures_scale, mixtures_mean, sn

	def transform_constraints(self, mixtures_weight, mixtures_scale, mixtures_mean, sn):
		"""

		"""
		mixtures_weight_transf = torch.log(mixtures_weight)
		mixtures_scale_transf = torch.log(mixtures_scale)
		mixtures_mean_transf = torch.log(mixtures_mean)
		sn_transf = torch.log(sn - self.min_sn)
		return mixtures_weight_transf, mixtures_scale_transf, mixtures_mean_transf, sn_transf

	def inverse_constraints(self, mixtures_weight_transf, mixtures_scale_transf, mixtures_mean_transf, sn_t):
		mixtures_weight = torch.exp(mixtures_weight_transf)
		mixtures_scale = torch.exp(mixtures_scale_transf)
		mixtures_mean = torch.exp(mixtures_mean_transf)
		sn = torch.exp(sn_t) + self.min_sn
		return mixtures_weight, mixtures_scale, mixtures_mean, sn

	def train(self, x, y, n_restarts=10, n_iters=10, lr=1e-3):
		if type(x) != torch.Tensor:
			x = torch.Tensor(x)
		if type(y) != torch.Tensor:
			y = torch.Tensor(y)
		if x.dim() == 1:
			x = x.reshape((x.shape[0], 1))
		self.x = x
		self.y_max = y.max(0).values
		self.y_min = y.min(0).values
		self.y = (y - self.y_min) / (self.y_max - self.y_min)
		self.train_gm(x, y, n_restarts, n_iters, lr)

	def train_gm(self, x, y, n_restarts=10, n_iters=10, lr=1e-3):
		best_marg_lk = math.inf
		best_mixtures_weight = None
		best_mixtures_scale = None
		best_mixtures_mean = None
		best_sn = None
		I = torch.eye(self.x.shape[0], self.x.shape[0])
		dist_matrix = self.compute_dist(self.x, self.x)
		for idx_restart in range(n_restarts):
			w_i, mixtures_scale, mixtures_mean_i, sn_i = self.init_params(x, y)
			mixtures_weight_transf, mixtures_scale_transf, mixtures_mean_transf, sn_t = \
				self.transform_constraints(w_i, mixtures_scale, mixtures_mean_i, sn_i)
			sn_t.requires_grad = True
			mixtures_mean_transf.requires_grad = True
			mixtures_scale_transf.requires_grad = True
			mixtures_weight_transf.requires_grad = True
			optimizer = torch.optim.LBFGS([
				{'params': [mixtures_weight_transf, mixtures_scale_transf, mixtures_mean_transf, sn_t]},  # Includes GaussianLikelihood parameters
			], lr=lr, line_search_fn='strong_wolfe')
			try:
				for idx_iter in tqdm(range(n_iters)):
					def closure():
						optimizer.zero_grad()
						mixtures_weight, mixtures_scale, mixtures_mean, sn = self.inverse_constraints(mixtures_weight_transf, mixtures_scale_transf, mixtures_mean_transf, sn_t)
						k = self.compute_k_cos(dist_matrix, mixtures_scale, mixtures_weight, mixtures_mean)
						sn2 = sn.pow(2)
						k_cholesky = torch.linalg.cholesky(k + sn2 * I)
						log_k_det = 2 * k_cholesky.diagonal().log().sum()
						fit_term = self.y.t() @ torch.cholesky_solve(self.y[..., None], k_cholesky)
						log_marg_lk = 0.5 * (fit_term + log_k_det + k.shape[0] * math.log(2 * math.pi))

						if torch.isnan(log_marg_lk):
							raise(ValueError("NaN in log_k_det"))
						log_marg_lk.backward()
						#print(" - best mixtures_weight: " + str(mixtures_weight.detach().numpy()) + " - best v: " + str(v.detach().numpy()) +
						#		" - best mu: " + str(mu.detach().numpy()) + " - best sn: " + str(sn2.sqrt().item()))
						print("marg lk: " + str(log_marg_lk.item()) + " - fit: " + str(fit_term.item()) + " - det: " + str(log_k_det.item()))
						return log_marg_lk

					log_marg_lk = optimizer.step(closure)
					if torch.isnan(log_marg_lk):
						break

					if log_marg_lk < best_marg_lk:
						best_mixtures_weight, best_mixtures_scale, best_mixtures_mean, best_sn = \
							self.inverse_constraints(mixtures_weight_transf, mixtures_scale_transf, mixtures_mean_transf, sn_t)
						best_marg_lk = log_marg_lk

			except Exception as e:
				print(e)
		self.mixtures_weight = best_mixtures_weight.detach()
		self.mixtures_scale = best_mixtures_scale.detach()
		self.mixtures_mean = best_mixtures_mean.detach()
		self.sn = best_sn.detach()
		print("best marg lk: " + str(best_marg_lk) + " - best mixtures_weight: " + str(self.mixtures_weight.detach().numpy()) +
			  " - best mixtures_scale: " + str(self.mixtures_scale.detach().numpy()) + " - best mixtures_mean: " + str(self.mixtures_mean.detach().numpy()) +
			  " - best sn: " + str(self.sn.item()))
		self.repr_spectre(self.mixtures_scale, self.mixtures_weight, self.mixtures_mean)
		print("Computing inverse of k...")
		self.k = self.compute_k_cos(dist_matrix, self.mixtures_scale, self.mixtures_weight, self.mixtures_mean)
		self.inv_k = torch.cholesky_inverse(torch.linalg.cholesky(self.k + self.sn.pow(2) * I))

	@staticmethod
	def compute_dist(x, x_star):
		return (x[None, ...].transpose(1, 0) - x_star[None, ...]).abs()

	def compute_k_cos(self, dist_matrix, mixtures_scale, mixtures_weight, mu):
		"""
		Compute the covariance matrix given the distance matrix between points
		distance matrix shape is (N, N, p)
		where N is the number of points, p the number of features
		and is resized in shape (q, N, N , p) where q is the number of gaussians in the mixture model
		"""
		v = mixtures_scale.pow(2)
		dist_matrix = dist_matrix[None, ...]
		v_matrix = v[:, None, None, :]
		mu_matrix = mu[:, None, None, :]
		w_matrix = mixtures_weight[:, None, None]
		temp_prod = torch.exp(-2 * math.pi ** 2 * dist_matrix.pow(2) * v_matrix) * torch.cos(2 * math.pi * dist_matrix * mu_matrix)
		temp_sum = temp_prod.prod(-1)
		compo_corr_matrix = w_matrix * temp_sum
		corr_matrix = compo_corr_matrix.sum(0)
		return corr_matrix

	def repr_spectre(self, mixtures_scale, mixtures_weight, mixtures_mean):
		v = mixtures_scale.pow(2)
		for idx_feat in range(mixtures_scale.shape[1]):
			plt.figure()
			max_mu = (mixtures_mean[:, 0] + 2 * v[:, 0]).max()
			x_axis = torch.arange(0, max_mu, max_mu / 1e4)[:, None].repeat(1, v.shape[0])
			spectre_compo = 1 / torch.sqrt(2 * math.pi * v[None, :, idx_feat]) * \
							torch.exp(- torch.pow((x_axis - mixtures_mean[None, :, idx_feat]), 2) / \
						torch.sqrt(2 * v[None, :, idx_feat]))
			spectre = torch.log((mixtures_weight[None, ...] * spectre_compo).sum(-1))
			plt.plot(x_axis, spectre)
			plt.title("Feature " + str(idx_feat))
			plt.xlabel("frequency")
			plt.ylabel("log amplitude (dB)")
			plt.show()

	def predict(self, xs):
		if type(xs) != torch.Tensor:
			xs = torch.Tensor(xs)
		if xs.dim() == 1:
			xs = xs.reshape((xs.shape[0], 1))
		dist_mat_xsxs = self.compute_dist(xs, xs)
		dist_mat_xsx = self.compute_dist(xs, self.x)
		k_xs_xs = self.compute_k_cos(dist_mat_xsxs, self.mixtures_scale, self.mixtures_weight, self.mixtures_mean)
		k_xs_x = self.compute_k_cos(dist_mat_xsx, self.mixtures_scale, self.mixtures_weight, self.mixtures_mean)
		coef_pred = k_xs_x @ self.inv_k
		mean_pred = coef_pred @ self.y * (self.y_max - self.y_min) + self.y_min
		reduc_cov = k_xs_x @ self.inv_k @ k_xs_x.t()
		cov_pred = (k_xs_xs - reduc_cov) * (self.y_max - self.y_min) ** 2
		return mean_pred.detach().numpy(), cov_pred.detach().numpy()


def main():
	gm_obj = GMSDGP(n_gm=10, sn_range=(1e-3, 1e-1))
	data = np.array(pd.read_csv("monthly_in_situ_co2_mlo.csv", header=56))
	y = data[:, 4].astype(np.float32)
	x = data[:, 0].astype(np.float32) * 12 + data[:, 1].astype(np.float32)
	inds_keep = np.nonzero(y > 0)[0]

	y = y[inds_keep]
	x = x[inds_keep]

	prop_train = 0.75
	num_train = int(x.shape[0] * prop_train)
	x_train = x[:num_train]
	y_train = y[:num_train]
	x_val = x[num_train:]
	y_val = y[num_train:]

	gm_obj.train(x=x_train, y=y_train, n_iters=100, n_restarts=10, lr=1e-3)

	max_x = max(x_train)
	min_x = min(x_train)
	border = (max_x - min_x) * 0.25
	min_repr = min_x - border
	max_repr = max_x + border
	tick = (max_x - min_x) / 1000
	x_pred = torch.arange(min_repr, max_repr, tick)

	y_pred, y_cov = gm_obj.predict(xs=x_pred)

	plt.figure()
	plt.fill_between(x_pred / 12, y_pred - 3 * np.sqrt(y_cov.diagonal()), y_pred + 3 * np.sqrt(y_cov.diagonal()), color="y")
	plt.plot(x_pred / 12, y_pred)
	plt.scatter(x_train / 12, y_train, color="k", marker='x', s=10)
	plt.scatter(x_val / 12, y_val, color="g", marker='x', s=10)
	plt.show()


if __name__ == "__main__":
	main()


