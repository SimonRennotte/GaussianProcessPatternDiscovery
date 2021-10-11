import math

import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

torch.set_default_tensor_type(torch.DoubleTensor)


class GMSDGP:
	def __init__(self, n_gm=10, n_pts=4096,
						l_range=((0.01, 3),), sf_range=(1e-2, 25), sn_range=(1e-3, 1)):
		self.n_gm = n_gm
		self.n_pts = n_pts
		self.l_range = torch.Tensor(l_range)
		self.sf_range = torch.Tensor(sf_range)
		self.sn_range = torch.Tensor(sn_range)

		self.l = None
		self.sf = None
		self.sn = None

	def train(self, x, y, n_restarts=10, n_iters=10, lr=1e-3):
		if type(x) != torch.Tensor:
			x = torch.Tensor(x)
		if type(y) != torch.Tensor:
			y = torch.Tensor(y)
		if x.dim() == 1:
			x = x.reshape((x.shape[0], 1))
		if self.l_range.shape[0] != x.shape[1]:
			raise(ValueError("lengthscales size must be the same as the x feature dimension"))
		self.x = x
		self.y = y
		self.train_se(n_restarts, n_iters, lr=lr)

	def train_se(self, n_restarts=10, n_iters=10, lr=1e-3):
		best_marg_lk = math.inf
		best_l = None
		best_sf = None
		best_sn = None
		I = torch.eye(self.x.shape[0], self.x.shape[0])
		dist_matrix = self.compute_dist(self.x, self.x)
		for idx_restart in range(n_restarts):
			l = torch.Tensor(self.l_range[:, 0] + torch.rand(self.l_range.shape[0]) * (self.l_range[:, 1] - self.l_range[:, 0]))
			sf = torch.Tensor(self.sf_range[0] + torch.rand(1) * (self.sf_range[1] - self.sf_range[0]))
			sn = torch.Tensor(self.sn_range[0] + torch.rand(1) * (self.sn_range[1] - self.sn_range[0]))
			l.requires_grad = True
			sf.requires_grad = True
			sn.requires_grad = True
			optimizer = torch.optim.LBFGS([
				{'params': [l, sf, sn]},  # Includes GaussianLikelihood parameters
			], lr=lr, line_search_fn='strong_wolfe')
			try:
				for idx_iter in tqdm(range(n_iters)):
					def closure():
						sf2 = sf.pow(2)
						sn2 = sn.pow(2)
						l2 = l.pow(2)
						optimizer.zero_grad()
						k = sf2 * torch.exp(- dist_matrix / (2 * l2))
						k_cholesky = torch.linalg.cholesky(k + sn2 * I)
						log_k_det = 2 * k_cholesky.diagonal().log().sum()
						fit_term = self.y.t() @ torch.cholesky_solve(self.y[..., None], k_cholesky)
						log_marg_lk = 0.5 * (fit_term + log_k_det + k.shape[0] * math.log(2 * math.pi))

						log_marg_lk.backward()
						print("Log marg: " + str(log_marg_lk.item()) + " - fit: " + str(
							fit_term.item()) + " - det: " + str(log_k_det.item()) + " - l: " + str(l2.sqrt().item()) +
							  " - sf: " + str(sf2.sqrt().item()) + " - sn: " + str(sn2.sqrt().item()))
						return log_marg_lk
					log_marg_lk = optimizer.step(closure)

					if log_marg_lk < best_marg_lk:
						best_l = l
						best_sf = sf
						best_sn = sn
						best_marg_lk = log_marg_lk

			except Exception as e:
				print(e)
		self.l = best_l.detach()
		self.sf = best_sf.detach()
		self.sn = best_sn.detach()
		print("best marg lk: " + str(best_marg_lk) + " - best l: " + str(self.l.item()) +
			  " - best sf: " + str(self.sf.item()) + " - best sn: " + str(self.sn.item()))
		print("Computing inverse...")
		self.k = self.compute_k(self.x, self.x, self.sf, self.l)
		self.inv_k = torch.cholesky_inverse(torch.linalg.cholesky(self.k + self.sn.pow(2) * I))

	@staticmethod
	def compute_dist(x, x_star):
		return (x[None, ...].transpose(1, 0) - x_star[None, ...]).sum(-1).pow(2)

	def compute_k(self, x, x_star, sf, l):
		return sf.pow(2) * torch.exp(- self.compute_dist(x, x_star) / (2 * l.pow(2)))

	def compute_k_cos(self, x, x_star, tau, mu, w):
		dist_matrix = self.compute_dist(x, x_star)
		dist_matrix = dist_matrix[None, None, ...]
		tau = tau[..., None]
		mu = mu[None, ...]
		dist_matrix = torch.exp(-2 * pow(math.pi, 2) * dist_matrix * tau) * torch.cos(2 * math.pi * tau * mu)
		dist_matrix.prod(1)
		dist_matrix = dist_matrix @ w
		dist_matrix.sum(0)
		return self.sf.pow(2) * torch.exp(- dist_matrix / (2 * self.l.pow(2)))

	def predict(self, xs):
		if type(xs) != torch.Tensor:
			xs = torch.Tensor(xs)
		if xs.dim() == 1:
			xs = xs.reshape((xs.shape[0], 1))
		k_xs_xs = self.compute_k(xs, xs, self.sf, self.l)
		k_xs_x = self.compute_k(xs, self.x, self.sf, self.l)
		coef_pred = k_xs_x @ self.inv_k
		mean_pred = coef_pred @ self.y
		reduc_cov = k_xs_x @ self.inv_k @ k_xs_x.t()
		cov_pred = k_xs_xs - reduc_cov
		return mean_pred.detach().numpy(), cov_pred.detach().numpy()


def main():
	gm_obj = GMSDGP(l_range=[[1e-4, 1e-1]], sn_range=(1e-3, 1e-2), sf_range=(1e-1, 1))

	data = np.array(pd.read_csv("monthly_in_situ_co2_mlo.csv", header=56))
	y = data[:, 4].astype(np.float32)
	x = data[:, 0].astype(np.float32) + data[:, 1].astype(np.float32) / 12
	inds_keep = np.nonzero(y > 0)[0]

	y = y[inds_keep]
	x = x[inds_keep]

	y = (y - y.mean()) / (y.max() - y.min())

	prop_train = 0.75
	num_train = int(x.shape[0] * prop_train)
	x_train = x[:num_train]
	y_train = y[:num_train]
	x_val = x[num_train:]
	y_val = y[num_train:]

	gm_obj.train(x=x_train, y=y_train, n_iters=100, n_restarts=5, lr=1e-7)

	max_x = max(x_train)
	min_x = min(x_train)
	border = (max_x - min_x) * 0.25
	min_repr = min_x - border
	max_repr = max_x + border
	tick = (max_x - min_x) / 1000
	x_pred = torch.arange(min_repr, max_repr, tick)

	y_pred, y_cov = gm_obj.predict(xs=x_pred)

	plt.figure()
	plt.fill_between(x_pred, y_pred - 3 * np.sqrt(y_cov.diagonal()), y_pred + 3 * np.sqrt(y_cov.diagonal()), color="y")
	plt.plot(x_pred, y_pred)
	plt.scatter(x_train, y_train, color="k", marker='x', s=10)
	plt.scatter(x_val, y_val, color="g", marker='x', s=10)
	plt.show()


if __name__ == "__main__":
	main()


