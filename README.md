# Gaussian Process Kernels for Pattern Discovery and Extrapolation with pytorch.

### Abstract of the paper

Gaussian processes are rich distributions over functions, which provide a Bayesian nonparametric approach to smoothing and interpolation. We introduce simple closed form kernels that can be used with Gaussian processes to discover patterns and enable extrapolation. These kernels are derived by modelling a spectral density -- the Fourier transform of a kernel -- with a Gaussian mixture. The proposed kernels support a broad class of stationary covariances, but Gaussian process inference remains simple and analytic. We demonstrate the proposed kernels by discovering patterns and performing long range extrapolation on synthetic examples, as well as atmospheric CO2 trends and airline passenger data. We also show that we can reconstruct standard covariances within our framework.

---

## Usage

x, y = read_co2_mlo()

x_train, y_train, x_val, y_val, x_test, y_test = split_train_val_test(x, y, prop_train=0.8, prop_val=0.0)

gp = Gp_pattern.GpPatternDiscovery(x_train, y_train, n_gm=10, sn_range_init=(1e-3, 1), sn_range_limits=(1e-3, math.inf), max_f_init=0.2)

gp.train_hyperparams(x_val=None, y_val=None, n_iters=250, n_restarts=5, lr=1e-3, prop_in=0.4)

x_pred = generate_array_pred(x_max=np.max((x_train.max(), x_test.max())), x_min=np.min((x_train.min(), x_test.min())), border_prop=0.1)

y_pred, y_pred_cov = gp.predict(xs=x_pred)

plot_results(x_train, x_val, x_test, x_pred, y_train, y_val, y_test, y_pred, y_pred_cov)

gp.plot_spectral_density(max_f=1 / (x_train[1:] - x_train[:-1]).min(0))

gp.plot_cov_fct(x=x_train)

## Results

### CO2 prediction 

#### Squared exponential kernel

Results:
<p align="middle">
  <img src="https://github.com/SimonRennotte/GaussianProcessPatternDiscovery/blob/master/images/co2_se_results.png?" width="80%" />
</p>

Spectral density of the kernel:
<p align="middle">
  <img src="https://github.com/SimonRennotte/GaussianProcessPatternDiscovery/blob/master/images/co2_se_freq.png?" width="80%" />
</p>

Correlation of points with distance:
<p align="middle">
  <img src="https://github.com/SimonRennotte/GaussianProcessPatternDiscovery/blob/master/images/co2_se_corr.png?" width="80%" />
</p>

#### Kernel for pattern discovery

Results:
<p align="middle">
  <img src="https://github.com/SimonRennotte/GaussianProcessPatternDiscovery/blob/master/images/co2_pat_results.png?" width="80%" />
</p>

Spectral density of the kernel:
<p align="middle">
  <img src="https://github.com/SimonRennotte/GaussianProcessPatternDiscovery/blob/master/images/co2_pat_freq.png?" width="80%" />
</p>

Correlation of points with distance:
<p align="middle">
  <img src="https://github.com/SimonRennotte/GaussianProcessPatternDiscovery/blob/master/images/co2_pat_corr.png?" width="80%" />
</p>

### Recovering sinc pattern

#### Squared exponential kernel

Results:
<p align="middle">
  <img src="https://github.com/SimonRennotte/GaussianProcessPatternDiscovery/blob/master/images/sinc_se_results.png?" width="80%" />
</p>

Spectral density of the kernel:
<p align="middle">
  <img src="https://github.com/SimonRennotte/GaussianProcessPatternDiscovery/blob/master/images/sinc_se_freq.png?" width="80%" />
</p>

Correlation of points with distance:
<p align="middle">
  <img src="https://github.com/SimonRennotte/GaussianProcessPatternDiscovery/blob/master/images/sinc_se_corr.png?" width="80%" />
</p>

#### Kernel for pattern discovery

Results:
<p align="middle">
  <img src="https://github.com/SimonRennotte/GaussianProcessPatternDiscovery/blob/master/images/sinc_pat_results.png?" width="80%" />
</p>

Spectral density of the kernel:
<p align="middle">
  <img src="https://github.com/SimonRennotte/GaussianProcessPatternDiscovery/blob/master/images/sinc_pat_freq.png?" width="80%" />
</p>

Correlation of points with distance:
<p align="middle">
  <img src="https://github.com/SimonRennotte/GaussianProcessPatternDiscovery/blob/master/images/sinc_pat_corr.png?" width="80%" />
</p>

