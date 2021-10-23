# Gaussian Process Kernels for Pattern Discovery and Extrapolation with pytorch.

### Abstract of the paper

Gaussian processes are rich distributions over functions, which provide a Bayesian nonparametric approach to smoothing and interpolation. We introduce simple closed form kernels that can be used with Gaussian processes to discover patterns and enable extrapolation. These kernels are derived by modelling a spectral density -- the Fourier transform of a kernel -- with a Gaussian mixture. The proposed kernels support a broad class of stationary covariances, but Gaussian process inference remains simple and analytic. We demonstrate the proposed kernels by discovering patterns and performing long range extrapolation on synthetic examples, as well as atmospheric CO2 trends and airline passenger data. We also show that we can reconstruct standard covariances within our framework.

---
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

