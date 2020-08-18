# AutoCTMC

Automated learning of CTMCs from arbitrary time-series data


### Prerequisites

Tested with Python 3.7

### Currently Implemented
- CTMC sampler
- Learning model from sample paths
- Learning model from incomplete noisy data (smoothing with forward-backward iteration)
- Learning likelihood model (unsupervised learning latentstate-> data) in Gaussian param. family
### TODO
- Add non-parametric likelihood model (VAE, neural Gaussian mixture,...)
- Extend to multi-variate processes (CTBNs)
