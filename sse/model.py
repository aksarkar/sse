"""Sum of single effects regression for Gaussian model

"""
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as spst

class GaussianSSE():
  def __init__(self):
    pass

  def _elbo(self):
    error = np.square(self.y - self.xr.sum(axis=1, keepdims=True)).sum()
    error -= np.square(self.xr).sum()
    # var doesn't depend on factor?
    error += (self.d * (self.pip * (np.square(self.mean) + self.var.T)).sum(axis=0, keepdims=True).T).sum()
    error /= self.residual_var
    error += self.y.shape[0] * np.log(2 * np.pi * self.residual_var)
    error *= -.5

    kl_z = (self.pip * (np.log(self.pip) - np.log(self.prior_prob.T))).sum()
    kl_b = .5 * (self.pip * (1 + np.log(self.effect_var * self.residual_var) - np.log(self.var.T) + (np.square(self.mean) + self.var.T) / (self.effect_var * self.residual_var))).sum()
    update = error - kl_z - kl_b
    return update, error, kl_z, kl_b

  def _update(self):
    xr = self.xr.sum(axis=1, keepdims=True)
    for k in range(self.pip.shape[0]):
      xr -= self.xr[:,k].reshape(-1, 1)
      self.mean[k:k + 1] = (self.var / self.residual_var * (self.xy - self.x.T.dot(xr))).T
      self.pip[k:k + 1] = (self.prior_prob * np.exp(.5 * (np.log(self.var / (self.effect_var * self.residual_var)) + np.square(self.mean[k:k + 1].T) / self.var))).T
      self.pip[k] /= self.pip[k].sum()
      self.xr[:,k] = self.x.dot(self.pip[k] * self.mean[k])
      xr += self.xr[:,k].reshape(-1, 1)

  def _diverged(self):
    if len(self.trace) > 1 and self.trace[-1][0] < self.trace[-2][0]:
      return True
    else:
      return False

  def _converged(self, tol):
    if len(self.trace) > 1 and self.trace[-1][0] - self.trace[-2][0] < tol:
      return True
    else:
      return False

  def fit(self, x, y, effect_var=None, residual_var=None, prior_prob=None, num_effects=1, tol=1e-4, max_epochs=200):
    """Fit the variational approximation assuming fixed hyperparameters

    Parameters:
      x - Centered genotypes (num_samples, num_snps)
      y - Centered phenotypes (num_samples, 1)

    Returns:
      pip - posterior inclusion probability (num_effects x num_snps)
      mean - posterior mean (num_effects x num_snps)
    """
    num_samples, num_snps = x.shape

    self.x = x - x.mean(axis=0)
    self.y = y - y.mean()

    if prior_prob is None:
      prior_prob = np.ones((num_snps, 1)) / num_snps
    if effect_var is None:
      effect_var = 1
    if residual_var is None:
      residual_var = self.y.var()

    self.prior_prob = prior_prob
    self.effect_var = effect_var
    self.residual_var = residual_var

    # Variational parameters
    self.pip = np.zeros((num_effects, num_snps))
    self.mean = np.zeros((num_effects, num_snps))

    # Make sure everything is two dimensional to catch numpy broadcasting gotchas
    self.d = np.einsum('ij,ij->j', self.x, self.x).reshape(-1, 1)
    self.xy = self.x.T.dot(self.y)
    self.xr = np.dot(self.x, (self.pip * self.mean).T)

    # Variational variance depends on a pre-computed quantity
    self.var = (self.effect_var * self.residual_var / (self.effect_var * self.d + 1)).reshape(-1, 1)

    self.trace = []
    for epoch in range(max_epochs):
      self._update()
      self.trace.append(self._elbo())
      if self._diverged():
        raise RuntimeError("ELBO decreased")
      if self._converged(tol):
        return self
    raise RuntimeError("Failed to converge")

  def predict(self, x):
    if self.pip is None:
      raise ValueError("Must fit the model before calling lfsr")
    return (x - x.mean(axis=0)).dot((self.pip * self.mean).sum(axis=0))

  def lfsr(self):
    if self.pip is None:
      raise ValueError("Must fit the model before calling lfsr")
    cdf = spst.norm(loc=self.mean, scale=np.sqrt(self.var)).cdf
    pos_prob = cdf(0)
    return 1 - (self.pip * np.maximum(pos_prob, 1 - pos_prob)).sum(axis=0)

  def plot(self, filename):
    plt.clf()
    fig, ax = plt.subplots(self.pip.shape[0], 1)
    fig.set_size_inches(6, 8)
    for i, a in enumerate(ax):
      a.scatter(np.arange(self.pip.shape[1]), self.pip[i])
      a.set_xlabel('')
      a.set_ylabel('PIP {}'.format(i))
      a.set_yticks([0, .5, 1])
      a.set_xticks([])
    ax[-1].set_xlabel('Genetic variants')
    return plt.gcf()

