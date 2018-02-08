import contextlib
import io
import numpy as np
import os
import pandas as pd
import tempfile
import types
import scipy.special as spsp
import sse
import subprocess
import sys

def dap(x, y, **kwargs):
  """Return PIP estimated by DAP (Wen et al, Am J Hum Genet 2016)

  Assume no heterogeneity in the samples, and fix the effect size
  hyperparameter grid to [1].

  """
  with open('data.txt', 'w') as f:
    print(*(['pheno', 'pheno', '0'] + list(y.ravel())), file=f)
    for j in range(x.shape[1]):
      print(*(['geno', 'snp{}'.format(j), '0'] + list(x[:,j])), file=f)
  with open('grid.txt', 'w') as f:
    print(0, 1, file=f)
  out, err = subprocess.Popen(['dap', '-d', 'data.txt', '-g', 'grid.txt', '-all'], stdout=subprocess.PIPE).communicate()
  with io.StringIO(str(out, 'utf-8')) as f:
    for line in f:
      if line.startswith('Posterior inclusion probability'):
        next(f)
        return pd.read_table(f, header=None, names=['rank', 'snp', 'pip', 'bf'], sep='\s+').set_index('snp')
  raise RuntimeError('Failed to parse output')

def exact_pip(x, y, effect_var=1, **kwargs):
  """Return exact PIP assuming one causal variant

  The PIP of SNP j is proportional to the Bayes factor of SNP j, which we can
  compute from the marginal MLE (Wakefield Genet Epi 2009).

  Fit linear regression models in parallel using matrix operations (Sikorska et
  al., BMC Bioinformatics 2013).

  """
  n, p = x.shape
  # Ignore SNPs which are monomorphic
  var = np.ma.masked_less(np.diag(x.T.dot(x)).reshape(-1, 1), 1e-3)
  beta = y.T.dot(x).T / var
  df = n - 1
  rss = ((y ** 2).sum() - beta ** 2 * var)
  sigma2 = rss / df
  V = sigma2 / var
  # Wakefield uses BF = p(y | M0) / p(y | M1), but we want its reciprocal
  #
  # [p(y | j causal) / p(y | none causal)] / sum_j [p(y | j causal) / p(y | none causal)]
  logbf = -.5 * np.log((V + effect_var) / V) - (-.5 * np.square(beta) / V * effect_var / (V + effect_var))
  pip = logbf / spsp.logsumexp(logbf.compressed())
  return pd.DataFrame({'snp': ['snp{}'.format(i) for i in range(p)], 'pip': pip.ravel()}).set_index('snp')

def sse_10(x, y, **kwargs):
  """Return SSE PIP assuming 10 causal effects.

  """
  if 'num_causal' in kwargs:
    raise ValueError('Cannot specify num_causal for method "sse_10"')
  m = sse.model.GaussianSSE().fit(x, y, num_effects=10, **kwargs)
  return pd.DataFrame(m.pip_df.apply(lambda x: 1 - np.prod(1 - x), axis=1), columns=['pip'])

# Dynamically build the list of wrappers for use in sse.evaluate
methods = [x for x in
           [getattr(sys.modules[__name__], y) for y in dir(sys.modules[__name__])]
           if isinstance(x, types.FunctionType)]

