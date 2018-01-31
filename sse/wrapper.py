import contextlib
import io
import numpy as np
import os
import pandas as pd
import tempfile
import types
import sse
import subprocess
import sys

def dap(x, y, **kwargs):
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

def exact_pip(x, y, effect_var, **kwargs):
  """Return exact PIP assuming one causal variant

  The PIP of SNP j is proportional to the Bayes factor of SNP j, which we can
  compute from the marginal MLE (Wakefield Genet Epi 2009).

  Fit linear regression models in parallel using matrix operations (Sikorska et
  al., BMC Bioinformatics 2013).

  """

  n, p = x.shape
  var = np.diag(x.T.dot(x)).reshape(-1, 1) + 1e-8  # Needed for monomorphic SNPs
  beta = y.T.dot(x).T / var
  df = n - 1
  s = ((y ** 2).sum() - beta ** 2 * var) / df
  V = s / var
  bf = np.sqrt((V + effect_var) / V) * np.exp(-.5 * beta * beta / V * effect_var / (V + effect_var))
  pip = bf / bf.sum()
  return pd.DataFrame({'snp': ['snp{}'.format(i) for i in range(p)], 'pip': pip.ravel()}).set_index('snp')

# Dynamically build the list of wrappers for use in sse.evaluate
methods = [x for x in
           [getattr(sys.modules[__name__], y) for y in dir(sys.modules[__name__])]
           if isinstance(x, types.FunctionType)]
