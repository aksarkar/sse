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

# Dynamically build the list of wrappers for use in sse.evaluate
methods = [x for x in
           [getattr(sys.modules[__name__], y) for y in dir(sys.modules[__name__])]
           if isinstance(x, types.FunctionType)]
