import pandas as pd
import numpy as np
import sse
import tabix

def read_vcf(genotype_files, chrom, start, end):
  x = []
  for record in tabix.open(genotype_files[chrom - 1]).query(str(chrom), start, end):
    dose = [_.split(':')[2] if _ != '.' else -1 for _ in record[9:]]
    x.append([float(_) if _ != '.' else -1 for _ in dose])
  return np.array(x)

def _call_n(x, f, n, *args, **kwargs):
  """Call f(i, x) n times, i=1..n

  """
  return [f(i, x, *args, **kwargs) for i in range(n)]

def _read_data(row, genotype_files, window):
  if row['strand'] == '+':
    start = row['start'] - window
    end = row['start']
  else:
    start = row['start']
    end = row['start'] + window

  x = read_vcf(genotype_files, int(row['chr'][2:]), start, end).T
  x = np.ma.masked_equal(x, -1)
  x = x.filled(x.mean())
  return x

def _generate_pheno(trial, x, num_causal):
  s = sse.simulation.Simulation(p=x.shape[1], seed=trial)
  s.estimate_mafs(x)
  s.sample_effects(pve=0.15, annotation_params=[(num_causal, 1)], permute=True)
  y = s.compute_liabilities(x).reshape(-1, 1)

  return y

def _pip_calibration(trial, row, genotype_files, num_causal=1, window=int(1e5), **kwargs):
  """Single simulation trial"""
  x = _read_data(row, genotype_files, window)
  y = _generate_pheno(trial, x, num_causal=num_causal)
  m = sse.model.GaussianSSE().fit(x, y, **kwargs)
  corr = {other.__name__: m.pip_df.agg(np.sum, axis=1).corr(other(x, y)['pip'])
          for other in sse.wrapper.methods}
  corr['num_snps'] = x.shape[1]
  return pd.Series(corr)

def pip_calibration(genes, genotype_files, num_genes=100, num_trials=10, num_causal=1, seed=0, **kwargs):
  """Evaluate the calibration of PIP

  genes - DataFrame of genes
  num_genes - number of genes to randomly sample cis-windows around
  num_trials - number of trials per gene
  num_causal - number of causal variants per gene
  kwargs - keyword arguments to GaussianSSE.fit

  """
  result = (genes
            .sample(num_genes, random_state=seed)
            .apply(_call_n, f=_pip_calibration, n=num_trials, genotype_files=genotype_files, num_causal=num_causal, **kwargs, axis=1)
            .apply(pd.DataFrame))
  return pd.concat(result.to_dict())
