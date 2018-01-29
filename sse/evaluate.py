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
  return s, y

def max_abs_error(trial, row, genotype_files, num_causal=1, window=int(1e5), **kwargs):
  """Single simulation trial

  Return pd.Series. For each method in sse.wrapper.methods:

  method - maximum error between SSE PIP and that method's PIP
  method_pip - method's PIP at maximum error
  sse_method_pip - SSE estimate of PIP at maximum error
  effect - true effect size at variant with maximum error
  maf - minor allele frequency at variant with maximum error

  """
  x = _read_data(row, genotype_files, window)
  s, y = _generate_pheno(trial, x, num_causal=num_causal)
  m = sse.model.GaussianSSE().fit(x, y, **kwargs)
  sse_pip = m.pip_df.apply(lambda x: 1 - np.prod(1 - x), axis=1)
  assert sse_pip.apply(lambda x: 0 <= x <= 1).all()
  other_pip = {other.__name__: other(x, y)['pip'] for other in sse.wrapper.methods}

  error = {('error', k): max(abs(sse_pip - v)) for k, v in other_pip.items()}
  # The Series sse_pip, v, (sse_pip - v) are all indexed like 'snpN'. But the
  # simulation is indexed by N
  snpname = {k: abs(sse_pip - v).idxmax() for k, v in other_pip.items()}
  pip_at_error = {('other_pip', k): other_pip[k][v] for k, v in snpname.items()}
  sse_pip_at_error = {('sse_pip', k): sse_pip[v] for k, v in snpname.items()}
  for (_, k), v in error.items():
    assert np.isclose(abs(pip_at_error[('other_pip', k)] - sse_pip_at_error[('sse_pip', k)]), v)

  index = {k: int(v[3:]) for k, v in snpname.items()}
  effect = {('effect', k): s.theta[v] for k, v in index.items()}
  maf = {('maf', k): s.maf[v] for k, v in index.items()}

  w = x - x.mean()
  w /= w.std()
  cov = w.T.dot(w) / w.shape[0]
  assert cov.shape == (w.shape[1], w.shape[1])
  ld_at_error = {('ld', k): cov[~np.isclose(s.theta, 0), v].max() for k, v in index.items()}

  result = error.copy()
  result.update(effect)
  result.update(maf)
  result.update(pip_at_error)
  result.update(sse_pip_at_error)
  result.update(ld_at_error)
  result[('num_snps', '')] = x.shape[1]
  return pd.Series(result)

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
            .apply(_call_n, f=max_abs_error, n=num_trials, genotype_files=genotype_files, num_causal=num_causal, **kwargs, axis=1)
            .apply(pd.DataFrame))
  return pd.concat(result.to_dict())
