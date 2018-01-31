import os
import pytest
import tempfile
import sse

@pytest.fixture
def tmpdir():
  return tempfile.TemporaryDirectory()

@pytest.fixture
def simulate(tmpdir):
  os.chdir(tmpdir.name)
  s = sse.simulation.Simulation(p=1000, seed=0)
  s.sample_effects(pve=0.15)
  x, y = s.sample_gaussian(n=500)
  return x, y

def test_dap(simulate):
  x, y = simulate
  sse.wrapper.dap(x, y)

def test_finemap(simulate):
  x, y = simulate
  res = sse.wrapper.finemap(x, y)
  assert res.shape == (x.shape[1], 3)

def test_exact_pip(simulate):
  x, y = simulate
  res = sse.wrapper.exact_pip(x, y, effect_var=1)
