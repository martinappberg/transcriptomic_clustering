import os
import pytest

import numpy as np
import scipy as scp
import pandas as pd
import scanpy as sc
import anndata as ad
import transcriptomic_clustering as tc


@pytest.fixture
def adata():
    rng = np.random.default_rng(1)
    n_obs = 20
    n_vars = 10
    obs = pd.DataFrame(index=[f'cell_{i}' for i in range(n_obs)])
    var = pd.DataFrame(index=[f'gene_{i}' for i in range(n_vars)])

    adata = ad.AnnData(rng.random((20,10)), obs=obs, var=var)
    return adata

@pytest.fixture
def pcs(adata):
    return pd.DataFrame(-np.eye(10,5), index=adata.var_names, columns=[f'PC{i}' for i in range(5)])

@pytest.fixture
def mean(adata):
    return pd.DataFrame(np.zeros((10,)), index=adata.var_names)

@pytest.fixture
def x_exp(adata):
    return -adata.X[:,0:5].copy()


def test_simple_proj(adata, mean, pcs, x_exp):
    ad_proj = tc.project(adata, pcs, mean)

    np.testing.assert_allclose(ad_proj.X, x_exp)
    assert ad_proj.obs.index.equals(adata.obs.index)

def test_chunk_proj(adata, mean, pcs, x_exp, tmpdir_factory):
    tmpdir = str(tmpdir_factory.mktemp("test_proj"))
    input_file_name = os.path.join(tmpdir, "input.h5ad")
    ad.AnnData(adata.X, obs=adata.obs, var=adata.var).write(input_file_name)
    adata = sc.read_h5ad(input_file_name, backed='r')

    ad_proj = tc.project(adata, pcs, mean, chunk_size=2)
    adata.file.close()

    np.testing.assert_allclose(ad_proj.X, x_exp)

def test_file_notchunked_proj(adata, mean, pcs, x_exp, tmpdir_factory):
    tmpdir = str(tmpdir_factory.mktemp("test_proj"))
    input_file_name = os.path.join(tmpdir, "input.h5ad")
    ad.AnnData(scp.sparse.csr_matrix(adata.X), obs=adata.obs, var=adata.var).write(input_file_name)
    adata = sc.read_h5ad(input_file_name, backed='r')

    ad_proj = tc.project(adata, pcs, mean)
    adata.file.close()

    np.testing.assert_allclose(ad_proj.X, x_exp)


def batch_adata(levels=("a", "b"), n_obs=120, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n_obs, 10))
    batch = np.array([levels[i % len(levels)] for i in range(n_obs)])
    x[batch == levels[-1]] += 3.0                     # a batch shift
    obs = pd.DataFrame({'batch': batch}, index=[f'cell_{i}' for i in range(n_obs)])
    return ad.AnnData(x, obs=obs, var=pd.DataFrame(index=[f'gene_{i}' for i in range(10)]))


def test_project_with_harmony():
    """Harmony on the projection (harmony_batch_key): the batch shift falls, info records the step"""
    pytest.importorskip('harmonypy')
    a = batch_adata(n_obs=600)
    pcs = pd.DataFrame(np.eye(10, 5), index=a.var_names, columns=[f'PC{i}' for i in range(5)])
    mean = pd.DataFrame(np.zeros((10,)), index=a.var_names)
    plain = tc.project(a, pcs, mean)
    info = {}
    corrected = tc.project(a, pcs, mean, harmony_batch_key='batch', harmony_kwargs={'random_state': 0}, info=info)
    shift = lambda m: np.abs(np.asarray(m.X)[a.obs.batch == 'b'].mean(0) - np.asarray(m.X)[a.obs.batch == 'a'].mean(0)).mean()
    assert corrected.shape == plain.shape
    assert shift(corrected) < 0.75 * shift(plain)
    assert info['n_pcs'] == 5 and info['batch_levels'] == 2 and info['harmony_ran'] and info['harmony_iterations'] >= 1


def test_project_with_harmony_skips_one_batch_and_small_groups():
    """One batch level, or fewer cells than harmonypy's 2 clusters need: no correction, and info says so"""
    pytest.importorskip('harmonypy')
    for a in (batch_adata(levels=("a",)), batch_adata(n_obs=44)):
        pcs = pd.DataFrame(np.eye(10, 5), index=a.var_names, columns=[f'PC{i}' for i in range(5)])
        mean = pd.DataFrame(np.zeros((10,)), index=a.var_names)
        info = {}
        out = tc.project(a, pcs, mean, harmony_batch_key='batch', info=info)
        assert np.allclose(out.X, tc.project(a, pcs, mean).X)
        assert info['harmony_ran'] is False
