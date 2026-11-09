import numpy as np
import pandas as pd
import anndata as ad
from scipy.sparse import csr_matrix
from mock import patch

import transcriptomic_clustering as tc
from transcriptomic_clustering import onestep_clustering as oc


def test_onestep_clust_with_one_community_is_one_cluster():
    """When the community detection finds one community, onestep_clust returns that one cluster (iter_clust then
    stops), and no markers"""
    rng = np.random.default_rng(0)
    adata = tc.normalize(ad.AnnData(csr_matrix(rng.poisson(2, (40, 50)).astype(np.float32)),
                                    obs=pd.DataFrame(index=[f"c{i}" for i in range(40)])))

    def one_community(adata, k, random_seed, **kwargs):
        n = adata.n_obs
        return np.zeros(n), {0: list(range(n))}, None, None

    with patch.object(tc, "cluster_louvain", one_community):
        clusters, markers = oc.onestep_clust(adata, onestep_kwargs=oc.OnestepKwargs(
            pca_kwargs={"cell_select": 500000}, filter_pcs_kwargs={"method": "elbow"}), random_seed=1)
    assert [sorted(c) for c in clusters] == [list(range(40))]
    assert not markers


def _k_of_onestep(n_cells, k=None):
    """The k that onestep_clust passes to cluster_louvain for n_cells cells (a mock records it)."""
    rng = np.random.default_rng(0)
    adata = tc.normalize(ad.AnnData(csr_matrix(rng.poisson(2, (n_cells, 50)).astype(np.float32)),
                                    obs=pd.DataFrame(index=[f"c{i}" for i in range(n_cells)])))
    seen = {}

    def one_community(adata, k, random_seed, **kwargs):
        seen["k"] = k
        seen["n_vars"] = adata.n_vars
        n = adata.n_obs
        return np.zeros(n), {0: list(range(n))}, None, None

    louvain = {} if k is None else {"k": k}
    with patch.object(tc, "cluster_louvain", one_community):
        oc.onestep_clust(adata, onestep_kwargs=oc.OnestepKwargs(
            pca_kwargs={"cell_select": 500000}, filter_pcs_kwargs={"method": "elbow", "max_pcs": 3},
            cluster_louvain_kwargs=louvain), random_seed=1)
    return seen


def test_onestep_clust_k_does_not_depend_on_the_number_of_pcs():
    """The neighbour count is k (15 by default) when the cells allow it, also with fewer principal components than k,
    as scrattch.hicat's onestep_clust: k = pmin(k.nn, round(nrow(rd.dat)/2))"""
    seen = _k_of_onestep(100)
    assert seen["n_vars"] <= 3
    assert seen["k"] == 15
    assert _k_of_onestep(100, k=20)["k"] == 20


def test_onestep_clust_k_is_at_most_half_the_cells():
    """With few cells the neighbour count is half the number of cells (rounded), as scrattch.hicat's onestep_clust"""
    assert _k_of_onestep(20)["k"] == 10
    assert _k_of_onestep(31)["k"] == 15
