import numpy as np
import pandas as pd
import anndata as ad
import pytest
from scipy.sparse import csr_matrix
from unittest.mock import patch

import transcriptomic_clustering as tc
from transcriptomic_clustering import onestep_clustering as oc
from transcriptomic_clustering.onestep_clustering import OnestepKwargs, onestep_clust


@pytest.mark.parametrize("latent", [False, True])
def test_onestep_real_separated_clusters(separated_expression, latent):
    data = separated_expression
    if latent:
        data.obsm["latent"] = np.column_stack(
            [np.repeat([0., 100.], 30), np.tile(np.arange(30), 2) / 100]
        )
    kwargs = OnestepKwargs(
        highly_variable_kwargs={"max_genes": 40},
        pca_kwargs={"n_comps": 5},
        filter_pcs_kwargs={"method": "elbow", "max_pcs": 3},
        latent_kwargs={"latent_component": "latent"} if latent else {},
        cluster_louvain_kwargs={"k": 5},
        merge_clusters_kwargs={"n_markers": 5},
    )
    clusters, markers = onestep_clust(data, kwargs, random_seed=42)
    assert {frozenset(c) for c in clusters} == {
        frozenset(range(30)), frozenset(range(30, 60))
    }
    assert markers
    assert markers <= {f"gene_{i}" for i in range(20)}
    assert len(set(np.concatenate(clusters))) == data.n_obs


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
