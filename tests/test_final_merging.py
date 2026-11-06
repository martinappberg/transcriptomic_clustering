import numpy as np
import pandas as pd
import anndata as ad
import pytest
from scipy.sparse import csr_matrix
from mock import patch

import transcriptomic_clustering as tc
from transcriptomic_clustering import final_merging


@pytest.fixture
def adata():
    rng = np.random.default_rng(0)
    x = rng.poisson(2, (90, 30)).astype(np.float32)
    return ad.AnnData(csr_matrix(x), obs=pd.DataFrame(index=[f"c{i}" for i in range(90)]),
                      var=pd.DataFrame(index=[f"g{i}" for i in range(30)]))


def test_sample_clusters_selects_only_the_sample(adata):
    """The mask is True for the sampled cells only (at most n_samples_per_clust of each cluster): a
    pd.Series(index=..., dtype=bool) without data is all True with pandas 1.4 or later"""
    clusters = {0: list(range(30)), 1: list(range(30, 60)), 2: list(range(60, 90))}
    mask = final_merging.sample_clusters(adata, clusters, n_samples_per_clust=5, random_seed=1)
    assert 0 < mask.sum() <= 15
    for cells in clusters.values():
        assert 0 < mask.iloc[cells].sum() <= 5


def test_final_merge_pca_uses_the_marker_genes_only(adata):
    """The PCA of final_merge takes the marker genes and the sampled cells, not every gene and cell"""
    seen = {}

    def spy(adata, gene_mask=None, cell_select=None, **kwargs):
        seen["genes"] = int(np.asarray(gene_mask).sum())
        seen["cells"] = int(np.asarray(cell_select).sum())
        raise StopIteration

    clusters = [list(range(30)), list(range(30, 60)), list(range(60, 90))]
    with patch.object(tc, "pca", spy):
        with pytest.raises(StopIteration):
            final_merging.final_merge(adata, clusters, marker_genes=["g1", "g2", "g3"], n_samples_per_clust=5,
                                      random_seed=1)
    assert seen["genes"] == 3
    assert 0 < seen["cells"] <= 15
