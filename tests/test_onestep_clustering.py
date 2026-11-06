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
