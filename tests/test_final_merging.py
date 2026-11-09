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


def three_clusters():
    """Clusters 0 and 1 agree on the marker genes m*, clusters 0 and 2 on the other genes o*"""
    rng = np.random.default_rng(0)
    up, down = [6.0, 1.0, 6.0, 1.0], [1.0, 6.0, 1.0, 6.0]
    means = {0: up + up, 1: up + down, 2: down + up}
    x = np.vstack([rng.poisson(means[c], (30, 8)) for c in (0, 1, 2)]).astype(np.float32)
    genes = [f"m{i}" for i in range(4)] + [f"o{i}" for i in range(4)]
    norm = ad.AnnData(csr_matrix(np.log1p(x)), obs=pd.DataFrame(index=[f"c{i}" for i in range(90)]),
                      var=pd.DataFrame(index=genes))
    return norm, [list(range(30)), list(range(30, 60)), list(range(60, 90))]


def first_neighbor_pairs(norm, clusters, **kwargs):
    """The cluster pairs of the first nearest-neighbor search of merge_clusters (k = 1), and the genes of its space"""
    from transcriptomic_clustering import merging
    seen = []
    real = merging.update_similarity

    def spy(cluster_means, similarity, changed):
        out = real(cluster_means, similarity, changed)
        seen.append((list(cluster_means.columns), set(merging.nearest_pairs_by_similarity(out, 1))))
        return out

    assign = {i: c for i, c in enumerate(clusters)}
    by_obs = np.concatenate([[i] * len(c) for i, c in enumerate(clusters)]).astype(float)
    with patch.object(merging, "update_similarity", spy):
        merging.merge_clusters(norm, kwargs.pop("reduced", None), assign, by_obs, k=1, n_markers=None, **kwargs)
    return seen[0]


def test_merge_clusters_neighbors_in_the_marker_space():
    """reduced_genes: the nearest clusters by the correlation of the cluster means over these genes (scrattch.hicat's
    final merge); without it, by the reduced space"""
    norm, clusters = three_clusters()
    genes, pairs = first_neighbor_pairs(norm, clusters, reduced_genes=["m0", "m1", "m2", "m3"])
    assert genes == ["m0", "m1", "m2", "m3"] and (0, 1) in pairs and (0, 2) not in pairs
    others = ad.AnnData(np.asarray(norm[:, ["o0", "o1", "o2", "o3"]].X.toarray()), obs=norm.obs)
    genes, pairs = first_neighbor_pairs(norm, clusters, reduced=others)
    assert (0, 2) in pairs and (0, 1) not in pairs


def test_final_merge_neighbor_space_markers():
    """neighbor_space='markers': no PCA, and merge_clusters gets the marker genes as its neighbor space"""
    norm, clusters = three_clusters()
    seen = {}

    def no_pca(*args, **kwargs):
        raise AssertionError("no PCA in the marker space")

    def spy(**kwargs):
        seen.update(kwargs)
        return {0: clusters[0]}, set()

    with patch.object(tc, "pca", no_pca), patch.object(tc, "merge_clusters", spy):
        final_merging.final_merge(norm, clusters, marker_genes=["m0", "m1", "m2", "m3"], neighbor_space="markers")
    assert seen["reduced_genes"] == ["m0", "m1", "m2", "m3"] and seen["adata_reduced"] is None
    with pytest.raises(ValueError):
        final_merging.final_merge(norm, clusters, marker_genes=["m0"], neighbor_space="other")


def test_gene_cluster_means_file_backed_with_a_one_cell_cluster(tmp_path):
    """The means of the marker space, read in chunks from a file-backed adata, with a cluster of one cell (whose
    variance get_cluster_means cannot compute)"""
    from transcriptomic_clustering import merging
    norm, _ = three_clusters()
    norm.write_h5ad(tmp_path / "norm.h5ad")
    backed = ad.read_h5ad(tmp_path / "norm.h5ad", backed="r")
    clusters = {0: list(range(29)), 1: [29], 2: list(range(30, 90))}
    by_obs = np.array([0] * 29 + [1] + [2] * 60)
    got = merging.gene_cluster_means(backed, clusters, by_obs, ["m0", "o3"], chunk_size=7)
    x = norm[:, ["m0", "o3"]].X.toarray()
    want = np.vstack([x[c].mean(0) for c in clusters.values()])
    assert list(got.columns) == ["m0", "o3"] and np.allclose(got.to_numpy(), want)
    backed.file.close()
