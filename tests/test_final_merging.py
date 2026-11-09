from unittest.mock import patch

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_series_equal
from scipy.sparse import csr_matrix

import transcriptomic_clustering as tc
from transcriptomic_clustering.final_merging import sample_clusters
from transcriptomic_clustering.final_merging import FinalMergeKwargs
from transcriptomic_clustering import final_merging


@pytest.fixture
def adata():
    return ad.AnnData(
        np.arange(32, dtype=np.float32).reshape(8, 4),
        obs=pd.DataFrame(index=[f"cell_{i}" for i in range(8)]),
        var=pd.DataFrame(index=[f"gene_{i}" for i in range(4)]),
    )


@pytest.mark.parametrize(
    "cluster_dict,n_samples,expected_cells",
    [
        ({"large": [0, 1, 2, 3], "equal": [4, 5], "small": [6]},
         2, ["cell_0", "cell_2", "cell_4", "cell_5", "cell_6"]),
        ({"small": [1, 3]}, 3, ["cell_1", "cell_3"]),
        ({"equal": [1, 3]}, 2, ["cell_1", "cell_3"]),
        ({"all": list(range(8))}, 8, [f"cell_{i}" for i in range(8)]),
        ({"large": [0, 1, 2, 3]}, 0, []),
        ({"empty": []}, 2, []),
        ({}, 2, []),
    ],
)
def test_sample_clusters_selects_only_sampled_cells(
    adata, cluster_dict, n_samples, expected_cells
):
    cell_mask = sample_clusters(
        adata, cluster_dict, n_samples_per_clust=n_samples, random_seed=123
    )

    expected_mask = pd.Series(
        adata.obs_names.isin(expected_cells), index=adata.obs_names
    )
    assert_series_equal(cell_mask, expected_mask)
    assert cell_mask.index[cell_mask].tolist() == expected_cells
    assert cell_mask.sum() == len(expected_cells)


@pytest.mark.parametrize(
    "marker_genes",
    [
        ["gene_3", "gene_1"],
        {"gene_3", "gene_1"},
        ["gene_2"],
        [],
        ["gene_0", "gene_1", "gene_2", "gene_3"],
    ],
)
def test_final_merge_passes_only_sampled_cells_and_marker_genes_to_pca(
    adata, marker_genes
):
    class StopAfterPCA(Exception):
        pass

    with patch.object(tc, "pca", side_effect=StopAfterPCA) as pca:
        with pytest.raises(StopAfterPCA):
            tc.final_merge(
                adata,
                [[0, 1, 2, 3], [4, 5], [6, 7]],
                marker_genes=marker_genes,
                n_samples_per_clust=2,
                random_seed=123,
            )

    pca.assert_called_once()
    assert pca.call_args.args[0] is adata
    cell_mask = pca.call_args.kwargs["cell_select"]
    gene_mask = pca.call_args.kwargs["gene_mask"]
    expected_cells = ["cell_0", "cell_2", "cell_4", "cell_5", "cell_6", "cell_7"]
    assert_series_equal(
        cell_mask,
        pd.Series(adata.obs_names.isin(expected_cells), index=adata.obs_names),
    )
    assert_series_equal(
        gene_mask,
        pd.Series(adata.var_names.isin(marker_genes), index=adata.var_names),
    )
    assert cell_mask.index[cell_mask].tolist() == expected_cells
    assert set(gene_mask.index[gene_mask]) == set(marker_genes)
    assert cell_mask.sum() == len(expected_cells)
    assert gene_mask.sum() == len(marker_genes)


@pytest.mark.parametrize("latent", [False, True])
def test_final_merge_real_marker_set(separated_expression, latent):
    data = separated_expression
    data.obsm["latent"] = np.column_stack(
        [np.repeat([0., 100.], 30), np.tile(np.arange(30), 2) / 100]
    )
    clusters, markers = tc.final_merge(
        data, [list(range(15)), list(range(15, 30)), list(range(30, 60))],
        marker_genes={f"gene_{i}" for i in range(20)},
        random_seed=42,
        final_merge_kwargs=FinalMergeKwargs(
            pca_kwargs={"n_comps": 3},
            filter_pcs_kwargs={"method": "elbow", "max_pcs": 2},
            latent_kwargs={"latent_component": "latent"} if latent else {},
            merge_clusters_kwargs={"n_markers": 5},
        ),
    )
    assert {frozenset(c) for c in clusters} == {
        frozenset(range(30)), frozenset(range(30, 60))
    }
    assert markers
    assert markers <= {f"gene_{i}" for i in range(20)}


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
