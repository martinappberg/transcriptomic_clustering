
import pandas as pd
import numpy as np
import pytest
from transcriptomic_clustering.markers import select_marker_genes
from transcriptomic_clustering.de_ebayes import de_pairs_ebayes


def test_select_marker_genes():
    n_genes = 9
    n_clusters = 5
    genes = [f"Gene_{i}" for i in range(n_genes)]
    clusters = [i for i in range(n_clusters)]

    clust_means = pd.DataFrame(
        index=clusters,
        columns=genes,
        data=np.asarray([
            [10, 10, 10, 0, 0, 0, 0, 0, 0],
            [10, 10, 10, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 10, 10, 10, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 10, 10, 10],
            [10, 10, 10, 10, 10, 10, 10, 10, 10]
        ])
    )
    clust_vars = pd.DataFrame(
        index=clusters,
        columns=genes,
        data=np.asarray([
            [3, 3, 3, 5, 5, 5, 1, 1, 1],
            [3, 3, 3, 5, 5, 5, 1, 1, 1],
            [3, 3, 3, 5, 5, 5, 1, 1, 1],
            [3, 3, 3, 5, 5, 5, 1, 1, 1],
            [3, 3, 3, 5, 5, 5, 1, 1, 1]
        ])
    )
    # Clusters 0 and 1 are same so no marker genes
    # Cluster 3 and 4 don't meet present threshold, so no markers
    # Clust 0 and 2 have marker genes 0,1 3,4
    present_clust_means = pd.DataFrame(
        index=clusters,
        columns=genes,
        data=np.asarray([
            [0.8, 0.8, 0.4, 0, 0, 0, 0, 0, 0],
            [0.8, 0.8, 0.4, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0.8, 0.8, 0.4, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0.4, 0.4, 0.4],
            [0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3]
        ])
    )
    cluster_assignments = {
        0: [1,2,3,24,25,26],
        1: [4,5,6,7,8,9],
        2: [10,11,12,13,14],
        3: [15,16,17,18],
        4: [19,20,21,22,23],
    }
    
    markers = select_marker_genes(
        cluster_assignments=cluster_assignments,
        cluster_means=clust_means,
        cluster_variances=clust_vars,
        present_cluster_means=present_clust_means,
        n_markers=6,
        thresholds={
            'q1_thresh': 0.5,
            'q2_thresh': None,
            'cluster_size_thresh': 3,
            'qdiff_thresh': 0.7,
            'padj_thresh': 0.01,
            'lfc_thresh': 1.0,
            'score_thresh': 150,
            'low_thresh': 1,
            'min_genes': 5
        },
        de_method='ebayes'
    )
    expected = {"Gene_0", "Gene_1", "Gene_3", "Gene_4"}
    assert markers == expected


@pytest.mark.parametrize("method", ["chisq", "ebayes"])
@pytest.mark.parametrize("return_frame", [False, True])
@pytest.mark.parametrize("column_index", ["plain", "categorical", "ordered"])
def test_marker_names_for_unsorted_gene_columns(method, return_frame, column_index):
    columns = ["up", "same", "down"]
    if column_index != "plain":
        columns = pd.CategoricalIndex(
            columns, categories=["same", "up", "down"],
            ordered=column_index == "ordered",
        )
    means = pd.DataFrame([[10., 5., 0.], [0., 5., 10.]], columns=columns)
    present = pd.DataFrame([[0.9, 0.5, 0.1], [0.1, 0.5, 0.9]], columns=columns)
    result = select_marker_genes(
        {0: list(range(100)), 1: list(range(100, 200))},
        means, pd.DataFrame([[3., 5., 1.], [3., 5., 1.]], columns=columns), present,
        thresholds={
            "q1_thresh": 0.5, "qdiff_thresh": 0.7,
            "cluster_size_thresh": 6, "padj_thresh": 0.05, "lfc_thresh": 1,
        },
        n_markers=1, de_method=method, return_markers_df=return_frame,
    )
    if return_frame:
        assert result.loc[(0, 1), "up_genes"] == ["up"]
        assert result.loc[(0, 1), "down_genes"] == ["down"]
    else:
        assert result == {"up", "down"}

def _pair_with_genes_out_of_pvalue_order():
    """Two clusters of 10 cells. In each direction the column order of the genes (big_noisy, mid, small_precise) is the
    reverse of their p-value order, and the gene with the largest |lfc| has the largest p-value (variance 40)."""
    genes = ["up_big_noisy", "up_mid", "up_small_precise", "down_big_noisy", "down_mid", "down_small_precise"]
    means = pd.DataFrame([[8., 5., 3., 0., 0., 0.], [0., 0., 0., 8., 5., 3.]], columns=genes)
    variances = pd.DataFrame([[40., 1., 0.05, 40., 1., 0.05]] * 2, columns=genes)
    present = pd.DataFrame([[0.6, 0.8, 0.95, 0.05, 0.05, 0.02], [0.05, 0.05, 0.02, 0.6, 0.8, 0.95]], columns=genes)
    assignments = {0: list(range(10)), 1: list(range(10, 20))}
    return assignments, means, variances, present


@pytest.mark.parametrize("method", ["chisq", "ebayes"])
def test_marker_genes_in_pvalue_order(method):
    """scrattch.hicat's select_markers takes the first n.markers up and down genes of each pair in the order of
    de_stats_pair, order(df$pval, -abs(df$lfc)): by p-value, not by gene column."""
    assignments, means, variances, present = _pair_with_genes_out_of_pvalue_order()
    thresholds = {"q1_thresh": 0.5, "qdiff_thresh": 0.7, "cluster_size_thresh": 3, "padj_thresh": 0.1,
                  "lfc_thresh": 1}
    kwargs = dict(cluster_assignments=assignments, cluster_means=means, cluster_variances=variances,
                  present_cluster_means=present, thresholds=thresholds, de_method=method)
    pairs = select_marker_genes(**kwargs, return_markers_df=True)
    assert pairs.loc[(0, 1), "up_genes"] == ["up_small_precise", "up_mid", "up_big_noisy"]
    assert pairs.loc[(0, 1), "down_genes"] == ["down_small_precise", "down_mid", "down_big_noisy"]
    assert select_marker_genes(**kwargs, n_markers=1) == {"up_small_precise", "down_small_precise"}
    if method == "ebayes":
        serial = de_pairs_ebayes([(0, 1)], means, variances, present, {0: 10, 1: 10}, thresholds)
        assert serial.loc[(0, 1), "up_genes"] == ["up_small_precise", "up_mid", "up_big_noisy"]
        assert serial.loc[(0, 1), "down_genes"] == ["down_small_precise", "down_mid", "down_big_noisy"]
