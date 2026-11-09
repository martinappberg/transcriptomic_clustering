import os
import pytest
from pandas.testing import assert_frame_equal

import numpy as np
import pandas as pd
import anndata as ad
from transcriptomic_clustering import merging
from transcriptomic_clustering import cluster_means as cm
import transcriptomic_clustering as tc
from scipy.sparse import random, csr_matrix
import scanpy as sc

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')


@pytest.fixture
def adata():

    X = np.array([
        [0, 1, 3],
        [0, 6, 2],
        [1, 0, 1],
        [4, 4, 7],
        [4, 0, 0],
        [6, 0, 6],
        [8, 6, 3],
        [2, 1, 6],
        [0, 0, 7],
        [2, 1, 0],
    ])
    n_obs = X.shape[0]
    n_var = X.shape[1]
    cell_names = [f"cell_{i}" for i in range(n_obs)]
    obs = pd.DataFrame(index=cell_names)
    obs['cluster_label'] = ['11', 2, 2, '11', '32', '11', 2, '32', 4, '11']
    var_names = [f"var_{i}" for i in range(n_var)]
    var = pd.DataFrame(index=var_names)
    adata = ad.AnnData(csr_matrix(X), obs=obs, var=var)

    return adata


@pytest.fixture
def clusters(adata):
    cluster_assignments = {
        '11': [0, 3, 5, 9],
        2: [1, 2, 6],
        '32': [4, 7],
        4: [8]
    }

    cluster_means = pd.DataFrame(
        np.array([[3., 1.5, 4.],
                  [3., 4., 2.],
                  [3., 0.5, 3.],
                  [0., 0., 7.]]),
        index = ['11', 2, '32', 4],
        columns=adata.var_names
    )

    present_cluster_means = pd.DataFrame(
        np.array([[0.5, 0.25, 0.75],
                  [(1/3), (2/3), (1/3)],
                  [0.5, 0., .5],
                  [0., 0., 1.]]),
        index = ['11', 2, '32', 4],
        columns=adata.var_names
    )
    cluster_variances = pd.DataFrame(
        np.array([[5, 2.25, 7.5],
                  [(12 + 2/3), 8.0, (2/3)],
                  [1.0, 0.25, 9.0],
                  [0.0, 0.0, 0.0]]),
        index=['11', 2, '32', 4],
        columns=adata.var_names
    )
    return cluster_means, present_cluster_means, cluster_variances, cluster_assignments


def test_merge_two_clusters(adata, clusters):

    cluster_means, present_cluster_means, cluster_variances, cluster_assignments = clusters

    merging.merge_two_clusters(cluster_assignments,
                               label_source=4,
                               label_dest=2,
                               cluster_means=cluster_means,
                               cluster_variances=cluster_variances,
                               present_cluster_means=present_cluster_means)

    expected_cluster_assignments = {
        '11': [0, 3, 5, 9],
        2: [1, 2, 6, 8],
        '32': [4, 7],
    }

    expected_cluster_means = pd.DataFrame(
        np.vstack([
            np.array([3., 1.5, 4.]),
            (np.array([3., 4., 2.])*3 + np.array([0., 0., 7.])*1)/(3+1),
            np.array([3., 0.5, 3.])
        ]),
        index = ['11', 2, '32'],
        columns=adata.var_names
    )

    expected_present_cluster_means = pd.DataFrame(
        np.vstack([
            np.array([0.5, 0.25, 0.75]),
            (np.array([(1/3), (2/3), (1/3)])*3 + np.array([0., 0., 1.])*1)/(3+1),
            np.array([0.5, 0., .5])
        ]),
        index = ['11', 2, '32'],
        columns=adata.var_names
    )
    expected_cluster_variances = pd.DataFrame(
        np.asarray([[5, 2.25, 7.5],
                    [10.694444, 9.333333, 6.694444],
                    [1.0, 0.25, 9.0]]),
        index = ['11', 2, '32'],
        columns=adata.var_names
    )

    assert set(cluster_assignments.keys()) == set(expected_cluster_assignments.keys())
    for k, v in expected_cluster_assignments.items():
        assert np.array_equal(cluster_assignments[k], expected_cluster_assignments[k])

    assert_frame_equal(cluster_means, expected_cluster_means)
    assert_frame_equal(cluster_variances,expected_cluster_variances)
    assert_frame_equal(present_cluster_means, expected_present_cluster_means)


@pytest.mark.parametrize("sizes", [(10, 100), (10, 10)])
@pytest.mark.parametrize("label_source,label_dest", [(0, 1), (1, 0)])
def test_merge_two_clusters_sample_variance(sizes, label_source, label_dest):
    """Merged sample variance matches raw data in either merge direction."""
    rng = np.random.default_rng(150)
    groups = [
        rng.normal(loc=0, scale=1, size=(sizes[0], 3)),
        rng.normal(loc=5, scale=2, size=(sizes[1], 3)),
    ]
    cluster_assignments = {
        0: list(range(sizes[0])),
        1: list(range(sizes[0], sum(sizes))),
    }
    cluster_means = pd.DataFrame([group.mean(axis=0) for group in groups])
    cluster_variances = pd.DataFrame([
        group.var(axis=0, ddof=1) for group in groups
    ])

    merging.merge_two_clusters(
        cluster_assignments,
        label_source=label_source,
        label_dest=label_dest,
        cluster_means=cluster_means,
        cluster_variances=cluster_variances,
    )

    combined = np.concatenate(groups)
    expected_variances = pd.DataFrame(
        [combined.var(axis=0, ddof=1)], index=[label_dest]
    )
    assert_frame_equal(cluster_variances, expected_variances)


def test_cdist_normalized():
    XA = np.array([[1], [3]])
    XB = np.array([[1], [6], [7]])
    expected_similarity = np.array(
        [[1, 1/6, 0],
         [4/6, 3/6, 2/6]]
    )

    obtained_similarity = merging.cdist_normalized(XA, XB)
    np.testing.assert_allclose(expected_similarity, obtained_similarity)


def test_find_most_similar(clusters):

    cluster_means, _, _, _ = clusters
    group_rows = ['32', 4]
    group_cols = ['11', 2, '32', 4]

    similarity_df = merging.calculate_similarity(cluster_means,
                                       group_rows=group_rows,
                                       group_cols=group_cols)

    source_label, dest_label, max_similarity = merging.find_most_similar(similarity_df)

    assert (source_label, dest_label) == ('32', '11')
    assert max_similarity == similarity_df.loc['32']['11']


def test_merge_small_clusters(clusters):

    cluster_means, _, _, cluster_assignments = clusters

    expected_cluster_assignments = {
        '11': [0, 3, 5, 9, 4, 7, 8],
        2: [1, 2, 6]
    }

    merging.merge_small_clusters(cluster_means, cluster_assignments, min_size=3)

    assert set(cluster_assignments.keys()) == set(expected_cluster_assignments.keys())
    for k, v in cluster_assignments.items():
        assert np.array_equal(cluster_assignments[k], expected_cluster_assignments[k])


def test_calculate_similarity(clusters):

    cluster_means, _, _, _ = clusters

    group_rows = ['32', 4]
    group_cols = ['11', 2, '32', 4]
    obtained_similarity = merging.calculate_similarity(cluster_means,
                                       group_rows=group_rows,
                                       group_cols=group_cols)

    expected_similarity = pd.DataFrame(
        [[0.917663, -0.866025, np.nan, 0.5],
         [0.802955, -0.866025, 0.5, np.nan]],
        index=group_rows,columns=group_cols)

    assert obtained_similarity.index.equals(obtained_similarity.index)
    assert obtained_similarity.columns.equals(obtained_similarity.columns)
    assert np.allclose(obtained_similarity.to_numpy(), expected_similarity.to_numpy(), equal_nan=True)


def test_get_k_nearest_clusters(clusters):

    cluster_means, _, _, _ = clusters

    expected_nns = [('11', 4), ('11', '32'), (2, '32'), (2, 4), ('32', 4)]

    knns = merging.get_k_nearest_clusters(cluster_means, k=2)

    assert len(expected_nns) == len(knns)
    for n in expected_nns:
        assert n in knns


def test_get_k_nearest_clusters_subset(clusters):

    cluster_means, _, _, _ = clusters

    expected_nns = [('11', 4), ('11', '32'), (2, '32'), (2, 4)]

    knns = merging.get_k_nearest_clusters(cluster_means, cluster_labels={'11', 2}, k=2)

    assert len(expected_nns) == len(knns)
    for n in expected_nns:
        assert n in knns


def test_get_cluster_assignments(adata, clusters):

    _, _, _, cluster_assignments = clusters
    obtained_cluster_assignments = merging.get_cluster_assignments(
        adata,
        cluster_label_obs="cluster_label")

    assert set(cluster_assignments.keys()) == set(obtained_cluster_assignments.keys())
    for k, v in cluster_assignments.items():
        assert set(cluster_assignments[k]) == set(obtained_cluster_assignments[k])


@pytest.fixture
def tasic_data_for_merge():

    normalized_path = os.path.join(DATA_DIR, "tasic_normed_select.h5ad")
    normalized_adata = sc.read_h5ad(normalized_path)

    reduced_dim_path = os.path.join(DATA_DIR, "tasic_projected.h5ad")
    reduced_dim_adata = sc.read_h5ad(reduced_dim_path)

    thresholds = {
        'q1_thresh': 0.5,
        'q2_thresh': None,
        'cluster_size_thresh': 6,
        'qdiff_thresh': 0.7,
        'padj_thresh': 0.05,
        'lfc_thresh': 1.0,
        'score_thresh': 40,
        'low_thresh': 1,
        'min_genes': 5
    }
    return normalized_adata, reduced_dim_adata, thresholds


@pytest.mark.parametrize("marker_kwargs, expected_markers", [
    ({}, set()),
    ({"n_markers": None}, None),
    ({"n_markers": 0}, set()),
    ({"return_markers_df": True}, pd.DataFrame()),
    ({"return_markers_df": True, "n_markers": None}, pd.DataFrame()),
])
def test_merge_clusters_single_cluster(adata, marker_kwargs, expected_markers):
    cluster_assignments = {0: list(range(adata.n_obs))}

    result = tc.merge_clusters(
        adata_norm=adata,
        adata_reduced=adata,
        cluster_assignments=cluster_assignments,
        cluster_by_obs=np.zeros(adata.n_obs),
        **marker_kwargs
    )

    assert isinstance(result, tuple)
    assignments, markers = result
    assert assignments == cluster_assignments
    assert assignments is not cluster_assignments
    assignments.pop(0)
    assert cluster_assignments == {0: list(range(adata.n_obs))}
    if isinstance(expected_markers, pd.DataFrame):
        assert_frame_equal(markers, expected_markers)
    else:
        assert markers == expected_markers


def test_merge_clusters_de_chisq(tasic_data_for_merge):

    tasic_norm_adata, tasic_reduced_dim_adata, thresholds = tasic_data_for_merge

    cluster_assignments_before_merging = merging.get_cluster_assignments(
        tasic_norm_adata,
        cluster_label_obs="cluster_label_before_merging")

    expected_cluster_assignments_after_merging = merging.get_cluster_assignments(
        tasic_norm_adata,
        cluster_label_obs="cluster_label_after_merging_chisq")

    cluster_assignments_after_merging, markers = tc.merge_clusters(
        adata_norm=tasic_norm_adata,
        adata_reduced=tasic_reduced_dim_adata,
        cluster_assignments=cluster_assignments_before_merging,
        cluster_by_obs=tasic_norm_adata.obs['cluster_label_before_merging'].values,
        thresholds=thresholds,
        de_method='chisq',
    )

    assert set(cluster_assignments_after_merging.keys()) == set(expected_cluster_assignments_after_merging.keys())
    for k, v in cluster_assignments_after_merging.items():
        assert set(v) == set(expected_cluster_assignments_after_merging[k])
    assert markers
    assert markers <= set(tasic_norm_adata.var_names)


def test_merge_clusters_de_ebayes(tasic_data_for_merge):

    tasic_norm_adata, tasic_reduced_dim_adata, thresholds = tasic_data_for_merge

    cluster_assignments_before_merging = merging.get_cluster_assignments(
        tasic_norm_adata,
        cluster_label_obs="cluster_label_before_merging")

    expected_cluster_assignments_after_merging = merging.get_cluster_assignments(
        tasic_norm_adata,
        cluster_label_obs="cluster_label_after_merging_ebayes")

    cluster_assignments_after_merging, markers = tc.merge_clusters(
        adata_norm=tasic_norm_adata,
        adata_reduced=tasic_reduced_dim_adata,
        cluster_assignments=cluster_assignments_before_merging,
        cluster_by_obs=tasic_norm_adata.obs['cluster_label_before_merging'].values,
        thresholds=thresholds,
        de_method='ebayes',
    )

    assert set(cluster_assignments_after_merging.keys()) == set(expected_cluster_assignments_after_merging.keys())
    for k, v in cluster_assignments_after_merging.items():
        assert set(v) == set(expected_cluster_assignments_after_merging[k])
    assert markers
    assert markers <= set(tasic_norm_adata.var_names)


def test_order_pairs():

    expected = [(2, 3), (4, 5)]
    obtained = merging.order_pairs([(2, 3), (5, 4)])
    assert obtained == expected


def test_merge_clusters_by_de():

    cluster_assignments = {
        11: [0, 3, 5, 9, 4],
        4: [1, 2, 8]
    }
    expected_cluster_assignments_after_merging = {4: [1, 2, 8, 0, 3, 5, 9, 4]}

    thresholds = {
        'q1_thresh': 0.3,
        'q2_thresh': None,
        'cluster_size_thresh': 1,
        'qdiff_thresh': 0.1,
        'padj_thresh': 0.5,
        'lfc_thresh': .4,
        'score_thresh': 40,
        'low_thresh': 1,
        'min_genes': 5
    }

    cluster_means = pd.DataFrame(
        np.array([[3.8, 1.5, 4.],
                  [3., 2., 3.]]),
        index=[11, 4],
        columns=['gene_a', 'gene_b', 'gene_c'])
    
    cluster_variances = pd.DataFrame(
        np.array([[10, 10, 10,],
                  [10, 10, 10]]),
        index=[11, 4],
        columns=['gene_a', 'gene_b', 'gene_c'])

    present_cluster_means = pd.DataFrame(
        np.array([[.1, .67, .51],
                  [.0, .6, .4]]),
        index=[11, 4],
        columns=['gene_a', 'gene_b', 'gene_c'])

    cluster_means_reduced = cluster_means.copy()

    merging.merge_clusters_by_de(cluster_assignments,
                                 cluster_means,
                                 cluster_variances,
                                 present_cluster_means,
                                 cluster_means_reduced,
                                 thresholds,
                                 k=1
                                 )
    assert set(cluster_assignments.keys()) == set(expected_cluster_assignments_after_merging.keys())
    for k, v in cluster_assignments.items():
        assert set(cluster_assignments[k]) == set(expected_cluster_assignments_after_merging[k])



@pytest.mark.parametrize("de_method", ["ebayes", "chisq"])
@pytest.mark.parametrize("up_in", [0, 1])
def test_merge_clusters_by_de_score_cap(de_method, up_in):
    """One very strong gene, up in the first or in the second cluster of the pair (the up- and the down-regulated
    score), for both DE methods: without a cap its score passes score_thresh, so the pair stays apart; with the cap of
    scrattch.hicat (each gene adds at most 20) one gene scores 20 < 150, and the pair merges"""
    def run(cap):
        cluster_assignments = {0: list(range(0, 500)), 1: list(range(500, 1000))}
        genes = ['gene_a', 'gene_b', 'gene_c']
        strong, weak = [8., 2., 2.], [0., 2., 2.]
        present_on, present_off = [1., 1., 1.], [0., 1., 1.]
        rows = (strong, weak) if up_in == 0 else (weak, strong)
        present = (present_on, present_off) if up_in == 0 else (present_off, present_on)
        variances = ([0.1, 1., 1.], [0.01, 1., 1.]) if up_in == 0 else ([0.01, 1., 1.], [0.1, 1., 1.])
        cluster_means = pd.DataFrame(np.array(rows), index=[0, 1], columns=genes)
        cluster_variances = pd.DataFrame(np.array(variances), index=[0, 1], columns=genes)
        present_cluster_means = pd.DataFrame(np.array(present), index=[0, 1], columns=genes)
        # min_genes 1: the pair of one DE gene is tested by its score only (hicat's test_merge also merges a pair of
        # fewer than min_genes DE genes)
        thresholds = {'q1_thresh': 0.5, 'q2_thresh': None, 'cluster_size_thresh': 4, 'qdiff_thresh': 0.7,
                      'padj_thresh': 0.01, 'lfc_thresh': 1.0, 'score_thresh': 150, 'low_thresh': 1, 'min_genes': 1}
        merging.merge_clusters_by_de(cluster_assignments, cluster_means, cluster_variances, present_cluster_means,
                                     cluster_means.copy(), thresholds, k=1, de_method=de_method, de_score_cap=cap)
        return len(cluster_assignments)

    assert run(None) == 2
    assert run(20) == 1


def _merge_with_stand_in_scores(sizes, rd, tag, score_of, k):
    """merge_clusters_by_de on clusters of the given sizes, positions in the reduced space (rd) and one tag gene, with
    a stand-in for the DE test: score_of(tag a, tag b, size a, size b) gives (score, number of DE genes)"""
    from unittest.mock import patch
    start, assignments = 0, {}
    for label, n in enumerate(sizes):
        assignments[label] = list(range(start, start + n))
        start += n
    labels = list(assignments)
    means = pd.DataFrame({'tag': tag}, index=labels, dtype=float)
    variances = pd.DataFrame({'tag': [1.0] * len(labels)}, index=labels)
    present = pd.DataFrame({'tag': [1.0] * len(labels)}, index=labels)
    rd = pd.DataFrame(rd, index=labels, dtype=float)

    def stand_in(pairs, cl_means, cl_vars, cl_present, cl_size, thresholds, de_score_cap=None):
        out = {}
        for a, b in pairs:
            score, num = score_of(cl_means.at[a, 'tag'], cl_means.at[b, 'tag'], cl_size[a], cl_size[b])
            out[(a, b)] = {'score': score, 'num': num}
        return pd.DataFrame(out).T

    thresholds = {'q1_thresh': 0.5, 'q2_thresh': None, 'cluster_size_thresh': 4, 'qdiff_thresh': 0.7,
                  'padj_thresh': 0.01, 'lfc_thresh': 1.0, 'score_thresh': 150, 'low_thresh': 1, 'min_genes': 5}
    with patch.object(tc, 'de_pairs_ebayes', stand_in):
        merging.merge_clusters_by_de(assignments, means, variances, present, rd, thresholds, k=k)
    return sorted(sorted(v) for v in assignments.values())


def test_merge_clusters_by_de_merges_one_pair_a_round_as_hicat():
    """Three clusters of type A (sizes 10, 10, 4) and one of type B (4) on a line in the reduced space. A pair of one
    type scores 10; a pair of two types scores 12 for each nucleus of the pair, so the two small clusters (A3, B1)
    score 96, under the threshold 150 but not under half of it, and B1 against the merged A clusters scores 336.
    scrattch.hicat's merge_cl merges the lowest pair of each round, and another pair only below half the threshold
    (`if(i == 1 | sc[i] < de.param$de.score.th /2 & length(intersect(p, merged))==0)`): A1 and A2, then A3 with
    them, and B1 stays apart. Merging every pair under the threshold in one round joins A3 and B1."""
    def score_of(tag_a, tag_b, n_a, n_b):
        return (10.0, 10) if abs(tag_a - tag_b) < 0.5 else (12.0 * (n_a + n_b), 10)

    got = _merge_with_stand_in_scores([10, 10, 4, 4], {'x': [0, 1, 2, 3], 'y': [0, 0, 0, 0]}, [0, 0, 0, 1],
                                      score_of, k=2)
    assert got == [list(range(24)), list(range(24, 28))]


def test_merge_clusters_by_de_min_genes_is_strict_as_hicat():
    """hicat's test_merge merges a pair whose score passes when it has fewer than min_genes DE genes
    (`to.merge=de.pair$num < de.param$min.genes`): a pair of exactly min_genes (5) genes with a passing score stays
    apart, and a pair of 4 merges"""
    def run(num):
        def score_of(tag_a, tag_b, n_a, n_b):
            if tag_a < 5 and tag_b < 5:
                return 10.0, 10
            if tag_a >= 5 and tag_b >= 5:
                return 200.0, num
            return 500.0, 30
        return len(_merge_with_stand_in_scores([10, 10, 10, 10], {'x': [0, 1, 10, 11], 'y': [0, 0, 0, 0]},
                                               [0, 0, 10, 10.5], score_of, k=1))

    assert run(5) == 3
    assert run(4) == 2


def test_update_similarity_of_the_merged_clusters_equals_a_full_computation():
    """The similarity computed again only for the merged clusters equals the similarity of all clusters, after a
    merge of two clusters (correlation, more than 2 dimensions)"""
    rng = np.random.default_rng(0)
    rd = pd.DataFrame(rng.normal(size=(6, 5)), index=list(range(6)))
    sim = merging.update_similarity(rd, None, [])
    assignments = {i: list(range(10 * i, 10 * i + 10)) for i in range(6)}
    merging.merge_cluster_means_vars(assignments, 4, 1, rd, None)
    assignments[1].extend(assignments.pop(4))
    got = merging.update_similarity(rd, sim, [1])
    full = merging.calculate_similarity(rd, group_rows=list(rd.index), group_cols=list(rd.index))
    assert_frame_equal(got, full)
    assert merging.nearest_pairs_by_similarity(got, 2) == merging.nearest_pairs_by_similarity(full, 2)
