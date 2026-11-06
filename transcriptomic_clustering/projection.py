from typing import Optional, Union, Sequence

import numpy as np
import scipy as scp
import pandas as pd
import scanpy as sc
import anndata as ad
import transcriptomic_clustering as tc

Mask = Union[Sequence[int], slice, np.ndarray]

def project(
        adata: ad.AnnData,
        principal_comps: pd.DataFrame,
        mean: Optional[pd.DataFrame]=None,
        chunk_size: Optional[int]=None,
        harmony_batch_key: Optional[str]=None,
        harmony_kwargs: Optional[dict]=None,
        info: Optional[dict]=None) -> np.ndarray:
    """
    Projects data into principal component space

    Parameters
    ----------
    adata:
        adata to project into principal component space
    principal_comps: 
        principal component Dataframe (rows=genes, columns=components)
    mean:
        mean used for zero centering (rows=genes, column=mean)
    harmony_batch_key:
        if given, an obs column of adata: the projection is corrected for it with Harmony
        (harmonypy; Korsunsky et al. 2019), so that each clustering step and the final merge
        work on batch-corrected principal components, as Siletti et al. 2023 ran Harmony at
        each level of their iterative clustering. See harmonize_projection.
        None (the default): no correction
    harmony_kwargs:
        keyword arguments of harmonypy.run_harmony (e.g. random_state, max_iter_harmony)
    info:
        if given, a dict that is updated with what the step did: n_pcs, and with a batch key
        batch_levels, harmony_ran and harmony_iterations

    Returns
    -------
    Adata object in principal component space
    """

    if not mean.index.equals(principal_comps.index):
        raise ValueError('mean and principal comps have different genes')
    _, vidx = adata._normalize_indices((slice(None), principal_comps.index)) # handle gene mask like anndata would
    pc_names = principal_comps.columns
    principal_comps = principal_comps.to_numpy()
    mean = mean.to_numpy().T

    n_obs = adata.n_obs
    n_vars = adata.n_vars
    n_comps = principal_comps.shape[1]
    n_genes = principal_comps.shape[0]

    issparse = False
    if adata.isbacked and hasattr(adata.X, "format_str") and adata.X.format_str == "csr":
        issparse = True
    
    # Estimate memory
    if not chunk_size:
        if not adata.is_view:  # .X on view will try to load entire X into memory
            itemsize = adata.X.dtype.itemsize
        else:
            itemsize = np.dtype(np.float64).itemsize
        process_memory = n_obs * n_vars * itemsize / (1024 ** 3)
        if issparse:
            process_memory *= 2

        output_memory = n_obs * n_comps * itemsize / (1024 ** 3)
        chunk_size = tc.memory.estimate_chunk_size(
            adata,
            process_memory=process_memory,
            output_memory=output_memory,
            percent_allowed=70,
            process_name='project',
        )

    # Transform
    if not adata.isbacked and chunk_size >= n_obs:
        X = adata.X
        if issparse:
            X = X.toarray()
        X = X[:, vidx]
        if mean is not None:
            X -= mean
        X_proj = X @ principal_comps

    else:
        X_proj = np.zeros((adata.n_obs, n_comps))
        for chunk, start, end in adata.chunked_X(chunk_size):
            if scp.sparse.issparse(chunk):
                chunk = chunk.toarray()
            chunk = chunk[:, vidx]
            if mean is not None:
                chunk -= mean
            X_proj[start:end,:] = chunk @ principal_comps

    projected = ad.AnnData(X_proj, obs=adata.obs, var=pd.DataFrame(index=pc_names))
    if info is not None:
        info['n_pcs'] = int(projected.n_vars)
    if harmony_batch_key is not None:
        harmonize_projection(projected, harmony_batch_key, harmony_kwargs, info)
    return projected


def harmonize_projection(
        projected: ad.AnnData,
        batch_key: str,
        harmony_kwargs: Optional[dict]=None,
        info: Optional[dict]=None) -> ad.AnnData:
    """
    Corrects projected principal components for a batch column with Harmony, in place

    A group with one batch level has nothing to correct. A group whose number of Harmony
    clusters (harmony_kwargs['nclust'], else harmonypy's default min(round(n_obs / 30), 100))
    is below 2 is not corrected either: harmonypy needs at least 2 clusters.

    Parameters
    ----------
    projected:
        adata in principal component space (cells x components) with batch_key in obs
    batch_key:
        the obs column of the batch
    harmony_kwargs:
        keyword arguments of harmonypy.run_harmony
    info:
        if given, updated with batch_levels, harmony_ran and harmony_iterations

    Returns
    -------
    projected, with X replaced by the corrected components when Harmony ran
    """
    harmony_kwargs = dict(harmony_kwargs or {})
    batch = projected.obs[batch_key].astype(str)
    levels = int(batch.nunique())
    nclust = harmony_kwargs.get('nclust') or int(min(round(projected.n_obs / 30.0), 100))
    ran, iterations = False, None
    if levels >= 2 and nclust >= 2:
        import harmonypy
        meta = pd.DataFrame({batch_key: batch.to_numpy()}, index=projected.obs_names)
        ho = harmonypy.run_harmony(np.asarray(projected.X, dtype=np.float64), meta, batch_key, **harmony_kwargs)
        z = np.asarray(ho.Z_corr)
        if z.shape != projected.shape:  # harmonypy before 2.0 returns components x cells
            z = z.T
        if z.shape != projected.shape:
            raise ValueError(f'Harmony returned shape {z.shape}, not {projected.shape}')
        projected.X = z
        ran, iterations = True, len(ho.kmeans_rounds)
    if info is not None:
        info.update(batch_levels=levels, harmony_ran=ran, harmony_iterations=iterations)
    return projected


def latent_project(adata: ad.AnnData,
                    latent_component: Optional[str]=None) -> ad.AnnData:
    """
    Extracts pre-existing data projection

    Parameters
    ----------
    adata:
        adata containing the projection.

    Returns
    -------
    Adata object in latent space
    """

    if not latent_component in adata.obsm.keys():
        raise ValueError('Latent space ' + latent_component + " does not exist in obsm")

    ## Extract latent space and define names
    latent_data = adata.obsm[latent_component]
    latent_names = ["latent-"+str(i) for i in range(latent_data.shape[1])]

    return ad.AnnData(X=latent_data, obs=adata.obs, var=pd.DataFrame(index=latent_names))