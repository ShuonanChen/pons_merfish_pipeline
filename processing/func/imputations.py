# imputations 
import pandas as pd
import anndata
import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.stats import rankdata

def normalize_cols(M, ranked=True):
    """
    Normalize the columns of M.
    If ranked=True, replace each column with its ranks (using average tie handling).
    Then subtract the mean and divide by the standard deviation.
    Parameters:
      M: numpy array of shape (n_samples, n_features)
      ranked: bool, whether to perform ranking.
      
    Returns:
      The normalized matrix.
    """
    result = M.toarray().copy()
    if ranked:  # output shape: 
        result = np.apply_along_axis(rankdata, 0, result)
    means = np.mean(result, axis=0)
    stds = np.std(result, axis=0, ddof=0)
    stds[stds == 0] = 1e-10
    result = (result - means) / stds
    return result


def impute_mer_data(adata_sc, adata_mer, k=10, n_hvg=1000):
    """
    Impute merFISH data (adata_mer) using nearest neighbors from snRNAseq data (adata_sc).
      k: int (default=10)
          Number of nearest neighbors to use.
      n_hvg: int (default=1000)
          Number of highly variable genes to select from adata_sc.          
    Returns:
      An AnnData object with imputed expression for the union gene set.
    """
    adata_sc = adata_sc.copy()  # work on a copy to avoid modifying the original data
    if n_hvg is not None:
        import scanpy as sc
        sc.pp.highly_variable_genes(adata_sc, n_top_genes=n_hvg, flavor='seurat_v3')
        hvg_genes = adata_sc.var_names[adata_sc.var['highly_variable']]
    else:
        hvg_genes = adata_sc.var_names
    
    union_genes = np.union1d(hvg_genes, adata_mer.var_names)  # 1183    
    observed_genes = np.array(adata_mer.var_names)
    adata_sc_union = adata_sc[:, union_genes].copy()  # we will impute based on this guy, 1183 genes in total     
    common_genes = np.intersect1d(adata_mer.var_names, union_genes)
    
    X_sc = adata_sc_union[:, common_genes].X
    X_mer = adata_mer[:, common_genes].X     
    X_sc_union = adata_sc_union.X 
    
    #  scale/normalize these matrices further.)    
    X_sc_norm = normalize_cols(X_sc, ranked=True)
    X_mer_norm = normalize_cols(X_mer, ranked=True)
        
    nbrs = NearestNeighbors(n_neighbors=k, metric='euclidean').fit(X_sc_norm)
    distances, indices = nbrs.kneighbors(X_mer_norm) # distance: size (3227, K)
    
    epsilon = 1e-10
    weights = 1 / (distances + epsilon)
    weights = weights / weights.sum(axis=1, keepdims=True)  # normalize so that weights sum to 1
    
    imputed_expr = np.zeros((adata_mer.n_obs, len(union_genes)))
    for i in range(adata_mer.n_obs):
        neighbor_idx = indices[i]
        imputed_expr[i, :] = np.dot(weights[i], X_sc_union[neighbor_idx, :].todense())

    adata_mer_imputed = anndata.AnnData(
        X=w,
        obs=adata_mer.obs.copy(),
        var=pd.DataFrame(index=union_genes))
    
    return adata_mer_imputed

