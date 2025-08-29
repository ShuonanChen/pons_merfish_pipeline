import numpy as np
import matplotlib.pyplot as plt
from sklearn.cross_decomposition import CCA
from sklearn.preprocessing import StandardScaler
import trimesh
from typing import Any, Tuple, List
import plotting
import preprocessing 

def flip(a: np.ndarray, xm: float) -> np.ndarray:
    """Flip array 'a' relative to a midpoint 'xm'."""
    return 2*xm - a

# def get_hemi(S_mer: np.ndarray, mesh: Any) -> np.ndarray:
#     """
#     Adjust spatial coordinates so that points on one hemisphere are flipped.
#     Assumes the axis of interest is the last axis.
#     """
#     xm = np.min(mesh.vertices[:, -1]) + np.ptp(mesh.vertices[:, -1]) / 2
#     new_coords = S_mer.copy()
#     new_coords[:, -1] = np.where(new_coords[:, -1] > xm, flip(new_coords[:, -1], xm), new_coords[:, -1])
#     return new_coords
def run_CCA(X: np.ndarray, S: np.ndarray, n_components: int = 3, visualize: bool = False) -> Tuple[CCA, np.ndarray, np.ndarray]:
    """
    Run Canonical Correlation Analysis (CCA) using gene expression (X) and spatial coordinates (S).
    If visualize is True, plot the spatial layout.
    
    Parameters:
    -----------
    X : np.ndarray
        Gene expression matrix (n_samples, n_genes)
    S : np.ndarray
        Spatial coordinates matrix (n_samples, n_dimensions)
    """
    allmeshes = plotting.load_mesh()
    mesh_LC, mesh_CD, mesh_CV = allmeshes
        
    S = preprocessing.get_hemi(S, mesh_LC)
    
    scaler_X = StandardScaler().fit(X)
    X_scaled = scaler_X.transform(X)
    scaler_S = StandardScaler().fit(S)
    S_scaled = scaler_S.transform(S)
    
    if visualize:
        plt.figure(figsize=(5, 5))
        if S.shape[1] >= 3:
            plt.scatter(S[:, 2], S[:, 1], s=0.3, alpha=0.8)
        else:
            plt.scatter(S[:, 0], S[:, 1], s=0.3, alpha=0.8)
        plt.gca().set_aspect('equal')
        plot_mesh(ax=plt.gca(), allmeshes=allmeshes, direction='c')
    
    cca = CCA(n_components=n_components)
    cca.fit(X_scaled, S_scaled)
    X_c, S_c = cca.transform(X_scaled, S_scaled)

    return cca, X_c, S_c


def compute_canonical_correlations(X_c: np.ndarray, S_c: np.ndarray) -> np.ndarray:
    """
    Compute canonical correlations between each pair of canonical variates.
    """
    n_components = X_c.shape[1]
    correlations = np.array([
        np.corrcoef(X_c[:, i], S_c[:, i])[0, 1] for i in range(n_components)
    ])
    return correlations


def plot_result(S: np.ndarray, cca: CCA, X_c: np.ndarray, S_c: np.ndarray, scale_factor: float = 1, plot_mesh_flag: bool = True) -> None:

    """
    Plot the CCA results along with the meshes.
    
    Parameters:
    -----------
    S : np.ndarray
        Original spatial coordinates matrix
    cca : CCA
        Fitted CCA object
    X_c : np.ndarray
        Transformed gene expression data
    S_c : np.ndarray
        Transformed spatial coordinates
    scale_factor : float, optional
        Scale factor for the direction vectors, default=50
    plot_mesh_flag : bool, optional
        Whether to plot the mesh, default=True
    """
    canonical_corrs = compute_canonical_correlations(X_c, S_c)
    for k in range(X_c.shape[1]):
        if canonical_corrs[k] < 0.5:
            print(f"Canonical correlation for component {k+1} is low: {canonical_corrs[k]:.3f}. Skipping plot.")
            continue

        vector = cca.y_weights_[:, k]
        print(vector)
        vector_scaled = vector * (-1)**(np.max(vector)>0)*scale_factor

        allmeshes = plotting.load_mesh() if plot_mesh_flag else None
        plt.figure(figsize=(15, 5))
        
        # Plot with first coordinate system
        plt.subplot(1, 2, 1)
        sca = plt.scatter(S[:, 0], S[:, 1], c=X_c[:, k], cmap='Reds', s=10)
        plt.colorbar(sca, label=f'CCA {k+1} canonical components ')
        plt.gca().set_aspect('equal')
        if plot_mesh_flag:
            plotting.plot_mesh(ax=plt.gca(), allmeshes=allmeshes, direction='s')
        plt.title("%.3f" % ((-1)**(np.max(vector)>0)*canonical_corrs)[k], color='magenta')


        start_point = (np.min(S[:, 0]), np.max(S[:, 1]))
        end_point = (start_point[0] + vector_scaled[0], start_point[1] + vector_scaled[1])
        plt.annotate('', color='magenta',
                    xy=end_point,
                    xytext=start_point,
                    arrowprops=dict(arrowstyle="->", color="magenta", lw=2))
        
        # Plot with second coordinate system
        plt.subplot(1, 2, 2)
        if S.shape[1] >= 3:
            sca = plt.scatter(S[:, 2], S[:, 1], c=X_c[:, k], cmap='Reds', s=10)
            start_point = (np.min(S[:, 2]), np.max(S[:, 1]))
            arrow_dx = vector_scaled[2]
        else:
            sca = plt.scatter(S[:, 0], S[:, 1], c=X_c[:, k], cmap='Reds', s=10)
            start_point = (np.min(S[:, 0]), np.max(S[:, 1]))
            arrow_dx = vector_scaled[0]
        plt.colorbar(sca, label=f'CCA {k+1} canonical components ')
        plt.gca().set_aspect('equal')
        if plot_mesh_flag:
            plotting.plot_mesh(ax=plt.gca(), allmeshes=allmeshes, direction='c')
        
        
        end_point = (start_point[0] + arrow_dx, start_point[1] + vector_scaled[1])
        plt.annotate('', color='magenta',
                    xy=end_point,
                    xytext=start_point,
                    arrowprops=dict(arrowstyle="->", color="magenta", lw=2))
        
        plt.tight_layout()
        plt.show()


def main(X: np.ndarray, S: np.ndarray, plot_mesh_flag: bool = True) -> None:
    """
    Main function to run CCA analysis on gene expression and spatial data.
    
    Parameters:
    -----------
    X : np.ndarray
        Gene expression matrix (n_samples, n_genes)
    S : np.ndarray
        Spatial coordinates matrix (n_samples, n_dimensions)
    plot_mesh_flag : bool, optional
        Whether to plot the mesh overlays, default=True
    """
    cca, X_c, S_c = run_CCA(X, S, n_components=3, visualize=False)
    plot_result(S, cca, X_c, S_c, plot_mesh_flag=plot_mesh_flag)

if __name__ == "__main__":
    try:
        main(X, S)  # X and S should be defined before running the script
    except NameError:
        print("Please provide X (gene expression matrix) and S (spatial coordinates) before running main().")