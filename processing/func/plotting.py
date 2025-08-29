from typing import Any, Tuple, List


def load_mesh():
    """
    Load three mesh objects for LC, CD and CV.
    """
    import trimesh
    mesh_LC = trimesh.load_mesh("/allen/aind/scratch/shuonan.chen/scripts/Pons_MERFISH/mesh/LC_ccf_v1_250102 2.obj")
    mesh_CD = trimesh.load_mesh("/allen/aind/scratch/shuonan.chen/scripts/Pons_MERFISH/mesh/subCD_ccf_v1_250102 2.obj")
    mesh_CV = trimesh.load_mesh("/allen/aind/scratch/shuonan.chen/scripts/Pons_MERFISH/mesh/subCV_ccf_v1_250102 2.obj")
    allmeshes = [mesh_LC,mesh_CD,mesh_CV]
    return allmeshes


def format_ticklabels(ticks, scale=25):
        return [f'{int(t * scale)}' if i % 2 == 0 else '' for i, t in enumerate(ticks)]


def draw_scale_bar(ax, length_px, px_to_mm=25*1e-3, loc=(0.1, 0.05), linewidth=4):
    """Draws a horizontal scale bar of length_px on ax."""
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    x0 = xlim[0] + loc[0] * (xlim[1] - xlim[0])
    y0 = ylim[0] + loc[1] * (ylim[1] - ylim[0])
    ax.plot([x0, x0 + length_px],[y0, y0],color='black',linewidth=linewidth)
    ax.text(x0 + length_px/2,  y0 - 3,#0.03*(ylim[1]-ylim[0]),
            f'{length_px * px_to_mm:.1f} mm',ha='center', va='bottom')

    
def plot_mesh(ax: Any, direction: str = 'c') -> None:
    """
    Plot the three meshes on the given axis.
    parameter direction: select index to choose coordinate ('c' uses index 2, otherwise index 0)
    """
    mesh_LC, mesh_CD, mesh_CV = load_mesh()
    ax.set_aspect('equal')
    i = 2 if direction == 'c' else 0
    ax.triplot(mesh_LC.vertices.T[i], mesh_LC.vertices.T[1], mesh_LC.faces, alpha=0.1, label='LC')
    ax.triplot(mesh_CD.vertices.T[i], mesh_CD.vertices.T[1], mesh_CD.faces, alpha=0.1, label='CD')
    ax.triplot(mesh_CV.vertices.T[i], mesh_CV.vertices.T[1], mesh_CV.faces, alpha=0.1, label='CV')
    ax.invert_yaxis()
    
from sklearn.preprocessing import LabelEncoder

def plot_spatial(ax,
                 *,
                 coords=None,
                 colorvalues=None,
                 adata=None,
                 color=None,
                 basis='spatial',
                 dims=(2,1),
                 s=10,
                 cmap=None,
                 vmin=0, vmax=1,
                 alpha=0.8,
                 meshes=None,
                 mesh_direction=None,
                 xlabel='',
                 ylabel='',
                 scale_px=40,
                 scale_color='black',
                 scale_linewidth=4,
                 changeticks=False,
                 colorbartitle=None
                ):
    
    """
    ax : matplotlib Axes
    coords : (N,3) array
    colorvalues : length-N array for coloring -> example: imputed_values
    meshes : dict of name->(vertices, faces)
    dims : which two dims of coords to plot (x_dim, y_dim)

    If `adata` is given, plots using generic matplotlib (instead of scanpy).
    Otherwise, does a raw scatter of `coords` + `colorvalues`.
    In both cases, overlays meshes + draws a scale bar + fixes aspect.
    """

    xdim, ydim = dims
    scatter_kwargs = dict(s=s, alpha=alpha)
    if cmap is not None:
        scatter_kwargs['cmap'] = cmap
    else:
        pass

    if adata is not None:
        # Handle categorical color values
        if color is not None:
            # Convert categorical values to numeric
            le = LabelEncoder()
            colorvalues = le.fit_transform(adata.obs[color].values)

        ax.scatter(adata.obsm[basis][:, xdim],
                   adata.obsm[basis][:, ydim],
                   c=colorvalues,
                   **scatter_kwargs
                  )
    else:
        # When `adata` is not provided, fallback to using the coords
        sca = ax.scatter(coords[:, dims[0]],
                         coords[:, dims[1]],
                         c=colorvalues,
                         vmin=vmin,vmax=vmax,
                        **scatter_kwargs)
        cbar = ax.figure.colorbar(sca, ax=ax)
        if colorbartitle is not None:
            cbar.set_label(colorbartitle, loc='center')

    ax.set_aspect('equal')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    
    
    if changeticks:
        xt = ax.get_xticks(); yt = ax.get_yticks()
        ax.set_xticks(xt); ax.set_yticks(yt)
        ax.set_xticklabels(format_ticklabels(xt))
        ax.set_yticklabels(format_ticklabels(yt))

    # overlay meshes
    plot_mesh(ax, direction = 'c') if dims[0]==2 else plot_mesh(ax, direction = 's')

    # scale bar
    draw_scale_bar(ax,
                   length_px=scale_px,
                   linewidth=scale_linewidth)
    leg = ax.get_legend()
    if leg:
        leg.remove()
