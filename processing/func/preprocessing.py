import numpy as np


def load_quickmesh():
    import trimesh
    mesh_LC = trimesh.load_mesh("/allen/aind/scratch/shuonan.chen/scripts/Pons_MERFISH/mesh/LC_ccf_v1_250102 2.obj")
    return(mesh_LC)

def flip(a, xm):
        return(2*xm-a)

def get_hemi(S_mer, mesh=None):
    '''assume the axis of interest are both on the last axis. '''
    if mesh is None:
        mesh = load_quickmesh()
    xm = np.min(mesh.vertices[:,-1]) + np.ptp(mesh.vertices[:,-1])/2 # this is the center line to indicate the hemisphere         
    new_coords = S_mer.copy()
    new_coords[:,-1] = np.where(new_coords[:,-1] > xm, flip(new_coords[:,-1],xm), new_coords[:,-1])    
    return(new_coords)