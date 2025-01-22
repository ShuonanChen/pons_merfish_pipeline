import matplotlib.pyplot as plt
import numpy as np
import tqdm, glob
import pandas as pd
import scanpy as sc
import xml.etree.ElementTree as ET

    
def scaling_factor(main_dir):    
    micron_to_mosaic_pixel_transform_file = main_dir + 'MouseEg_Eg2' + '/' + 'analyzed_data/*/region_1/' + '/images/micron_to_mosaic_pixel_transform.csv'
    micron_to_mosaic_pixel_transform_file = glob.glob(micron_to_mosaic_pixel_transform_file)[0]
    transformation_mtx = pd.read_csv(micron_to_mosaic_pixel_transform_file, header=None, sep=' ')
    sc_factor_1 = transformation_mtx.iloc[0,0]
    sc_factor_2 = transformation_mtx.iloc[1,1]
    return(sc_factor_1,sc_factor_2)

def run_denovo_cluster(countmatrix_path):
    count_mtx = pd.read_csv(countmatrix_path, index_col=0).sort_index()
    adata_mer = sc.AnnData(count_mtx)
    sc.pp.normalize_total(adata_mer)
    sc.pp.log1p(adata_mer)
    sc.tl.pca(adata_mer, n_comps=2)
    sc.pp.neighbors(adata_mer, use_rep='X_pca')#, n_neighbors=4)
    sc.tl.leiden(adata_mer, resolution=.1, n_iterations=2)
    sc.pl.pca(adata_mer,  color=["leiden"])
    return(adata_mer)

def save_filt_neuron_df(cell_pos_orig, adata_mer, outputfile,verbose=True):
    neuron_id_series = pd.Series(cell_pos_orig.index, index=cell_pos_orig.index, name = 'neuron_id')
    cell_pos_orig.columns = ['x','y'] 
    slice_seires = pd.Series(np.ones(cell_pos_orig.shape[0]).astype(int),name = 'slice', index=cell_pos_orig.index)
    subclass_series = pd.Series(np.array(adata_mer.obs['leiden']).astype(int),name = 'subclass', index=cell_pos_orig.index)
    clustid_series = pd.Series(np.array(adata_mer.obs['leiden']).astype(int),name = 'clustid', index=cell_pos_orig.index)
    filt_neuron = pd.concat([neuron_id_series,cell_pos_orig, slice_seires, subclass_series,clustid_series], axis=1)
    filt_neuron.to_csv(outputfile, index=False)
    if verbose: print("done")
        
        
def get_coldict(adata_mer, cmapname='hsv'):
    cmap = plt.get_cmap(cmapname)
    colors = cmap(np.linspace(0, 1, len(np.unique(np.array(adata_mer.obs['leiden']).astype(int)))))
    col_dict = dict(zip(np.unique(np.array(adata_mer.obs['leiden']).astype(int)), colors))
    return(col_dict)


def plot_jpg(cell_pos, adata_mer, savefilename=None, save_bool=False, dpi = 100, verbose = True):
    width, height = np.max(cell_pos.values, 0)
    fig = plt.figure(figsize=(width/dpi, height/dpi), dpi=dpi, facecolor='black')
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    col_dict = get_coldict(adata_mer, cmapname='hsv')
    for k,col in col_dict.items():
        ids = np.where(np.array(adata_mer.obs['leiden']).astype(int)==k)[0]
        X_k = cell_pos.iloc[ids].values
        ax.scatter(X_k[:,0],X_k[:,1], color = col,s=1, alpha =1, label = k)
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)    
    ax.set_aspect('equal')
    if save_bool:
        assert savefilename is not None, "provide savefilename if you want to save the image"
        plt.savefig(savefilename, bbox_inches='tight', pad_inches=0)
        plt.close()
        if verbose: print('saved .jpg file')
    else:
        if verbose: print('image is not saved.')
        


def create_image_xml(name_of_slice, width, height, save_destination, verbose= True):    
    tree = ET.parse(save_destination+'/image_xml/template.xml')
    root = tree.getroot()
    root[0].attrib['filename'] = f'cell_img_{name_of_slice}.jpg'
    root[0].attrib['width'] = f'{int(width)}'
    root[0].attrib['height'] = f'{int(height)}'
    tree.write(save_destination+f'/image_xml/image_{name_of_slice}.xml')
    if verbose: print('done')        