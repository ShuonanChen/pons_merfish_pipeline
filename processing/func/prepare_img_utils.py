import matplotlib.pyplot as plt
import numpy as np
import glob
import pandas as pd
import scanpy as sc
import xml.etree.ElementTree as ET
import skimage.transform
import tifffile
import logging

    

# Setup logging
logging.basicConfig(level=logging.INFO)

def scaling_factor(pix_tran_filepath):
    """
    Calculate scaling factors and shift from the transformation matrix file.

    Parameters:
    pix_tran_filepath (str): Path to the transformation matrix file.

    Returns:
    tuple: Scaling factors and shift.
    """
    try:
        transformation_mtx = pd.read_csv(pix_tran_filepath, header=None, sep=' ')
        sc_factor_1 = transformation_mtx.iloc[0, 0]
        sc_factor_2 = transformation_mtx.iloc[1, 1]
        shift = np.array(transformation_mtx.iloc[:2, -1])
        return sc_factor_1, sc_factor_2, shift
    except Exception as e:
        logging.error(f"Error reading transformation matrix file: {e}")
        raise


def run_denovo_cluster(countmatrix_path):
    """
    Perform de novo clustering on the count matrix.

    Parameters:
    countmatrix_path (str): Path to the count matrix file.

    Returns:
    AnnData: Annotated data matrix.
    """
    try:
        count_mtx = pd.read_csv(countmatrix_path, index_col=0).sort_index()
        adata_mer = sc.AnnData(count_mtx)
        sc.pp.normalize_total(adata_mer)
        sc.pp.log1p(adata_mer)
        sc.tl.pca(adata_mer)
        sc.pp.neighbors(adata_mer, use_rep='X_pca')
        sc.tl.leiden(adata_mer,flavor="igraph", n_iterations=2)
        sc.pl.pca(adata_mer, color=["leiden"])
        return adata_mer
    except Exception as e:
        logging.error(f"Error running de novo clustering: {e}")
        raise


def save_filt_neuron_df(cell_pos_orig, adata_mer, outputfile, verbose=True):
    """
    Save filtered neuron data to a CSV file.

    Parameters:
    cell_pos_orig (DataFrame): Original cell positions.
    adata_mer (AnnData): Annotated data matrix.
    outputfile (str): Path to the output file.
    verbose (bool): Whether to print a message upon completion.
    """
    try:
        neuron_id_series = pd.Series(cell_pos_orig.index, index=cell_pos_orig.index, name='neuron_id')
        cell_pos_orig.columns = ['x', 'y']
        slice_series = pd.Series(np.ones(cell_pos_orig.shape[0]).astype(int), name='slice', index=cell_pos_orig.index)
        subclass_series = pd.Series(np.array(adata_mer.obs['leiden']).astype(int), name='subclass', index=cell_pos_orig.index)
        clustid_series = pd.Series(np.array(adata_mer.obs['leiden']).astype(int), name='clustid', index=cell_pos_orig.index)
        filt_neuron = pd.concat([neuron_id_series, cell_pos_orig, slice_series, subclass_series, clustid_series], axis=1)
        filt_neuron.to_csv(outputfile, index=False)
        if verbose:
            logging.info("Filtered neuron data saved.")
    except Exception as e:
        logging.error(f"Error saving filtered neuron data: {e}")
        raise

    
def get_coldict(adata_mer, cmapname='gist_ncar'):
    """
    Generate a color dictionary for clusters.

    Parameters:
    adata_mer (AnnData): Annotated data matrix.
    cmapname (str): Name of the colormap.

    Returns:
    dict: Color dictionary for clusters.
    """
    cmap = plt.get_cmap(cmapname)
    colors = cmap(np.linspace(0, 1, len(np.unique(np.array(adata_mer.obs['leiden']).astype(int)))))
    col_dict = dict(zip(np.unique(np.array(adata_mer.obs['leiden']).astype(int)), colors))
    return col_dict


def get_tifimage(thisdircheck, globalscaling=32):
    """
    Load and rescale a TIFF image.

    Parameters:
    thisdircheck (str): Directory path to the image.
    globalscaling (int): Scaling factor.

    Returns:
    ndarray: Rescaled image.
    """
    try:
        image_dir = glob.glob(thisdircheck + 'images/*.tif')
        tif_img = skimage.transform.rescale(tifffile.imread(image_dir[0], maxworkers=-1), 1/globalscaling, order=0)
        return tif_img
    except IndexError:
        logging.error(f"No TIFF images found in directory: {thisdircheck}")
        raise
    except Exception as e:
        logging.error(f"Error loading TIFF image: {e}")
        raise



def plot_jpg(cell_pos, adata_mer, tif_img, savefilename=None, save_bool=False, dpi=100, verbose=True, delta = 500):
    """
    Plot and optionally save a JPG image of cell positions overlaid on a TIFF image.

    Parameters:
    cell_pos (DataFrame): Cell positions.
    adata_mer (AnnData): Annotated data matrix.
    tif_img (ndarray): TIFF image.
    savefilename (str): Path to save the image.
    save_bool (bool): Whether to save the image.
    dpi (int): Dots per inch for the image.
    verbose (bool): Whether to print a message upon completion.
    """
    try:
        height, width = tif_img.shape
        fig = plt.figure(figsize=((width+2*delta)/dpi, (height+2*delta)/dpi), dpi=dpi, facecolor='black')
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        col_dict = get_coldict(adata_mer, cmapname='hsv')
        plt.imshow(tif_img, cmap='gray', vmax=tif_img.max() * .1)
        for k, col in col_dict.items():
            ids = np.where(np.array(adata_mer.obs['leiden']).astype(int) == k)[0]
            X_k = cell_pos.iloc[ids].values
            ax.scatter(X_k[:, 0], X_k[:, 1], color=col, s=1, alpha=.3, label=k)
        ax.set_xlim(-delta, width+delta)
        ax.set_ylim(height+delta, -delta)
        ax.set_aspect('equal')
        if save_bool:
            assert savefilename is not None, "Provide savefilename if you want to save the image"
            plt.savefig(savefilename, bbox_inches='tight', pad_inches=0)
            plt.close()
            if verbose:
                logging.info('Saved .jpg file')
        else:
            if verbose:
                logging.info('Image is not saved.')
    except Exception as e:
        logging.error(f"Error plotting JPG image: {e}")
        raise

        

def create_image_xml(name_of_slice, width, height, save_destination, verbose=True, delta = 500):
    """
    Create an XML file for the image.

    Parameters:
    name_of_slice (str): Name of the slice.
    width (int): Width of the image.
    height (int): Height of the image.
    save_destination (str): Directory to save the XML file.
    verbose (bool): Whether to print a message upon completion.
    """
    try:
        tree = ET.parse(save_destination + '/image_xml/template.xml')
        root = tree.getroot()
        root.attrib['name'] = f'cell_img_{name_of_slice}.jpg'
        root[0].attrib['filename'] = f'cell_img_{name_of_slice}.jpg'
        root[0].attrib['width'] = f'{int(width+2*delta)}'
        root[0].attrib['height'] = f'{int(height+2*delta)}'
        tree.write(save_destination + f'/image_xml/image_{name_of_slice}.xml')
        if verbose:
            logging.info('XML file created.')
    except FileNotFoundError:
        logging.error(f"Template XML file not found in: {save_destination}/image_xml/template.xml")
        raise
    except ET.ParseError:
        logging.error(f"Error parsing the template XML file.")
        raise
    except Exception as e:
        logging.error(f"Error creating XML file: {e}")
        raise