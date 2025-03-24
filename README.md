# pons_merfish_pipeline
MERFISH registration and analysis

## registration:
### Input:  
- Raw images (tif files) of polyT or DAPI  
- Cells locations (N x 3) 
- Cells gene expression (N x G) 

### Tools needed: 
- Some clustering method (simple de novo clustering)
    - this is just to visualize your cells in different colors during the registration
- Filbuilder (may come with QuickNII)
- QuickNII. 2017 mouse version  
- Visualign  



### steps
1. create config.toml file. example:
    ```    
    # Root directory for the code
    code_root = "yourpath/pons_merfish_pipeline/processing/"
    
    # Root directory for the data
    data_root = "yourpath/merfish_images/"
    ```
2. create images with neurons location (`00_preparedata.ipynb`)
3. quicknii -- needs to save json and xml
4. visualign 
5. inverse the learned transformation (`01_apply_invert.ipynb`)
6. visualization (`02_overlay_all.ipynb`)
