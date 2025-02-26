# pons_merfish_pipeline
MERFISH registration and analysis



1. create config.toml file 
2. create images with neurons location (`00_preparedat.ipynba`)
    ```
    rsync -a hpc-login:/allen/aind/scratch/shuonan.chen/code/pons_merfish_pipeline/processing/filt_neurons ~/Documents/projecty/merfish_register/go_register_w_QuickNiii/2025more_brain/
    ```

    ```
    rsync -a hpc-login:/allen/aind/scratch/shuonan.chen/code/pons_merfish_pipeline/processing/image_xml ~/Documents/projecty/merfish_register/go_register_w_QuickNiii/2025more_brain/
    ```

3. quicknii -- needs to save json and xml
4. visualign 
5. inverse the learned transformation (`01_apply_invert.ipynb`)
6. visualization (`02_overlay_all.ipynb`)