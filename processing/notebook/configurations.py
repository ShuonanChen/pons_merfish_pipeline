import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


# set where your stuff is, 
# under this directory it should look something like this:
# IMAGE_DIR/
# ├── filt_neurons/
# ├── image_xml/
# ├── results/
# ├── visualign_rez/


IMAGE_DIR = '/Users/shuonan.chen/Documents/project/merfish_register/go_register_w_QuickNiii/2025more_brain/'

# your downsampling scales.
GLOBAL_SCALING = 32