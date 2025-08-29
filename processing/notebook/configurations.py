import os
import sys

# Add the project root to the path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


# set where your stuff is, 
IMAGE_DIR = '/Users/shuonan.chen/Documents/project/merfish_register/go_register_w_QuickNiii/2025more_brain/'
