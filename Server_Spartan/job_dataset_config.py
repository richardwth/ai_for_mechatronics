import sys
sys.path.insert(0, '/home/richard_wth/')
from GeneralTools.misc_fun import FLAGS
FLAGS.DEFAULT_IN = '/data/cephfs/punim0811/dataset/'  # add data folder
FLAGS.DEFAULT_OUT = '/home/richard_wth/Results/'  # add folder to save the final results
# FLAGS.SPECTRAL_NORM_MODE = 'sn_paper'
# FLAGS.WEIGHT_INITIALIZER = 'sn_paper'

# import packages
import numpy as np
from GeneralTools.layer_fun import .

# configurate the model hyper-parameters
learning_rate = 1e-3
architecture = {}

# instantiate a model 
mdl = Classifier(architecture, learning_rate=learning_rate)

# train the model
mdl.training()

# evaluate the model and print the reults to file
mdl.test()

# setup another set of hyper-parameters and re-run the model
# ...

# indicate the code run successfully
print('Chunk of code finished.')
