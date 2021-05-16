#!/usr/bin/env python
# coding: utf-8

# # Loader
# Externally (Re)Load required packages and initialize path(s):

# In[6]:


# Load Packages/Modules
exec(open('./Helper_Scripts_and_Loading/Init_Dump.py').read())
# Load Hyper-parameter Grid
exec(open('./Helper_Scripts_and_Loading/CV_Grid.py').read())
# Load Helper Function(s)
exec(open('./Helper_Scripts_and_Loading/Helper_Functions.py').read())
# Architecture Builder
exec(open('./Helper_Scripts_and_Loading/Benchmarks_Model_Builder.py').read())
# Auxiliary Helper Function(s)
exec(open('./Helper_Scripts_and_Loading/MISC_HELPER_FUNCTIONS.py').read())
# Import time separately
import time
#os.environ['CUDA_VISIBLE_DEVICES'] = '0'


# Load Path Strings

# In[ ]:


results_path = "./outputs/models/"
results_tables_path = "./outputs/results/"
raw_data_path_folder = "./inputs/raw/"
data_path_folder = "./inputs/data/"


# ## Set Seeds

# In[3]:


random.seed(2021)
np.random.seed(2021)
tf.random.set_seed(2021)


# ---
# # Fin
# ---
