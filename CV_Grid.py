# This file contains the hyper-parameter grids used to train the imprinted-tree nets.

#----------------------#
########################
# Hyperparameter Grids #
########################
#----------------------#


# Hyperparameter Grid (Readout)
#------------------------------#
if trial_run == True:

    
    # Training Parameters
    #----------------------#
    # Number of Jobs (Cores to use)
    n_jobs = 4
    # Number of Random CV Draws
    n_iter = 1
    n_iter_trees = 1#20
    # Number of CV Folds
    CV_folds = 2


    param_grid_Deep_Classifier = {'batch_size': [32],
                        'epochs': [200],
                        'learning_rate': [0.00001],
                        'height': [200],
                        'depth': [3],
                        'input_dim':[15],
                        'output_dim':[1]}

    
else:
    
    # Training Parameters
    #----------------------#
    # Number of Jobs (Cores to use)
    n_jobs = 70
    # Number of Random CV Draws
    n_iter = 50
    n_iter_trees = 50
    # Number of CV Folds
    CV_folds = 4
    

    param_grid_Deep_Classifier = {'batch_size': [16,32,64],
                        'epochs': [100, 150, 200, 250,500],
                        'learning_rate': [0.0001,0.0005,0.001, 0.005],
                        'height': [200,300,400,500,600],
                        'depth': [3,4,5],
                        'input_dim':[15],
                        'output_dim':[1]}
                        
                       

