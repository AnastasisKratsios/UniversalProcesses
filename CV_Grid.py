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
    n_iter = 2
    n_iter_trees = 2
    # Number of CV Folds
    CV_folds = 2
    # Number of Boostrapped Confidence Intervals
    N_Boostraps_BCA = 50


    param_grid_Deep_Classifier = {'batch_size': [32],
                        'epochs': [50],
                        'learning_rate': [0.00001],
                        'height': [20],
                        'depth': [1],
                        'input_dim':[15],
                        'output_dim':[1]}
    
    # Random Forest Grid
    #--------------------#
    Rand_Forest_Grid = {'learning_rate': [0.01],
                        'max_depth': [10],
                        'min_samples_leaf': [3],
                        'n_estimators': [500]}
    
    # Kernel Ridge #
    #--------------#
    param_grid_kernel_Ridge={"alpha": np.linspace(1e0, 0.1, 10),
                             "gamma": np.logspace(-2, 2, 50),
                             "kernel": ["rbf", "laplacian", "polynomial", "cosine", "sigmoid"]}
    
    # Gaussian Process Regression #
    #-----------------------------#
    param_grid_GAUSSIAN = {'kernel':[kernels.RBF(),
                                     kernels.Matern(),
                                     kernels.RationalQuadratic(),
                                     kernels.WhiteKernel()],
                           'n_restarts_optimizer':[5,10,20,25],
                           'random_state':[2020]}
    
   
    
else:
    
    # Training Parameters
    #----------------------#
    # Number of Jobs (Cores to use)
    n_jobs = 40
    # Number of Random CV Draws
    n_iter = 10
    n_iter_trees = 10
    # Number of CV Folds
    CV_folds = 4
    # Number of Boostrapped Confidence Intervals
    N_Boostraps_BCA = 10**3
    

    param_grid_Deep_Classifier = {'batch_size': [8,16,32],
                        'epochs': [150,200,250,400,600,700],
                        'learning_rate': [0.00001,0.000001,0.000001],
                        'height': [50,100,200,250,400],
                        'depth': [2,3,4],
                        'input_dim':[15],
                        'output_dim':[1]}
    
    # Random Forest Grid
    #--------------------#
    Rand_Forest_Grid = {'learning_rate': [0.0005,0.0001,0.00005,0.00001],
                        'max_depth': [3,4,5,6, 7, 8,9, 10],
                        'min_samples_leaf': [5, 9, 17, 20,50],
                       'n_estimators': [1500]}
    
    # Kernel Ridge #
    #--------------#
    param_grid_kernel_Ridge={"alpha": [1e0, 0.1, 1e-2, 1e-3],
                             "gamma": np.logspace(-2, 2, 10**2),
                             "kernel": ["rbf", "laplacian", "polynomial", "cosine", "sigmoid"]}
    
    # Gaussian Process Regression #
    #-----------------------------#
    param_grid_GAUSSIAN = {'kernel':[kernels.RBF(),
                                     kernels.Matern(),
                                     kernels.RationalQuadratic(),
                                     kernels.WhiteKernel()],
                           'n_restarts_optimizer':[5,10,20,25],
                           'random_state':[2020]}

                        
                       

