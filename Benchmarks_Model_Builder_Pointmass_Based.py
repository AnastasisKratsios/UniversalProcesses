#!/usr/bin/env python
# coding: utf-8

# # Define Model(s)

# ---

# ## Elastic Net Regressor

# In[ ]:


#=====================#
# Elastic Net Version #
#=====================#
# Block warnings that spam when performing coordinate descent (by default) in 1-d.
import warnings
from sklearn.linear_model import ElasticNetCV
warnings.filterwarnings("ignore")
# Initialize Elastic Net Regularization Model
Lin_reg = ElasticNetCV(cv=5, random_state=0, alphas = np.linspace(0,(10**2),(10**2)),
                           l1_ratio=np.linspace(0,1,(10**2)))


# ## ffNN Builder

# In[1]:


def get_ffNN(height, depth, learning_rate, input_dim, output_dim):
    #----------------------------#
    # Maximally Interacting Layer #
    #-----------------------------#
    # Initialize Inputs
    input_layer = tf.keras.Input(shape=(input_dim,))
   
    
    #------------------#
    #   Core Layers    #
    #------------------#
    core_layers = fullyConnected_Dense(height)(input_layer)
    # Activation
    core_layers = tf.nn.swish(core_layers)
    # Train additional Depth?
    if depth>1:
        # Add additional deep layer(s)
        for depth_i in range(1,depth):
            core_layers = fullyConnected_Dense(height)(core_layers)
            # Activation
            core_layers = tf.nn.swish(core_layers)
    
    #------------------#
    #  Readout Layers  #
    #------------------# 
    # Affine (Readout) Layer (Dense Fully Connected)
    output_layers = fullyConnected_Dense(output_dim)(core_layers)  
    # Define Input/Output Relationship (Arch.)
    trainable_layers_model = tf.keras.Model(input_layer, output_layers)
    
    
    #----------------------------------#
    # Define Optimizer & Compile Archs.
    #----------------------------------#
    opt = Adam(lr=learning_rate)
    trainable_layers_model.compile(optimizer=opt, loss="mae", metrics=["mse", "mae", "mape"])

    return trainable_layers_model



def build_ffNN(n_folds , n_jobs, n_iter, param_grid_in, X_train, y_train,X_test):
    # Update Dictionary
    param_grid_in_internal = param_grid_in
    param_grid_in_internal['input_dim'] = [(X_train.shape[1])]
    
    # Deep Feature Network
    ffNN_CV = tf.keras.wrappers.scikit_learn.KerasRegressor(build_fn=get_ffNN, 
                                                            verbose=True)
    
    # Randomized CV
    ffNN_CVer = RandomizedSearchCV(estimator=ffNN_CV, 
                                    n_jobs=n_jobs,
                                    cv=KFold(n_folds, random_state=2020, shuffle=True),
                                    param_distributions=param_grid_in_internal,
                                    n_iter=n_iter,
                                    return_train_score=True,
                                    random_state=2020,
                                    verbose=10)
    
    # Fit Model #
    #-----------#
    ffNN_CVer.fit(X_train,y_train)

    # Write Predictions #
    #-------------------#
    y_hat_train = ffNN_CVer.predict(X_train)
    y_hat_test = ffNN_CVer.predict(X_test)
    
    # Counter number of parameters #
    #------------------------------#
    # Extract Best Model
    best_model = ffNN_CVer.best_estimator_
    # Count Number of Parameters
    N_params_best_ffNN = np.sum([np.prod(v.get_shape().as_list()) for v in best_model.model.trainable_variables])
    
    
    # Return Values #
    #---------------#
    return y_hat_train, y_hat_test, N_params_best_ffNN

# Update User
#-------------#
print('Deep Feature Builder - Ready')


# # Gradient-Boosted Random Forest Regressor

# In[ ]:


def get_GBRF(X_train,X_test,y_train):

    # Run Random Forest Util
    rand_forest_model_grad_boosted = GradientBoostingRegressor()

    # Grid-Search CV
    Random_Forest_GridSearch = RandomizedSearchCV(estimator = rand_forest_model_grad_boosted,
                                                  n_iter=n_iter_trees,
                                                  cv=KFold(CV_folds, random_state=2020, shuffle=True),
                                                  param_distributions=Rand_Forest_Grid,
                                                  return_train_score=True,
                                                  random_state=2020,
                                                  verbose=10,
                                                  n_jobs=n_jobs)

    random_forest_trained = Random_Forest_GridSearch.fit(X_train,y_train)
    random_forest_trained = random_forest_trained.best_estimator_

    #--------------------------------------------------#
    # Write: Model, Results, and Best Hyper-Parameters #
    #--------------------------------------------------#

    # Save Model
    # pickle.dump(random_forest_trained, open('./outputs/models/Gradient_Boosted_Tree/Gradient_Boosted_Tree_Best.pkl','wb'))

    # Save Readings
    cur_path = os.path.expanduser('./outputs/tables/best_params_Gradient_Boosted_Tree.txt')
    with open(cur_path, "w") as f:
        f.write(str(Random_Forest_GridSearch.best_params_))

    best_params_table_tree = pd.DataFrame({'N Estimators': [Random_Forest_GridSearch.best_params_['n_estimators']],
                                        'Min Samples Leaf': [Random_Forest_GridSearch.best_params_['min_samples_leaf']],
                                        'Learning Rate': [Random_Forest_GridSearch.best_params_['learning_rate']],
                                        'Max Depth': [Random_Forest_GridSearch.best_params_['max_depth']],
                                        })
    
    # Count Number of Parameters in Random Forest Regressor
    N_tot_params_per_tree = [ (x[0].tree_.node_count)*random_forest_trained.n_features_ for x in random_forest_trained.estimators_]
    N_tot_params_in_forest = sum(N_tot_params_per_tree)
    best_params_table_tree['N_parameters'] = [N_tot_params_in_forest]
    # Write Best Parameter(s)
    best_params_table_tree.to_latex('./outputs/tables/Best_params_table_Gradient_Boosted_Tree.txt')
    #---------------------------------------------#
    
    # Generate Prediction(s) #
    #------------------------#
    y_train_hat_random_forest_Gradient_boosting = random_forest_trained.predict(X_train)
    y_test_hat_random_forest_Gradient_boosting = random_forest_trained.predict(X_test)
    
    # Return Predictions #
    #--------------------#
    return y_train_hat_random_forest_Gradient_boosting, y_test_hat_random_forest_Gradient_boosting, random_forest_trained, N_tot_params_in_forest


# ## Kernel Ridge Regressor

# In[ ]:


def get_Kernel_Ridge_Regressor(data_x_in,data_x_test_in,data_y_in):
    # Imports
    from sklearn.svm import SVR
    from sklearn.kernel_ridge import KernelRidge

    # Initialize Randomized Gridsearch
    kernel_ridge_CVer = RandomizedSearchCV(estimator = KernelRidge(),
                                           n_jobs=n_jobs,
                                           cv=KFold(CV_folds, random_state=2020, shuffle=True),
                                           param_distributions=param_grid_kernel_Ridge,
                                           n_iter=n_iter,
                                           return_train_score=True,
                                           random_state=2020,
                                           verbose=10)
    kernel_ridge_CVer.fit(data_x_in,data_y_in)

    # Get best Kernel ridge regressor
    best_kernel_ridge_model = kernel_ridge_CVer.best_estimator_

    # Get Predictions
    f_hat_kernel_ridge_train = best_kernel_ridge_model.predict(data_x_in)
    f_hat_kernel_ridge_test = best_kernel_ridge_model.predict(data_x_test_in)

    # Count Parameters of best model
    N_params_kR = len(relic.get_params()) + 2*(data_x_in)
    
    Path('./outputs/models/Kernel_Ridge/').mkdir(parents=True, exist_ok=True)
    pd.DataFrame.from_dict(kernel_ridge_CVer.best_params_,orient='index').to_latex("./outputs/models/Kernel_Ridge/Best_Parameters.tex")
    
    
    
    # Return
    return f_hat_kernel_ridge_train, f_hat_kernel_ridge_test, best_kernel_ridge_model, N_params_kR


# ---

# # Train Models and Get Prediction(s)

# ### Elastic Net

# In[ ]:


print("--------------")
print("Training: ENET")
print("--------------")

# Fit Elastic Net Model
Lin_reg.fit(X_train,Y_train_mean_emp)

# Get Predictions
ENET_predict = Lin_reg.predict(X_train)
ENET_predict_test = Lin_reg.predict(X_test)

# Get Prediction Errors
## Train
ENET_errors = get_deterministic_errors(X_train,ENET_predict,Y_train,N_Bootstraps=N_Boostraps_BCA)
## Test
ENET_errors_test = get_deterministic_errors(X_test,ENET_predict_test,Y_test,N_Bootstraps=N_Boostraps_BCA)


# ## Kernel Ridge Regression

# In[ ]:


print("-----------------")
print("Training: K-Ridge")
print("-----------------")

Xhat_Kridge, Xhat_Kridge_test , relic, kRidge_N_params = get_Kernel_Ridge_Regressor(X_train,X_test,Y_train_mean_emp)


# Get Prediction Errors
## Train
kRidge_errors = get_deterministic_errors(X_train,Xhat_Kridge,Y_train,N_Bootstraps=N_Boostraps_BCA)
## Test
kRidge_errors_test = get_deterministic_errors(X_test,Xhat_Kridge_test,Y_test,N_Bootstraps=N_Boostraps_BCA)


# ## Gradient-Boosted Random Forest

# In[ ]:


print("--------------")
print("Training: GBRF")
print("--------------")

GBRF_y_hat_train, GBRF_y_hat_test, GBRF_model, GBRF_N_Params = get_GBRF(X_train,X_test,Y_train_mean_emp)

# Get Prediction Errors
## Train
GBRF_errors = get_deterministic_errors(X_train,GBRF_y_hat_train,Y_train,N_Bootstraps=N_Boostraps_BCA)
## Test
GBRF_errors_test = get_deterministic_errors(X_test,GBRF_y_hat_test,Y_test,N_Bootstraps=N_Boostraps_BCA)


# ## Feed-Forward DNN

# In[ ]:


print("-------------")
print("Training: DNN")
print("-------------")

# Redefine (Dimension-related) Elements of Grid
param_grid_Deep_Classifier['input_dim'] = [problem_dim]
param_grid_Deep_Classifier['output_dim'] = [1]

YHat_ffNN, YHat_ffNN_test, ffNN_N_Params = build_ffNN(n_folds = CV_folds,
                                                      n_jobs = n_jobs, 
                                                      n_iter = n_iter,
                                                      param_grid_in = param_grid_Deep_Classifier,  
                                                      X_train = X_train,
                                                      y_train = Y_train_mean_emp,
                                                      X_test = X_test)


# Get Prediction Errors
## Train
ffNN_errors = get_deterministic_errors(X_train,YHat_ffNN,Y_train,N_Bootstraps=N_Boostraps_BCA)
## Test
ffNN_errors_test = get_deterministic_errors(X_test,YHat_ffNN_test,Y_test,N_Bootstraps=N_Boostraps_BCA)


# # Compute Error Metric(s)

# In[ ]:


print("-----------------------")
print("Computing Error Metrics")
print("-----------------------")

Summary_mean_models = pd.DataFrame({"ENET":ENET_errors_test[:,1],
                                    "kRidge":kRidge_errors_test[:,1],
                                    "GBRF":GBRF_errors_test[:,1],
                                    "ffNN":ffNN_errors_test[:,1],
                                   },index=["W1","Mean","Var","Skewness","Ex_Kur","Higher"])

Summary_mean_models_test = pd.DataFrame({"ENET":ENET_errors[:,1],
                                    "kRidge":kRidge_errors[:,1],
                                    "GBRF":GBRF_errors[:,1],
                                    "ffNN":ffNN_errors[:,1],
                                   },index=["W1","Mean","Var","Skewness","Ex_Kur","Higher"])

## Save Facts
Summary_mean_models.to_latex((results_tables_path+str(f_unknown_mode)+"Problemdimension"+str(problem_dim)+"__SUMMARY_METRICS.tex"))
Summary_mean_models_test.to_latex((results_tables_path+str(f_unknown_mode)+"Problemdimension"+str(problem_dim)+"__SUMMARY_METRICS_test.tex"))

