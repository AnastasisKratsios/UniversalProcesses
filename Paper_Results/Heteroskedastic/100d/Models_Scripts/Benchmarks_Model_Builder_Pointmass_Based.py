#!/usr/bin/env python
# coding: utf-8

# # Define Model(s)

# ---

# In[8]:


# %run Debug_Menu.ipynb


# ## Elastic Net Regressor

# In[16]:


#=====================#
# Elastic Net Version #
#=====================#
# Block warnings that spam when performing coordinate descent (by default) in 1-d.
import warnings
from sklearn.linear_model import ElasticNetCV
warnings.filterwarnings("ignore")
# Initialize Elastic Net Regularization Model
ENET_reg = ElasticNetCV(cv=5, random_state=0, alphas = np.linspace(0,(10**2),(10**2)),
                           l1_ratio=np.linspace(0,1,(10**2)))


# ## ffNN Builder

# In[17]:


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
    
    eval_time_ffNN = time.time()
    y_hat_test = ffNN_CVer.predict(X_test)
    eval_time_ffNN = time.time() - eval_time_ffNN
    
    # Counter number of parameters #
    #------------------------------#
    # Extract Best Model
    best_model = ffNN_CVer.best_estimator_
    # Count Number of Parameters
    N_params_best_ffNN = np.sum([np.prod(v.get_shape().as_list()) for v in best_model.model.trainable_variables])
    
    
    # Return Values #
    #---------------#
    return y_hat_train, y_hat_test, N_params_best_ffNN, eval_time_ffNN

# Update User
#-------------#
print('Deep Feature Builder - Ready')


# # Gradient-Boosted Random Forest Regressor

# In[18]:


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
    eval_time_GBRF = time.time()
    y_test_hat_random_forest_Gradient_boosting = random_forest_trained.predict(X_test)
    eval_time_GBRF = time.time() - eval_time_GBRF
    
    # Return Predictions #
    #--------------------#
    return y_train_hat_random_forest_Gradient_boosting, y_test_hat_random_forest_Gradient_boosting, random_forest_trained, N_tot_params_in_forest, eval_time_GBRF


# ## Kernel Ridge Regressor

# In[19]:


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
    eval_time_kr = time.time()
    f_hat_kernel_ridge_test = best_kernel_ridge_model.predict(data_x_test_in)
    eval_time_kr = time.time() - eval_time_kr

    # Count Parameters of best model
    N_params_kR = len(best_kernel_ridge_model.get_params()) + 2*problem_dim
    
    Path('./outputs/models/Kernel_Ridge/').mkdir(parents=True, exist_ok=True)
    pd.DataFrame.from_dict(kernel_ridge_CVer.best_params_,orient='index').to_latex("./outputs/models/Kernel_Ridge/Best_Parameters.tex")
    
    
    
    # Return
    return f_hat_kernel_ridge_train, f_hat_kernel_ridge_test, best_kernel_ridge_model, N_params_kR, eval_time_kr


# ---

# # Train Models and Get Prediction(s)

# ### Elastic Net

# In[1]:


# %run Evaluation.ipynb
exec(open('./Helper_Scripts_and_Loading/Evaluation.py').read())


# In[21]:


# Start Timer
Timer_ENET = time.time()

print("--------------")
print("Training: ENET")
print("--------------")

if output_dim == 1:
    # Fit Elastic Net Model
    ENET_reg.fit(X_train,Y_train_mean_emp)

    # Get Predictions
    ENET_predict = ENET_reg.predict(X_train)
    ENET_eval_time = time.time()
    ENET_predict_test = ENET_reg.predict(X_test)
    ENET_eval_time = time.time() - ENET_eval_time
    ENET_N_Params = X_train.shape[1]*2
    
else:
    for d in tqdm(range(output_dim)):
        # Fit Elastic Net Model
        ENET_reg.fit(X_train,Y_train_mean_emp[:,d])

        # Get Predictions
        ENET_predict_loop = ENET_reg.predict(X_train)
        ENET_eval_time_loop = time.time()
        ENET_predict_loop_test = ENET_reg.predict(X_test)
        ENET_eval_time_loop = time.time() - ENET_eval_time_loop
    
        if d == 0:
            ENET_predict = ENET_predict_loop.reshape(-1,1)
            ENET_predict_test = ENET_predict_loop_test.reshape(-1,1)
            ENET_N_Params = X_train.shape[1]*2
            ENET_eval_time = ENET_eval_time_loop
        else:
            ENET_predict = np.append(ENET_predict,ENET_predict_loop.reshape(-1,1),axis=1)
            ENET_predict_test = np.append(ENET_predict_test,ENET_predict_loop_test.reshape(-1,1),axis=1)
            ENET_N_Params = ENET_N_Params + X_train.shape[0]*2
            ENET_eval_time = ENET_eval_time + ENET_eval_time_loop

# Stop Timer
Timer_ENET = time.time() - Timer_ENET
print("---------------------")
print("Training: ENET - Done")
print("---------------------")


# #### Evaluate Prediction Quality

# In[15]:


## Train
ENET_errors_W1, ENET_errors_M1 = get_deterministic_errors(X_train,
                                                          ENET_predict,
                                                          Y_train,
                                                          N_Bootstraps=N_Boostraps_BCA)
## Test
ENET_errors_W1_test, ENET_errors_M1_test = get_deterministic_errors(X_test,
                                                                    ENET_predict_test,
                                                                    Y_test,
                                                                    N_Bootstraps=N_Boostraps_BCA)
# Stop Timer
Timer_ENET = time.time() - Timer_ENET


# #### Update DataFrame

# In[13]:


# Train
Summary_pred_Qual_models_internal["ENET"] = pd.Series(np.append(np.append(ENET_errors_W1,
                                                                ENET_errors_M1),
                                                         np.array([ENET_N_Params,
                                                                   Timer_ENET,
                                                                   (ENET_eval_time/Test_Set_PredictionTime_MC)])), index=Summary_pred_Qual_models.index)
# Test
Summary_pred_Qual_models_test["ENET"] = pd.Series(np.append(np.append(ENET_errors_W1_test,
                                                                     ENET_errors_M1_test),
                                                           np.array([ENET_N_Params,
                                                                     Timer_ENET,
                                                                     (ENET_eval_time/Test_Set_PredictionTime_MC)])), index=Summary_pred_Qual_models_test.index)

print("Updated DataFrame")
## Get Worst-Case
Summary_pred_Qual_models_train = Summary_pred_Qual_models_internal
Summary_pred_Qual_models_internal = np.maximum(Summary_pred_Qual_models_internal,Summary_pred_Qual_models_test)
Summary_pred_Qual_models = np.copy(Summary_pred_Qual_models_internal)
## Write Performance Metrics
Summary_pred_Qual_models_internal.to_latex((results_tables_path+"Performance_metrics_Problem_Type_"+str(f_unknown_mode)+"Problemdimension"+str(problem_dim)+"__SUMMARY_METRICS.tex"))
Summary_pred_Qual_models_train.to_latex((results_tables_path+"Performance_metrics_Problem_Type_"+str(f_unknown_mode)+"Problemdimension"+str(problem_dim)+"__SUMMARY_METRICS_train.tex"))
Summary_pred_Qual_models_test.to_latex((results_tables_path+"Performance_metrics_Problem_Type_"+str(f_unknown_mode)+"Problemdimension"+str(problem_dim)+"__SUMMARY_METRICS_test.tex"))

# Remove W1 estimates from x \mapsto \delta_{f(x)}
Summary_pred_Qual_models = Summary_pred_Qual_models_internal.copy()
Summary_pred_Qual_models.loc[['W1-95L','W1','W1-95R'],['ENET']] = "-"
Summary_pred_Qual_models.to_latex((results_tables_path+"Final_Results/Performance_metrics_Problem_Type_"+str(f_unknown_mode)+"Problemdimension"+str(problem_dim)+"__SUMMARY_METRICS.tex"))

# Update User
print(Summary_pred_Qual_models)
print("------------------------------------------------")
print("Updated Performance Metrics Dataframe and Saved!")
print("------------------------------------------------")


# ## Kernel Ridge Regression

# In[14]:


# Stop Timer
Timer_kRidge = time.time()


print("-----------------")
print("Training: K-Ridge")
print("-----------------")

Xhat_Kridge, Xhat_Kridge_test , relic, kRidge_N_params, KRidge_eval_time = get_Kernel_Ridge_Regressor(X_train,X_test,Y_train_mean_emp)


# #### Get Prediction Errors

# In[15]:


## Train
kRidge_errors_W1, kRidge_errors_M1 = get_deterministic_errors(X_train,Xhat_Kridge,Y_train,N_Bootstraps=N_Boostraps_BCA)
## Test
kRidge_errors_W1_test, kRidge_errors_M1_test = get_deterministic_errors(X_test,Xhat_Kridge_test,Y_test,N_Bootstraps=N_Boostraps_BCA)
# Stop Timer
Timer_kRidge = time.time() - Timer_kRidge


# #### Update DataFrame

# In[16]:


# Train
Summary_pred_Qual_models_internal["KRidge"] = pd.Series(np.append(np.append(kRidge_errors_W1,
                                                                kRidge_errors_M1),
                                                         np.array([0,
                                                                   Timer_kRidge,
                                                                   (KRidge_eval_time/Test_Set_PredictionTime_MC)])), index=Summary_pred_Qual_models.index)
# Test
Summary_pred_Qual_models_test["KRidge"] = pd.Series(np.append(np.append(kRidge_errors_W1_test,
                                                                     kRidge_errors_M1_test),
                                                           np.array([0,
                                                                     Timer_kRidge,
                                                                     (KRidge_eval_time/Test_Set_PredictionTime_MC)])), index=Summary_pred_Qual_models_test.index)

print("Updated DataFrame")
## Get Worst-Case
Summary_pred_Qual_models_train = Summary_pred_Qual_models_internal
Summary_pred_Qual_models_internal = np.maximum(Summary_pred_Qual_models_internal,Summary_pred_Qual_models_test)
Summary_pred_Qual_models = np.copy(Summary_pred_Qual_models_internal)
## Write Performance Metrics
Summary_pred_Qual_models_internal.to_latex((results_tables_path+"Performance_metrics_Problem_Type_"+str(f_unknown_mode)+"Problemdimension"+str(problem_dim)+"__SUMMARY_METRICS.tex"))
Summary_pred_Qual_models_train.to_latex((results_tables_path+"Performance_metrics_Problem_Type_"+str(f_unknown_mode)+"Problemdimension"+str(problem_dim)+"__SUMMARY_METRICS_train.tex"))
Summary_pred_Qual_models_test.to_latex((results_tables_path+"Performance_metrics_Problem_Type_"+str(f_unknown_mode)+"Problemdimension"+str(problem_dim)+"__SUMMARY_METRICS_test.tex"))

# Remove W1 estimates from x \mapsto \delta_{f(x)}
Summary_pred_Qual_models = Summary_pred_Qual_models_internal.copy()
Summary_pred_Qual_models.loc[['W1-95L','W1','W1-95R'],['ENET','KRidge']] = "-"
Summary_pred_Qual_models.to_latex((results_tables_path+"Final_Results/Performance_metrics_Problem_Type_"+str(f_unknown_mode)+"Problemdimension"+str(problem_dim)+"__SUMMARY_METRICS.tex"))

# Update User
print(Summary_pred_Qual_models)
print("------------------------------------------------")
print("Updated Performance Metrics Dataframe and Saved!")
print("------------------------------------------------")


# ## Gradient-Boosted Random Forest

# In[17]:


# Start Timer
Timer_GBRF = time.time()

print("--------------")
print("Training: GBRF")
print("--------------")

if output_dim == 1:
    GBRF_y_hat_train, GBRF_y_hat_test, GBRF_model, GBRF_N_Params, GBRF_eval_time = get_GBRF(X_train,
                                                                                            X_test,
                                                                                            Y_train_mean_emp)
else:
    for d in range(output_dim):
        GBRF_y_hat_train_loop, GBRF_y_hat_test_loop, GBRF_model, GBRF_N_Params_loop, GBRF_eval_time_loop = get_GBRF(X_train,
                                                                                                                X_test,
                                                                                                                Y_train_mean_emp[:,d])
        if d == 0:
            GBRF_y_hat_train = GBRF_y_hat_train_loop.reshape(-1,1)
            GBRF_y_hat_test = GBRF_y_hat_test_loop.reshape(-1,1)
            GBRF_N_Params = GBRF_N_Params_loop
            GBRF_eval_time = GBRF_eval_time_loop
        else:
            GBRF_y_hat_train = np.append(GBRF_y_hat_train,GBRF_y_hat_train_loop.reshape(-1,1),axis=1)
            GBRF_y_hat_test = np.append(GBRF_y_hat_test,GBRF_y_hat_test_loop.reshape(-1,1),axis=1)
            GBRF_N_Params = GBRF_N_Params + GBRF_N_Params_loop
            GBRF_eval_time = GBRF_eval_time + GBRF_eval_time_loop


# #### Get Prediction Errors

# In[18]:


## Train
GBRF_errors_W1, GBRF_errors_M1 = get_deterministic_errors(X_train,GBRF_y_hat_train,Y_train,N_Bootstraps=N_Boostraps_BCA)
## Test
GBRF_errors_W1_test, GBRF_errors_M1_test = get_deterministic_errors(X_test,GBRF_y_hat_test,Y_test,N_Bootstraps=N_Boostraps_BCA)


# Compute Lapsed Time Needed For Training
Timer_GBRF = time.time() - Timer_GBRF


# #### Update Dataframe

# In[19]:


# Train
Summary_pred_Qual_models_internal["GBRF"] = pd.Series(np.append(np.append(GBRF_errors_W1,
                                                                GBRF_errors_M1),
                                                         np.array([GBRF_N_Params,
                                                                   Timer_GBRF,
                                                                   (GBRF_eval_time/Test_Set_PredictionTime_MC)])), index=Summary_pred_Qual_models.index)
# Test
Summary_pred_Qual_models_test["GBRF"] = pd.Series(np.append(np.append(GBRF_errors_W1_test,
                                                                     GBRF_errors_M1_test),
                                                           np.array([GBRF_N_Params,
                                                                     Timer_GBRF,
                                                                     (GBRF_eval_time/Test_Set_PredictionTime_MC)])), index=Summary_pred_Qual_models_test.index)

print("Updated DataFrame")
## Get Worst-Case
Summary_pred_Qual_models_train = Summary_pred_Qual_models_internal
Summary_pred_Qual_models_internal = np.maximum(Summary_pred_Qual_models_internal,Summary_pred_Qual_models_test)
Summary_pred_Qual_models = np.copy(Summary_pred_Qual_models_internal)
## Write Performance Metrics
Summary_pred_Qual_models_internal.to_latex((results_tables_path+"Performance_metrics_Problem_Type_"+str(f_unknown_mode)+"Problemdimension"+str(problem_dim)+"__SUMMARY_METRICS.tex"))
Summary_pred_Qual_models_train.to_latex((results_tables_path+"Performance_metrics_Problem_Type_"+str(f_unknown_mode)+"Problemdimension"+str(problem_dim)+"__SUMMARY_METRICS_train.tex"))
Summary_pred_Qual_models_test.to_latex((results_tables_path+"Performance_metrics_Problem_Type_"+str(f_unknown_mode)+"Problemdimension"+str(problem_dim)+"__SUMMARY_METRICS_test.tex"))

# Remove W1 estimates from x \mapsto \delta_{f(x)}
Summary_pred_Qual_models = Summary_pred_Qual_models_internal.copy()
Summary_pred_Qual_models.loc[['W1-95L','W1','W1-95R'],['ENET','KRidge','ENET','GBRF']] = "-"
Summary_pred_Qual_models.to_latex((results_tables_path+"Final_Results/Performance_metrics_Problem_Type_"+str(f_unknown_mode)+"Problemdimension"+str(problem_dim)+"__SUMMARY_METRICS.tex"))

# Update User
print(Summary_pred_Qual_models)
print("------------------------------------------------")
print("Updated Performance Metrics Dataframe and Saved!")
print("------------------------------------------------")


# ## Feed-Forward DNN

# In[20]:


# Stop Timer
Timer_ffNN = time.time()
print("-------------")
print("Training: DNN")
print("-------------")

# Redefine (Dimension-related) Elements of Grid
param_grid_Deep_ffNN['input_dim'] = [problem_dim]
param_grid_Deep_ffNN['output_dim'] = [output_dim]

YHat_ffNN, YHat_ffNN_test, ffNN_N_Params, ffNN_eval_time = build_ffNN(n_folds = CV_folds,
                                                                      n_jobs = n_jobs, 
                                                                      n_iter = n_iter,
                                                                      param_grid_in = param_grid_Deep_ffNN,  
                                                                      X_train = X_train,
                                                                      y_train = Y_train_mean_emp,
                                                                      X_test = X_test)


# #### Get Prediction Errors

# In[21]:


## Train
ffNN_errors_W1,ffNN_errors_M1 = get_deterministic_errors(X_train,YHat_ffNN,Y_train,N_Bootstraps=N_Boostraps_BCA)
## Test
ffNN_errors_W1_test,ffNN_errors_M1_test = get_deterministic_errors(X_test,YHat_ffNN_test,Y_test,N_Bootstraps=N_Boostraps_BCA)

# Compute Lapsed Time Needed For Training
Timer_ffNN =  time.time() - Timer_ffNN


# #### Update DataFrame

# In[22]:


# Train
Summary_pred_Qual_models_internal["DNN"] = pd.Series(np.append(np.append(ffNN_errors_W1,
                                                                ffNN_errors_M1),
                                                         np.array([ffNN_N_Params,
                                                                   Timer_ffNN,
                                                                   (ffNN_eval_time/Test_Set_PredictionTime_MC)])), index=Summary_pred_Qual_models.index)
# Test
Summary_pred_Qual_models_test["DNN"] = pd.Series(np.append(np.append(ffNN_errors_W1_test,
                                                                     ffNN_errors_M1_test),
                                                           np.array([ffNN_N_Params,
                                                                     Timer_ffNN,
                                                                     (ffNN_eval_time/Test_Set_PredictionTime_MC)])), index=Summary_pred_Qual_models_test.index)

print("Updated DataFrame")
## Get Worst-Case
Summary_pred_Qual_models_train = Summary_pred_Qual_models_internal
Summary_pred_Qual_models_internal = np.maximum(Summary_pred_Qual_models_internal,Summary_pred_Qual_models_test)
Summary_pred_Qual_models = np.copy(Summary_pred_Qual_models_internal)
## Write Performance Metrics
Summary_pred_Qual_models_internal.to_latex((results_tables_path+"Performance_metrics_Problem_Type_"+str(f_unknown_mode)+"Problemdimension"+str(problem_dim)+"__SUMMARY_METRICS.tex"))
Summary_pred_Qual_models_train.to_latex((results_tables_path+"Performance_metrics_Problem_Type_"+str(f_unknown_mode)+"Problemdimension"+str(problem_dim)+"__SUMMARY_METRICS_train.tex"))
Summary_pred_Qual_models_test.to_latex((results_tables_path+"Performance_metrics_Problem_Type_"+str(f_unknown_mode)+"Problemdimension"+str(problem_dim)+"__SUMMARY_METRICS_test.tex"))

# Remove W1 estimates from x \mapsto \delta_{f(x)}
Summary_pred_Qual_models = Summary_pred_Qual_models_internal.copy()
Summary_pred_Qual_models.loc[['W1-95L','W1','W1-95R'],['ENET','KRidge','ENET','GBRF','DNN']] = "-"
Summary_pred_Qual_models.to_latex((results_tables_path+"Final_Results/Performance_metrics_Problem_Type_"+str(f_unknown_mode)+"Problemdimension"+str(problem_dim)+"__SUMMARY_METRICS.tex"))

# Update User
print(Summary_pred_Qual_models)
print("------------------------------------------------")
print("Updated Performance Metrics Dataframe and Saved!")
print("------------------------------------------------")


# ### Remove meaningless value of $\delta_{f(x)}$

# In[1]:


# Remove W1 estimates from x \mapsto \delta_{f(x)}
Summary_pred_Qual_models = Summary_pred_Qual_models_internal.copy()
Summary_pred_Qual_models.loc[['W1-95L','W1','W1-95R'],['ENET','KRidge','ENET','GBRF','DNN']] = "-"
Summary_pred_Qual_models.to_latex((results_tables_path+"Final_Results/Performance_metrics_Problem_Type_"+str(f_unknown_mode)+"Problemdimension"+str(problem_dim)+"__SUMMARY_METRICS.tex"))


# ---

# ---
# # Fin
# ---
