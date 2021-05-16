#!/usr/bin/env python
# coding: utf-8

# # Training of Bishop's Mixture Density Network

# In[ ]:


# Trial_run = True
# exec(open('Init_Dump.py').read())
# %run Debug_Menu.ipynb


# #### Reset Meta-Parameters

# In[ ]:


# Redefine (Dimension-related) Elements of Grid
exec(open('./Helper_Scripts_and_Loading/Init_Dump.py').read())
import time


# ---

# # Re-Define ffNN Builder (Internally to this script)

# In[ ]:


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


# ---

# #### Start Timer:

# In[ ]:


Bishop_MDN_Timer = time.time()


# ## Prepare Training Data

# In[ ]:


print("======================================================")
print("Preparing Training Outputs for MDNs using EM-Algorithm")
print("======================================================")

# Initializizations #
#-------------------#
## Count Number of Centers
N_GMM_clusters = int(np.minimum(N_Quantizers_to_parameterize,Y_train.shape[1]-1))
## Timer: Start
timer_GMM_data_preparation = time.time()

# Get Training Data #
#-------------------#
for i in tqdm(range(X_train.shape[0])):
    # Train GMM
    gmm_loop = GaussianMixture(n_components=N_GMM_clusters)
    gmm_loop.fit(Y_train[i,].reshape(-1,1))
    # Get Fit Parameter(s)
    means_GMM_loop = gmm_loop.means_.reshape(1,-1)
    sds_GMM_loop = gmm_loop.covariances_.reshape(1,-1)
    mixture_coefficients = gmm_loop.weights_.reshape(1,-1)
    
    # Update Targets #
    #----------------#
    if i == 0:
        Y_MDN_targets_train_mean = means_GMM_loop
        Y_MDN_targets_train_sd = sds_GMM_loop
        Y_MDN_targets_train_mixture_weights = mixture_coefficients
    else:
        Y_MDN_targets_train_mean = np.append(Y_MDN_targets_train_mean,means_GMM_loop,axis=0)
        Y_MDN_targets_train_sd = np.append(Y_MDN_targets_train_sd,sds_GMM_loop,axis=0)
        Y_MDN_targets_train_mixture_weights = np.append(Y_MDN_targets_train_mixture_weights,mixture_coefficients,axis=0)

# Timer: Stop
timer_GMM_data_preparation = time.time() - timer_GMM_data_preparation

print("======================================================")
print("Prepared Training Outputs for MDNs using EM-Algorithm!")
print("======================================================")


# ## Define Model Components (Sub-Networks)

# #### Update Grid Based on Identified Cluster Number

# In[ ]:


param_grid_Deep_ffNN['input_dim'] = [problem_dim]
param_grid_Deep_ffNN['output_dim'] = [N_GMM_clusters]


# ### Means Network

# This is just a vanilla ffNN!

# ### Train SDs Network
# This one needs some customization!

# #### Define Architecture and Network Builder

# In[ ]:


# Affine Readout post-composed with UAP-preserving readout map to G_d
class SD_output(tf.keras.layers.Layer):

    def __init__(self, units=16, input_dim=32):
        super(SD_output, self).__init__()
        self.units = units

    def build(self, input_shape):
        self.w = self.add_weight(name='Weights_ffNN',
                                 shape=(input_shape[-1], self.units),
                               initializer='random_normal',
                               trainable=True)
        self.b = self.add_weight(name='bias_ffNN',
                                 shape=(self.units,),
                               initializer='random_normal',
                               trainable=True)

    def call(self, inputs):
        parameters = tf.matmul(inputs, self.w) + self.b
        sd_out = tf.math.exp(parameters)
        return sd_out


# In[ ]:


def get_MDN_SDs_SubNetwork(height, depth, learning_rate, input_dim, output_dim):
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
    # Gaussian Splitter Layer
    output_layers = SD_output(output_dim)(core_layers)  
    # Define Input/Output Relationship (Arch.)
    trainable_layers_model = tf.keras.Model(input_layer, output_layers)
    
    
    #----------------------------------#
    # Define Optimizer & Compile Archs.
    #----------------------------------#
    opt = Adam(lr=learning_rate)
    trainable_layers_model.compile(optimizer=opt, loss="mae", metrics=["mse", "mae", "mape"])

    return trainable_layers_model

#----------------------------------------------------------------------------------------------------#

def build_MDN_SDs_SubNetwork(n_folds , n_jobs, n_iter, param_grid_in, X_train, y_train,X_test):
    # Update Dictionary
    param_grid_in_internal = param_grid_in
    param_grid_in_internal['input_dim'] = [(X_train.shape[1])]
    
    # Deep Feature Network
    ffNN_CV = tf.keras.wrappers.scikit_learn.KerasRegressor(build_fn=get_MDN_SDs_SubNetwork, 
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


# ---

# ## Train Sub-Networks

# #### Train Means Network

# In[ ]:


print("(0)")
print("=====================================================")
print("Training Mixture Density Network (MDN): Means: Start!")
print("=====================================================")
# Train simple deep classifier
timer_MDN_Means = time.time()

MDN_Means_train, MDN_Means_test, N_params_MDN_MeansNet, timer_output_MDN_MeansNet = build_ffNN(n_folds = CV_folds,
                                                                                               n_jobs = n_jobs,
                                                                                               n_iter = n_iter,
                                                                                               param_grid_in=param_grid_Deep_ffNN,
                                                                                               X_train = X_train,
                                                                                               y_train = Y_MDN_targets_train_mean,
                                                                                               X_test = X_test)

# Format as float
MDN_Means_train = np.array(MDN_Means_train,dtype=float)
MDN_Means_test = np.array(MDN_Means_test,dtype=float)
timer_MDN_Means = time.time() - timer_MDN_Means
print("===================================================")
print("Training Mixture Density Network (MDN): Means: END!")
print("===================================================")


# #### Train Standard-Deviations Network

# In[ ]:


print("(1)")
print("===================================================")
print("Training Mixture Density Network (MDN): SD: Start!")
print("===================================================")

# Train simple deep classifier
timer_MDN_SDs = time.time()
MDN_SDs_train, MDN_SDs_test, N_params_MDN_SDsNet, timer_output_MDN_SDsNet = build_MDN_SDs_SubNetwork(n_folds = CV_folds,
                                                                                                             n_jobs = n_jobs,
                                                                                                             n_iter = n_iter,
                                                                                                             param_grid_in=param_grid_Deep_ffNN,
                                                                                                             X_train = X_train,
                                                                                                             y_train = Y_MDN_targets_train_sd,
                                                                                                             X_test = X_test)
# Format as float
MDN_SDs_train = np.array(MDN_SDs_train,dtype=float)
MDN_SDs_test = np.array(MDN_SDs_test,dtype=float)
timer_MDN_SDs = time.time() - timer_MDN_SDs
print("=================================================")
print("Training Mixture Density Network (MDN): SD: END!")
print("=================================================")


# #### Train Mixture Coefficient Network

# In[ ]:


print("(2)")
print("====================================================================")
print("Training Mixture Density Network (MDN): Mixture Coefficients: Start!")
print("====================================================================")
# Train simple deep classifier
timer_MDN_Mix = time.time()
MDN_Mix_train, MDN_Mix_test, N_params_MDN_MixNet, timer_output_MDN_MixNet = build_simple_deep_classifier(n_folds = CV_folds,
                                                                                                         n_jobs = n_jobs,
                                                                                                         n_iter = n_iter,
                                                                                                         param_grid_in=param_grid_Deep_ffNN,
                                                                                                         X_train = X_train,
                                                                                                         y_train = Y_MDN_targets_train_mixture_weights,
                                                                                                         X_test = X_test)

# Format as float
MDN_Mix_train = np.array(MDN_Mix_train,dtype=float)
MDN_Mix_test = np.array(MDN_Mix_test,dtype=float)
timer_MDN_Mix = time.time() - timer_MDN_Mix
print("==================================================================")
print("Training Mixture Density Network (MDN): Mixture Coefficients: END!")
print("==================================================================")


# ### Get Prediction(s)

# #### Train:

# In[ ]:


print("#--------------------#")
print(" Get Training Error(s)")
print("#--------------------#")
for i in tqdm(range((X_train.shape[0]))):
    for j in range(N_GMM_clusters):
        points_of_mass_loop = np.random.normal(MDN_Means_train[i,j],
                                               MDN_SDs_train[i,j],
                                               N_Monte_Carlo_Samples)
        b_loop = np.repeat(MDN_Mix_train[i,j],
                           N_Monte_Carlo_Samples)
        if j == 0:
            b = b_loop
            points_of_mass_MDN_train = points_of_mass_loop
        else:
            b = np.append(b,b_loop)
            points_of_mass_MDN_train = np.append(points_of_mass_MDN_train,points_of_mass_loop)
        points_of_mass_MDN_train = points_of_mass_MDN_train.reshape(-1,1)
        b = b.reshape(-1,1)
    points_of_mass_MDN_train = np.array(points_of_mass_MDN_train,dtype=float).reshape(-1,)
    b = np.array(b,dtype=float).reshape(-1,)
    b = b/N_Monte_Carlo_Samples
    
    # Compute Error(s)
    ## W1
    W1_loop_MDN_train = ot.emd2_1d(points_of_mass_MDN_train,
                                   np.array(Y_train[i,]).reshape(-1,),
                                   b,
                                   empirical_weights)
    
    ## M1
    Mu_hat_MDN = np.sum(b*(points_of_mass_MDN_train))
    Mu_MC = np.mean(np.array(Y_train[i,]))
    if f_unknown_mode == "Heteroskedastic_NonLinear_Regression":
        Mu = direct_facts[i,]
    else:
        Mu = Mu_MC
        
    ### Error(s)
    Mean_loop = np.sum(np.abs(Mu_hat_MDN-Mu))
    
    ## Variance
    Var_hat_MDN = np.sum(((points_of_mass_MDN_train-Mu_hat_MDN)**2)*b)
    Var_MC = np.mean(np.array(Y_train[i]-Mu_MC)**2)
    if f_unknown_mode == "Heteroskedastic_NonLinear_Regression":
        Var = 2*np.sum(X_train[i,]**2)
    else:
        Var = Var_MC     
    ### Error(s)
    Var_loop = np.sum(np.abs(Var_hat-Var))
        
    # Skewness
    Skewness_hat_MDN = np.sum((((points_of_mass_MDN_train-Mu_hat_MDN)/Var_hat_MDN)**3)*b)
    Skewness_MC = np.mean((np.array(Y_train[i]-Mu_MC)/Var_MC)**3)
    if f_unknown_mode == "Heteroskedastic_NonLinear_Regression":
        Skewness = 0
    else:
        Skewness = Skewness_MC
    ### Error(s)
    Skewness_loop = np.abs(Skewness_hat_MDN-Skewness)
    
    # Skewness
    Ex_Kurtosis_hat_MDN = np.sum((((points_of_mass_MDN_train-Mu_hat_MDN)/Var_hat_MDN)**4)*b) - 3
    Ex_Kurtosis_MC = np.mean((np.array(Y_train[i]-Mu_MC)/Var_MC)**4) - 3
    if f_unknown_mode == "Heteroskedastic_NonLinear_Regression":
        Ex_Kurtosis = 3
    else:
        Ex_Kurtosis = Ex_Kurtosis_MC
    ### Error(s)
    Ex_Kurtosis_loop = np.abs(Ex_Kurtosis-Ex_Kurtosis_hat_MDN)
    
    
    # Update
    if i == 0:
        W1_Errors_MDN = W1_loop_MDN_train
        # Moments
        Mean_Errors_MDN =  Mean_loop
        Var_Errors_MDN = Var_loop
        Skewness_Errors_MDN = Skewness_loop
        Ex_Kurtosis_Errors_MDN = Ex_Kurtosis_loop
        
        
    else:
        W1_Errors_MDN = np.append(W1_Errors_MDN,W1_loop_MDN_train)
        # Moments
        Mean_Errors_MDN =  np.append(Mean_Errors_MDN,Mean_loop)
        Var_Errors_MDN = np.append(Var_Errors_MDN,Var_loop)
        Skewness_Errors_MDN = np.append(Skewness_Errors_MDN,Skewness_loop)
        Ex_Kurtosis_Errors_MDN = np.append(Ex_Kurtosis_Errors_MDN,Ex_Kurtosis_loop)
        
print("#-------------------------#")
print(" Get Training Error(s): END")
print("#-------------------------#")


# #### Test:

# In[ ]:


print("#--------------------#")
print(" Get Test Error(s)")
print("#--------------------#")
for i in tqdm(range((X_test.shape[0]))):
    for j in range(N_GMM_clusters):
        points_of_mass_loop = np.random.normal(MDN_Means_test[i,j],
                                               MDN_SDs_test[i,j],
                                               N_Monte_Carlo_Samples)
        b_loop = np.repeat(MDN_Mix_test[i,j],N_Monte_Carlo_Samples)
        if j == 0:
            b = b_loop
            points_of_mass_MDN = points_of_mass_loop
        else:
            b = np.append(b,b_loop)
            points_of_mass_MDN = np.append(points_of_mass_MDN,points_of_mass_loop)
        points_of_mass_MDN = points_of_mass_MDN.reshape(-1,1)
        b = b.reshape(-1,1)
    points_of_mass_MDN = np.array(points_of_mass_MDN,dtype=float).reshape(-1,)
    b = np.array(b,dtype=float).reshape(-1,)
    b = b/N_Monte_Carlo_Samples
    
    # Compute Error(s)
    ## W1
    W1_loop_MDN = ot.emd2_1d(points_of_mass_MDN,
                             np.array(Y_test[i,]).reshape(-1,),
                             b,
                             empirical_weights)
    
    ## M1
    Mu_hat_MDN = np.sum(b*(points_of_mass_MDN))
    Mu_MC = np.mean(np.array(Y_test[i,]))
    if f_unknown_mode == "Heteroskedastic_NonLinear_Regression":
        Mu = direct_facts_test[i,]
    else:
        Mu = Mu_MC
        
    ### Error(s)
    Mean_loop = np.sum(np.abs(Mu_hat_MDN-Mu))
    
    ## Variance
    Var_hat_MDN = np.sum(((points_of_mass_MDN-Mu_hat_MDN)**2)*b)
    Var_MC = np.mean(np.array(Y_test[i]-Mu_MC)**2)
    if f_unknown_mode == "Heteroskedastic_NonLinear_Regression":
        Var = 2*np.sum(X_test[i,]**2)
    else:
        Var = Var_MC     
    ### Error(s)
    Var_loop = np.sum(np.abs(Var_hat-Var))
        
    # Skewness
    Skewness_hat_MDN = np.sum((((points_of_mass_MDN-Mu_hat_MDN)/Var_hat_MDN)**3)*b)
    Skewness_MC = np.mean((np.array(Y_test[i]-Mu_MC)/Var_MC)**3)
    if f_unknown_mode == "Heteroskedastic_NonLinear_Regression":
        Skewness = 0
    else:
        Skewness = Skewness_MC
    ### Error(s)
    Skewness_loop = np.abs(Skewness_hat_MDN-Skewness)
    
    # Skewness
    Ex_Kurtosis_hat_MDN = np.sum((((points_of_mass_MDN-Mu_hat_MDN)/Var_hat_MDN)**4)*b) - 3
    Ex_Kurtosis_MC = np.mean((np.array(Y_test[i]-Mu_MC)/Var_MC)**4) - 3
    if f_unknown_mode == "Heteroskedastic_NonLinear_Regression":
        Ex_Kurtosis = 3
    else:
        Ex_Kurtosis = Ex_Kurtosis_MC
    ### Error(s)
    Ex_Kurtosis_loop = np.abs(Ex_Kurtosis-Ex_Kurtosis_hat_MDN)
    
    
    # Update
    if i == 0:
        W1_Errors_MDN_test = W1_loop_MDN
        # Moments
        Mean_Errors_MDN_test =  Mean_loop
        Var_Errors_MDN_test = Var_loop
        Skewness_Errors_MDN_test = Skewness_loop
        Ex_Kurtosis_Errors_MDN_test = Ex_Kurtosis_loop
        
        
    else:
        W1_Errors_MDN_test = np.append(W1_Errors_MDN_test,W1_loop_MDN)
        # Moments
        Mean_Errors_MDN_test =  np.append(Mean_Errors_MDN_test,Mean_loop)
        Var_Errors_MDN_test = np.append(Var_Errors_MDN_test,Var_loop)
        Skewness_Errors_MDN_test = np.append(Skewness_Errors_MDN_test,Skewness_loop)
        Ex_Kurtosis_Errors_MDN_test = np.append(Ex_Kurtosis_Errors_MDN_test,Ex_Kurtosis_loop)
        
print("#---------------------#")
print(" Get Test Error(s): END")
print("#---------------------#")


# #### Stop Timer

# In[ ]:


Bishop_MDN_Timer = time.time() - Bishop_MDN_Timer


# # Get Performance Metric(s)

# ## Predictive Performance Metrics

# #### Train

# In[ ]:


print("#---------------------------#")
print(" Get Training Error(s): Begin")
print("#---------------------------#")
W1_Errors_MDN = np.array(bootstrap(np.abs(W1_Errors_MDN),n=N_Boostraps_BCA)(.95))
Mean_Errors_MDN = np.array(bootstrap(np.abs(Mean_Errors_MDN),n=N_Boostraps_BCA)(.95))
Var_Errors_MDN = np.array(bootstrap(np.abs(Var_Errors_MDN),n=N_Boostraps_BCA)(.95))
Skewness_Errors_MDN = np.array(bootstrap(np.abs(Skewness_Errors_MDN),n=N_Boostraps_BCA)(.95))
Ex_Kurtosis_Errors_MDN = np.array(bootstrap(np.abs(Ex_Kurtosis_Errors_MDN),n=N_Boostraps_BCA)(.95))

print("#-------------------------#")
print(" Get Training Error(s): END")
print("#-------------------------#")


# #### Test

# In[ ]:


print("#--------------------------#")
print(" Get Testing Error(s): Begin")
print("#--------------------------#")
W1_Errors_MDN_test = np.array(bootstrap(np.abs(W1_Errors_MDN_test),n=N_Boostraps_BCA)(.95))
Mean_Errors_MDN_test = np.array(bootstrap(np.abs(Mean_Errors_MDN_test),n=N_Boostraps_BCA)(.95))
Var_Errors_MDN_test = np.array(bootstrap(np.abs(Var_Errors_MDN_test),n=N_Boostraps_BCA)(.95))
Skewness_Errors_MDN_test = np.array(bootstrap(np.abs(Skewness_Errors_MDN_test),n=N_Boostraps_BCA)(.95))
Ex_Kurtosis_Errors_MDN_test = np.array(bootstrap(np.abs(Ex_Kurtosis_Errors_MDN_test),n=N_Boostraps_BCA)(.95))
print("#------------------------#")
print(" Get Testing Error(s): END")
print("#------------------------#")


# #### Compute MDN Complexities

# In[ ]:


# Tally MDN Complexities #
#------------------------#
## Tally N-Parameters
MDNs_Tot_N_Params = N_params_MDN_SDsNet + N_params_MDN_MixNet + N_params_MDN_MeansNet
## Tally Time
MDNs_Tot_time = timer_output_MDN_MixNet + timer_output_MDN_SDsNet + timer_output_MDN_MeansNet


# ### Update Prediction Quality Metrics

# In[ ]:


print("-------------------------------------------------")
print("Updating Performance Metrics Dataframe and Saved!")
print("-------------------------------------------------")
# Train
Summary_pred_Qual_models_internal["MDN"] = pd.Series(np.append(np.append(W1_Errors_MDN,
                                                                Mean_Errors_MDN),
                                                         np.array([MDNs_Tot_N_Params,
                                                                   MDNs_Tot_time,
                                                                   (Bishop_MDN_Timer/Test_Set_PredictionTime_MC)])), index=Summary_pred_Qual_models.index)
# Test
Summary_pred_Qual_models_test["MDN"] = pd.Series(np.append(np.append(W1_Errors_MDN_test,
                                                                     Mean_Errors_MDN_test),
                                                           np.array([MDNs_Tot_N_Params,
                                                                     MDNs_Tot_time,
                                                                     (Bishop_MDN_Timer/Test_Set_PredictionTime_MC)])), index=Summary_pred_Qual_models_test.index)

print("Updated DataFrame")
print("-------------------------------------------------")
print("Train")
print(Summary_pred_Qual_models)
print("-------------------------------------------------")
print("Test")
print(Summary_pred_Qual_models_test)
print("-------------------------------------------------")
Summary_pred_Qual_models_test

#----------------------#
# Save Quality Metrics #
#----------------------#
## Get Worst-Case
Summary_pred_Qual_models_train = Summary_pred_Qual_models_internal
Summary_pred_Qual_models_internal = np.maximum(Summary_pred_Qual_models_internal,Summary_pred_Qual_models_test)
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
Summary_pred_Qual_models
print("------------------------------------------------")
print("Updated Performance Metrics Dataframe and Saved!")
print("------------------------------------------------")


# In[ ]:


print("Have a jolly old day!")


# ---
# # Fin
# ---
