#!/usr/bin/env python
# coding: utf-8

# # Distributional Model(s)
# 
# **Note:** *NB, this means that this script *must* be run after the point-mass benchmarks script!*

# ---

# ### Gaussian Process Regressor

# In[ ]:


def get_GPR(X_train_in,X_test_in,y_means_in):
    # Initialize Cross-Vlidator of GPR
    CV_GPR = RandomizedSearchCV(estimator=GaussianProcessRegressor(),
                                n_jobs=n_jobs,
                                cv=KFold(2, random_state=2020, shuffle=True),
                                param_distributions=param_grid_GAUSSIAN,
                                n_iter=n_iter,
                                return_train_score=True,
                                random_state=2021,
                                verbose=10)

    CV_GPR.fit(X_train,Y_train_mean_emp)
    # Get Best Model
    best_GPR = CV_GPR.best_estimator_

    # Get Training-Set Prediction
    GPR_means = best_GPR.predict(X_train,return_std=True)[0]
    GPR_vars = (best_GPR.predict(X_train,return_std=True)[1])**2

    # Get Test-Set Predictions
    GPR_test_time_prediction = time.time()
    GPR_means_test = best_GPR.predict(X_test,return_std=True)[0]
    GPR_vars_test = (best_GPR.predict(X_test,return_std=True)[1])**2
    GPR_test_time_prediction = time.time() - GPR_test_time_prediction
    
    # Return Trained Predictions + Model
    GPR_means_test = best_GPR.predict(X_test,return_std=True)[0]
    return GPR_means,GPR_vars, GPR_means_test, GPR_vars_test, best_GPR, GPR_test_time_prediction


# # Universal Gaussian DNN

# Maps $\varrho:\mathbb{R}^d\ni \to (\hat{\mu},\sigma)\in \mathbb{R}\times (0,\infty)$.  
# 
# Implictly:
# $
# \rho:\mathbb{R}^d\ni \to \nu\circ \varrho(x)\in \mathcal{P}_2(\mathbb{R})
# .
# $
# 
# The universal approximation theorem for this architecture is given in [Corollary 7: Quantitative Rates and Fundamental Obstructions to Non-EuclideanUniversal Approximation with Deep Narrow Feed-Forward Networks](https://arxiv.org/pdf/2101.05390.pdf)

# In[ ]:


class Gaussian_Splitter(tf.keras.layers.Layer):

    def __init__(self):
        super(fullyConnected_Dense, self).__init__()
        self.units = units

    def call(self):
        return tf.math.pow(self,2)


# In[ ]:


def get_ffNN_Gaussian(height, depth, learning_rate, input_dim, output_dim):
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
    core_layers = fullyConnected_Dense(output_dim)(core_layers)  
    #------------------#
    #  Readout Layers  #
    #------------------# 
    # Gaussian Splitter Layer
    output_layers = Gaussian_Splitter(core_layers)
    # Define Input/Output Relationship (Arch.)
    trainable_layers_model = tf.keras.Model(input_layer, output_layers)
    
    
    #----------------------------------#
    # Define Optimizer & Compile Archs.
    #----------------------------------#
    opt = Adam(lr=learning_rate)
    trainable_layers_model.compile(optimizer=opt, loss="mae", metrics=["mse", "mae", "mape"])

    return trainable_layers_model



def build_ffNN_Gaussian(n_folds , n_jobs, n_iter, param_grid_in, X_train, y_train,X_test):
    # Update Dictionary
    param_grid_in_internal = param_grid_in
    param_grid_in_internal['input_dim'] = [(X_train.shape[1])]
    
    # Deep Feature Network
    ffNN_CV = tf.keras.wrappers.scikit_learn.KerasRegressor(build_fn=get_ffNN_Gaussian, 
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

# #### Gaussian Output Layer
# UAP-Preserving
# $\rho:\mathbb{R}^d\ni x \mapsto (x_1,\exp(x_2))\in \mathbb{R}\times (0,\infty)$; Thus, $\rho_{\star}[\mathcal{NN}_{d,2}^{\sigma}]$ is dense in $C(\mathbb{R}^d,\mathbb{R}\times [0,\infty))$ by [this paper's main result.](https://proceedings.neurips.cc/paper/2020/hash/786ab8c4d7ee758f80d57e65582e609d-Abstract.html).

# In[ ]:


# Affine Readout post-composed with UAP-preserving readout map to G_d
class Gaussian_Splitter(tf.keras.layers.Layer):

    def __init__(self, units=16, input_dim=32):
        super(Gaussian_Splitter, self).__init__()
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
        mean_and_cov = tf.concat([parameters,tf.math.exp(parameters)],-1)
        return mean_and_cov


# Implements the above deep network.

# In[ ]:


def get_ffNN_Gaussian(height, depth, learning_rate, input_dim, output_dim):
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
    output_layers = Gaussian_Splitter(output_dim)(core_layers)  
    # Define Input/Output Relationship (Arch.)
    trainable_layers_model = tf.keras.Model(input_layer, output_layers)
    
    
    #----------------------------------#
    # Define Optimizer & Compile Archs.
    #----------------------------------#
    opt = Adam(lr=learning_rate)
    trainable_layers_model.compile(optimizer=opt, loss="mae", metrics=["mse", "mae", "mape"])

    return trainable_layers_model



def build_ffNN_Gaussian(n_folds , n_jobs, n_iter, param_grid_in, X_train, y_train,X_test):
    # Update Dictionary
    param_grid_in_internal = param_grid_in
    param_grid_in_internal['input_dim'] = [(X_train.shape[1])]
    
    # Deep Feature Network
    ffNN_CV = tf.keras.wrappers.scikit_learn.KerasRegressor(build_fn=get_ffNN_Gaussian, 
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

# # Run Model(s)

# ### Train Gaussian Process Regression

# In[ ]:


GRP_time = time.time()
GPR_means, GPR_vars, GPR_means_test, GPR_vars_test, GPR_trash, GPR_test_time_prediction = get_GPR(X_train,X_test,Y_train_mean_emp) 
GRP_time = time.time() - GRP_time


# ### Train Deep Gaussian Network

# In[ ]:


print("===============================")
print("Training Deep Gaussian Network!")
print("===============================")

# Redefine (Dimension-related) Elements of Grid
param_grid_Deep_Classifier['input_dim'] = [problem_dim]
param_grid_Deep_Classifier['output_dim'] = [1]

# Train simple deep classifier
timer_DGP = time.time()
Deep_Gaussian_train_parameters, Deep_Gaussian_test_parameters, N_params_deep_Gaussian, timer_output_Deep_Gaussian = build_ffNN_Gaussian(n_folds = CV_folds, 
                                                                                                                                        n_jobs = n_jobs, 
                                                                                                                                        n_iter = n_iter, 
                                                                                                                                        param_grid_in=param_grid_Deep_Classifier, 
                                                                                                                                        X_train = X_train, 
                                                                                                                                        y_train = Y_train_var_emp,
                                                                                                                                        X_test = X_test)
# Format as float
Deep_Gaussian_train_parameters = np.array(Deep_Gaussian_train_parameters,dtype=float)
Deep_Gaussian_test_parameters = np.array(Deep_Gaussian_test_parameters,dtype=float)
timer_DGP = time.time() - timer_DGP
print("====================================")
print("Training Deep Gaussian Network!: END")
print("====================================")


# # Get Quality and Prediction Metrics
# ---

# #### Training Set

# In[ ]:


N_Bootstraps = N_Boostraps_BCA
print("#---------------------------------------#")
print(" Get Training Errors for: Gaussian Models")
print("#---------------------------------------#")
for i in tqdm(range((X_train.shape[0]))):    
    # Get Samples
    ## From: Deep Gaussian Network (DGN)
    hat_mu_DGaussianNet = Deep_Gaussian_train_parameters[i,][0]
    hat_sd_DGaussianNet = np.sqrt(Deep_Gaussian_train_parameters[i,][1])
    sample_DGaussianNet = np.random.normal(hat_mu_DGaussianNet,hat_sd_DGaussianNet,N_Monte_Carlo_Samples)
    ## From: Gaussian Process Regressor (GPR)
    hat_mu_GRP = GPR_means[i]
    hat_sd_GPR = np.sqrt(GPR_vars[i])
    sample_GRP = np.random.normal(hat_mu_GRP,hat_sd_GPR,N_Monte_Carlo_Samples)
    
    # Compute Error(s)
    ## W1
    ### DGN
    W1_loop_DGN = ot.emd2_1d(sample_DGaussianNet,
                             np.array(Y_train[i,]).reshape(-1,),
                             empirical_weights,
                             empirical_weights)
    ### GPR
    W1_loop_GPR = ot.emd2_1d(sample_GRP,
                             np.array(Y_train[i,]).reshape(-1,),
                             empirical_weights,
                             empirical_weights)
    
    ## M1
    Mu_MC = np.mean(np.array(Y_train[i,]))
    if f_unknown_mode == "Heteroskedastic_NonLinear_Regression":
        Mu = direct_facts[i,]
    else:
        Mu = Mu_MC
    ### Error(s)
    Mean_loop_DGN = (hat_mu_DGaussianNet-Mu)
    Mean_loop_GPR = (hat_mu_GRP-Mu_MC)
    
    ## Variance
    Var_loop_DGN = hat_sd_DGaussianNet**2
    Var_loop_GPR = hat_sd_GPR**2
    if f_unknown_mode == "Heteroskedastic_NonLinear_Regression":
        Var = 2*np.sum(X_train[i,]**2)
    else:
        Var_MC = np.mean(np.array(Y_train[i]-Mu_MC)**2)
        Var = Var_MC     
    ### Error(s)
    Var_loop_DGN = np.abs(Var_loop_DGN-Var)
    Var_loop_GPR = np.abs(Mean_loop_GPR-Var)
        
    # Skewness
    Skewness_DGN = 0
    Skewness_GPR = 0
    if f_unknown_mode == "Heteroskedastic_NonLinear_Regression":
        Skewness = 0
    else:
        Skewness_MC = np.mean((np.array(Y_train[i]-Mu_MC)/Var_MC)**3)
        Skewness = Skewness_MC
    ### Error(s)
    Skewness_loop_DGN = np.abs(Skewness_DGN-Skewness)
    Skewness_loop_GPR = np.abs(Skewness_GPR-Skewness)
    
    # Skewness
    Ex_Kurtosis_DGN = 0
    Ex_Kurtosis_GPR = 0
    if f_unknown_mode == "Heteroskedastic_NonLinear_Regression":
        Ex_Kurtosis = 3
    else:
        Ex_Kurtosis_MC = np.mean((np.array(Y_train[i]-Mu_MC)/Var_MC)**4) - 3
        Ex_Kurtosis = Ex_Kurtosis_MC
    ### Error(s)
    Ex_Kurtosis_loop_DGN = np.abs(Ex_Kurtosis-Ex_Kurtosis_DGN)
    Ex_Kurtosis_loop_GPR = np.abs(Ex_Kurtosis-Ex_Kurtosis_GPR)
    
    
    
    # Get Higher Moments Loss
#     Higher_momentserrors_loop_GPR,Higher_MC_momentserrors_loop_GPR = Higher_Moments_Loss(sample_GRP,np.array(Y_train[i,]))
#     Higher_momentserrors_loop_DGN,Higher_MC_momentserrors_loop_DGN = Higher_Moments_Loss(sample_DGaussianNet,np.array(Y_train[i,]))
    
    
    # Update
    if i == 0:
        W1_Errors_GPR = W1_loop_GPR
        W1_Errors_DGN = W1_loop_DGN
        # Moments
        ## GPR
        Mean_Errors_GPR =  Mean_loop_GPR
        Var_Errors_GPR = Var_loop_GPR
        Skewness_Errors_GPR = Skewness_loop_GPR
        Ex_Kurtosis_Errors_GPR = Ex_Kurtosis_loop_GPR
#         Higher_Moments_Errors_GPR = Higher_momentserrors_loop_GPR
        ## DGN
        Mean_Errors_DGN =  Mean_loop_DGN
        Var_Errors_DGN = Var_loop_DGN
        Skewness_Errors_DGN = Skewness_loop_DGN
        Ex_Kurtosis_Errors_DGN = Ex_Kurtosis_loop_DGN
#         Higher_Moments_Errors_DGN = Higher_momentserrors_loop_DGN
        
        
    else:
        W1_Errors_GPR = np.append(W1_Errors_GPR,W1_loop_GPR)
        W1_Errors_DGN = np.append(W1_Errors_DGN,W1_loop_DGN)
        # Moments
        ## GPR
        Mean_Errors_GPR =  np.append(Mean_Errors_GPR,Mean_loop_GPR)
        Var_Errors_GPR = np.append(Var_Errors_GPR,Var_loop_GPR)
        Skewness_Errors_GPR = np.append(Skewness_Errors_GPR,Skewness_loop_GPR)
        Ex_Kurtosis_Errors_GPR = np.append(Ex_Kurtosis_Errors_GPR,Ex_Kurtosis_loop_GPR)
#         Higher_Moments_Errors_GPR = np.append(Higher_Moments_Errors_GPR,Higher_momentserrors_loop_GPR)
        ## DGN
        Mean_Errors_DGN =  np.append(Mean_Errors_DGN,Mean_loop_DGN)
        Var_Errors_DGN = np.append(Var_Errors_DGN,Var_loop_DGN)
        Skewness_Errors_DGN = np.append(Skewness_Errors_DGN,Skewness_loop_DGN)
        Ex_Kurtosis_Errors_DGN = np.append(Ex_Kurtosis_Errors_DGN,Ex_Kurtosis_loop_DGN)
#         Higher_Moments_Errors_DGN = np.append(Higher_Moments_Errors_DGN,Higher_momentserrors_loop_DGN)
        
    
# Compute Error Metrics with Bootstrapped Confidence Intervals
W1_Errors_GPR = np.array(bootstrap(np.abs(W1_Errors_GPR),n=N_Bootstraps)(.95))
W1_Errors_DGN = np.array(bootstrap(np.abs(W1_Errors_DGN),n=N_Bootstraps)(.95))
Mean_Errors_GPR = np.array(bootstrap(np.abs(Mean_Errors_GPR),n=N_Bootstraps)(.95))
Mean_Errors_DGN = np.array(bootstrap(np.abs(Mean_Errors_DGN),n=N_Bootstraps)(.95))
Var_Errors_GPR = np.array(bootstrap(np.abs(Var_Errors_GPR),n=N_Bootstraps)(.95))
Var_Errors_DGN = np.array(bootstrap(np.abs(Var_Errors_DGN),n=N_Bootstraps)(.95))
Skewness_Errors_GPR = np.array(bootstrap(np.abs(Skewness_Errors_GPR),n=N_Bootstraps)(.95))
Skewness_Errors_DGN = np.array(bootstrap(np.abs(Skewness_Errors_DGN),n=N_Bootstraps)(.95))
Ex_Kurtosis_Errors_GPR = np.array(bootstrap(np.abs(Ex_Kurtosis_Errors_GPR),n=N_Bootstraps)(.95))
Ex_Kurtosis_Errors_DGN = np.array(bootstrap(np.abs(Ex_Kurtosis_Errors_DGN),n=N_Bootstraps)(.95))
#     Higher_Moment_Errors = np.array(bootstrap(np.abs(Higher_Moments_Errors),n=N_Bootstraps)(.95))

# Format Error Metrics
Summary_pred_Qual_models_DGN_train = np.array([W1_Errors_DGN,
                                               Mean_Errors_DGN,
                                               Var_Errors_DGN,
                                               Skewness_Errors_DGN,
                                               Ex_Kurtosis_Errors_DGN])
Summary_pred_Qual_models_GPR_train = np.array([W1_Errors_GPR,
                                               Mean_Errors_GPR,
                                               Var_Errors_GPR,
                                               Skewness_Errors_GPR,
                                               Ex_Kurtosis_Errors_GPR])
print("#-------------------------#")
print(" Get Training Error(s): END")
print("#-------------------------#")


# #### Testing Set

# In[ ]:


N_Bootstraps = N_Boostraps_BCA
print("#---------------------------------------#")
print(" Get Testing Errors for: Gaussian Models")
print("#---------------------------------------#")
for i in tqdm(range((X_test.shape[0]))):    
    # Get Samples
    ## From: Deep Gaussian Network (DGN)
    hat_mu_DGaussianNet_test = Deep_Gaussian_test_parameters[i,][0]
    hat_sd_DGaussianNet_test = np.sqrt(Deep_Gaussian_test_parameters[i,][1])
    sample_DGaussianNet_test = np.random.normal(hat_mu_DGaussianNet_test,hat_sd_DGaussianNet_test,N_Monte_Carlo_Samples)
    ## From: Gaussian Process Regressor (GPR)
    hat_mu_GRP_test = GPR_means_test[i]
    hat_sd_GPR_test = np.sqrt(GPR_vars_test[i])
    sample_GRP_test = np.random.normal(hat_mu_GRP_test,hat_sd_GPR_test,N_Monte_Carlo_Samples)
    
    # Compute Error(s)
    ## W1
    ### DGN
    W1_loop_DGN_test = ot.emd2_1d(sample_DGaussianNet_test,
                             np.array(Y_test[i,]).reshape(-1,),
                             empirical_weights,
                             empirical_weights)
    ### GPR
    W1_loop_GPR_test = ot.emd2_1d(sample_GRP_test,
                             np.array(Y_test[i,]).reshape(-1,),
                             empirical_weights,
                             empirical_weights)
    
    ## M1
    Mu_MC_test = np.mean(np.array(Y_test[i,]))
    if f_unknown_mode == "Heteroskedastic_NonLinear_Regression":
        Mu_test = direct_facts_test[i,]
    else:
        Mu_test = Mu_MC_test
    ### Error(s)
    Mean_loop_DGN_test = (hat_mu_DGaussianNet_test-Mu_test)
    Mean_loop_GPR_test = (hat_mu_GRP_test-Mu_MC_test)
    
    ## Variance
    Var_loop_DGN_test = hat_sd_DGaussianNet_test**2
    Var_loop_GPR_test = hat_sd_GPR_test**2
    if f_unknown_mode == "Heteroskedastic_NonLinear_Regression":
        Var_test = 2*np.sum(X_test[i,]**2)
    else:
        Var_MC_test = np.mean(np.array(Y_test[i]-Mu_MC_test)**2)
        Var_test = Var_MC_test
    ### Error(s)
    Var_loop_DGN_test = np.abs(Var_loop_DGN_test-Var_test)
    Var_loop_GPR_test = np.abs(Mean_loop_GPR_test-Var_test)
        
    # Skewness
    Skewness_DGN_test = 0
    Skewness_GPR_test = 0
    if f_unknown_mode == "Heteroskedastic_NonLinear_Regression":
        Skewness_test = 0
    else:
        Skewness_MC_test = np.mean((np.array(Y_test[i]-Mu_MC_test)/Var_MC_test)**3)
        Skewness_test = Skewness_MC_test
    ### Error(s)
    Skewness_loop_DGN = np.abs(Skewness_DGN-Skewness)
    Skewness_loop_GPR = np.abs(Skewness_GPR-Skewness)
    
    # Skewness
    Ex_Kurtosis_DGN_test = 0
    Ex_Kurtosis_GPR_test = 0
    if f_unknown_mode == "Heteroskedastic_NonLinear_Regression":
        Ex_Kurtosis_test = 3
    else:
        Ex_Kurtosis_MC_test = np.mean((np.array(Y_test[i]-Mu_MC)/Var_MC_test)**4) - 3
        Ex_Kurtosis_test = Ex_Kurtosis_MC_test
    ### Error(s)
    Ex_Kurtosis_loop_DGN_test = np.abs(Ex_Kurtosis_test-Ex_Kurtosis_DGN_test)
    Ex_Kurtosis_loop_GPR_test = np.abs(Ex_Kurtosis_test-Ex_Kurtosis_GPR_test)
    
    
    
    # Get Higher Moments Loss
#     Higher_momentserrors_loop_GPR,Higher_MC_momentserrors_loop_GPR = Higher_Moments_Loss(sample_GRP,np.array(Y_train[i,]))
#     Higher_momentserrors_loop_DGN,Higher_MC_momentserrors_loop_DGN = Higher_Moments_Loss(sample_DGaussianNet,np.array(Y_train[i,]))
    
    
    # Update
    if i == 0:
        W1_Errors_GPR_test = W1_loop_GPR_test
        W1_Errors_DGN_test = W1_loop_DGN_test
        # Moments
        ## GPR
        Mean_Errors_GPR_test =  Mean_loop_GPR_test
        Var_Errors_GPR_test = Var_loop_GPR_test
        Skewness_Errors_GPR_test = Skewness_GPR_test
        Ex_Kurtosis_Errors_GPR_test = Ex_Kurtosis_loop_GPR_test
#         Higher_Moments_Errors_GPR = Higher_momentserrors_loop_GPR
        ## DGN
        Mean_Errors_DGN_test =  Mean_loop_DGN_test
        Var_Errors_DGN_test = Var_loop_DGN_test
        Skewness_Errors_DGN_test = Skewness_DGN_test
        Ex_Kurtosis_Errors_DGN_test = Ex_Kurtosis_loop_DGN_test
#         Higher_Moments_Errors_DGN = Higher_momentserrors_loop_DGN
        
        
    else:
        W1_Errors_GPR_test = np.append(W1_Errors_GPR_test,W1_loop_GPR_test)
        W1_Errors_DGN_test = np.append(W1_Errors_DGN_test,W1_loop_DGN_test)
        # Moments
        ## GPR
        Mean_Errors_GPR_test =  np.append(Mean_Errors_GPR_test,Mean_loop_GPR_test)
        Var_Errors_GPR_test = np.append(Var_Errors_GPR_test,Var_loop_GPR_test)
        Skewness_Errors_GPR_test = np.append(Skewness_Errors_GPR_test,Skewness_GPR_test)
        Ex_Kurtosis_Errors_GPR_test = np.append(Ex_Kurtosis_Errors_GPR_test,Ex_Kurtosis_loop_GPR_test)
#         Higher_Moments_Errors_GPR = np.append(Higher_Moments_Errors_GPR,Higher_momentserrors_loop_GPR)
        ## DGN
        Mean_Errors_DGN_test =  np.append(Mean_Errors_DGN_test,Mean_loop_DGN_test)
        Var_Errors_DGN_test = np.append(Var_Errors_DGN_test,Var_loop_DGN_test)
        Skewness_Errors_DGN_test = np.append(Skewness_Errors_DGN_test,Skewness_DGN_test)
        Ex_Kurtosis_Errors_DGN_test = np.append(Ex_Kurtosis_Errors_DGN_test,Ex_Kurtosis_loop_DGN_test)
#         Higher_Moments_Errors_DGN = np.append(Higher_Moments_Errors_DGN,Higher_momentserrors_loop_DGN)
        
    
# Compute Error Metrics with Bootstrapped Confidence Intervals
W1_Errors_GPR_test = np.array(bootstrap(np.abs(W1_Errors_GPR_test),n=N_Bootstraps)(.95))
W1_Errors_DGN_test = np.array(bootstrap(np.abs(W1_Errors_DGN_test),n=N_Bootstraps)(.95))
Mean_Errors_GPR_test = np.array(bootstrap(np.abs(Mean_Errors_GPR_test),n=N_Bootstraps)(.95))
Mean_Errors_DGN_test = np.array(bootstrap(np.abs(Mean_Errors_DGN_test),n=N_Bootstraps)(.95))
Var_Errors_GPR_test = np.array(bootstrap(np.abs(Var_Errors_GPR_test),n=N_Bootstraps)(.95))
Var_Errors_DGN_test = np.array(bootstrap(np.abs(Var_Errors_DGN_test),n=N_Bootstraps)(.95))
Skewness_Errors_GPR_test = np.array(bootstrap(np.abs(Skewness_Errors_GPR_test),n=N_Bootstraps)(.95))
Skewness_Errors_DGN_test = np.array(bootstrap(np.abs(Skewness_Errors_DGN_test),n=N_Bootstraps)(.95))
Ex_Kurtosis_Errors_GPR_test = np.array(bootstrap(np.abs(Ex_Kurtosis_Errors_GPR_test),n=N_Bootstraps)(.95))
Ex_Kurtosis_Errors_DGN_test = np.array(bootstrap(np.abs(Ex_Kurtosis_Errors_DGN_test),n=N_Bootstraps)(.95))
#     Higher_Moment_Errors = np.array(bootstrap(np.abs(Higher_Moments_Errors),n=N_Bootstraps)(.95))

# Format Error Metrics
Summary_pred_Qual_models_DGN_test = np.array([W1_Errors_DGN_test,
                                              Mean_Errors_DGN_test,
                                              Var_Errors_DGN_test,
                                              Skewness_Errors_DGN_test,
                                              Ex_Kurtosis_Errors_DGN_test])
Summary_pred_Qual_models_GPR_test = np.array([W1_Errors_GPR_test,
                                              Mean_Errors_GPR_test,
                                              Var_Errors_GPR_test,
                                              Skewness_Errors_GPR_test,
                                              Ex_Kurtosis_Errors_GPR_test])
print("#------------------------#")
print(" Get Testing Error(s): END")
print("#------------------------#")


# # Update Performance Metrics:
# NB, this means that this script *must* be run after the point-mass benchmarks script!
# 
# ## Update Prediction-Quality Metrics

# In[ ]:


print("-------------------------------------------------")
print("Updating Performance Metrics Dataframe and Saved!")
print("-------------------------------------------------")
# Append Gaussian Process Regressor Performance
## Train
Summary_pred_Qual_models["GPR"] = pd.Series((Summary_pred_Qual_models_GPR_train[:,1]), index=Summary_pred_Qual_models.index)
## Test
Summary_pred_Qual_models_test["GPR"] = pd.Series((Summary_pred_Qual_models_GPR_test[:,1]), index=Summary_pred_Qual_models_test.index)

# Append Deep Gaussian Network Performance
## Train
Summary_pred_Qual_models["DGN"] = pd.Series((Summary_pred_Qual_models_DGN_train[:,1]), index=Summary_pred_Qual_models.index)
## Test
Summary_pred_Qual_models_test["DGN"] = pd.Series((Summary_pred_Qual_models_DGN_test[:,1]), index=Summary_pred_Qual_models_test.index)

# Update Performance Metrics
## Train
Summary_pred_Qual_models.to_latex((results_tables_path+str(f_unknown_mode)+"Problemdimension"+str(problem_dim)+"__SUMMARY_METRICS.tex"))
## Test
Summary_pred_Qual_models_test.to_latex((results_tables_path+str(f_unknown_mode)+"Problemdimension"+str(problem_dim)+"__SUMMARY_METRICS_test.tex"))

print("------------------------------------------------")
print("Updated Performance Metrics Dataframe and Saved!")
print("------------------------------------------------")


# ## Update Model Complexity Metrics

# ### Compute Performance Metrics for GPR and DGN Model(s)

# In[1]:


print("--------------------------------------------")
print("Computing and Updating Complexity Metrics...")
print("--------------------------------------------")
# Coercion
Summary_Complexity_models = Summary_Complexity_models.T
# Compute Complexity Metrics for GPR
GPR_Facts = np.array([0,GRP_time,GPR_test_time_prediction/Test_Set_PredictionTime_MC])
DGN_Facts = np.array([N_params_deep_Gaussian,timer_DGP,timer_output_Deep_Gaussian/Test_Set_PredictionTime_MC])
# Update Model Complexities
Summary_Complexity_models["GPR"] = pd.Series(GPR_Facts, index=Summary_Complexity_models.index)
Summary_Complexity_models["DGN"] = pd.Series(DGN_Facts, index=Summary_Complexity_models.index)
# Coercion
Summary_Complexity_models = Summary_Complexity_models.T

# Save Facts
Summary_Complexity_models.to_latex((results_tables_path+str(f_unknown_mode)+"Problemdimension"+str(problem_dim)+"__Complexity_Metrics.tex"))
print("-----------------------------------------------")
print("Updated Complexity Metrics Dataframe and Saved!")
print("-----------------------------------------------")


# ---
