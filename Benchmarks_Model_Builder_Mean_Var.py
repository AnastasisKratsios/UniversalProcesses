#!/usr/bin/env python
# coding: utf-8

# # Distributional Model(s)
# 
# **Note:** *NB, this means that this script *must* be run after the point-mass benchmarks script!*

# In[4]:


# ##### DEGUBBING MENU:
# trial_run = True
# N_train_size= 20
# train_test_ratio = .5
# N_Monte_Carlo_Samples = 10**3
# # # Random DNN
# # f_unknown_mode = "Heteroskedastic_NonLinear_Regression"

# # # Random DNN internal noise
# f_unknown_mode = "DNN_with_Random_Weights"
# Depth_Bayesian_DNN = 2
# width = 50

# # # Random Dropout applied to trained DNN
# # f_unknown_mode = "DNN_with_Bayesian_Dropout"
# Dropout_rate = 0.1

# # GD with Randomized Input
# # f_unknown_mode = "GD_with_randomized_input"
# GD_epochs = 100

# # SDE with fractional Driver
# # f_unknown_mode = "Rough_SDE"
# N_Euler_Steps = 10**1
# Hurst_Exponent = 0.5
# problem_dim = 3

# # Hyper-parameters of Cover
# delta = 0.01
# Proportion_per_cluster = .75

# # %run Loader.ipynb
# exec(open('Loader.py').read())
# # Load Packages/Modules
# exec(open('Init_Dump.py').read())
# import time as time #<- Note sure why...but its always seems to need 'its own special loading...'

# # %run Data_Simulator_and_Parser.ipynb
# exec(open('Data_Simulator_and_Parser.py').read())

# print("------------------------------")
# print("Running script for main model!")
# print("------------------------------")
# # %run Universal_Measure_Valued_Networks_Backend.ipynb
# exec(open('Universal_Measure_Valued_Networks_Backend.py').read())

# print("------------------------------------")
# print("Done: Running script for main model!")
# print("------------------------------------")


# ---

# ### Gaussian Process Regressor

# In[5]:


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

# In[6]:


if output_dim == 1:
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
else:
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


# ### Without Cholesky

# In[7]:


if output_dim == 1:
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


# ### With Cholesky

# In[8]:


if output_dim > 1:
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
    print('DNN Builder - Ready')


# ---

# ---

# # Run Model(s)

# ### Train Gaussian Process Regression

# In[9]:


GRP_time = time.time()
GPR_means, GPR_vars, GPR_means_test, GPR_vars_test, GPR_trash, GPR_test_time_prediction = get_GPR(X_train,
                                                                                                  X_test,
                                                                                                  Y_train_mean_emp) 
GRP_time = time.time() - GRP_time


# ### Train Deep Gaussian Network

# #### Infer Parameters to train on *(for training-set)* for deep Gaussian Network

# In[10]:


# Initializations #
#-----------------#
print("Infering Parameters for Deep Gaussian Network to train on!")
# Start timer:
timeBuilding_Training_Set_DGN = time.time()
# Set Gaussian Dimension
dim_Gaussian_space = output_dim*(1+output_dim)


# Get Optimized Parameters to train Deep Gaussian Network On
for i in tqdm(range(X_train.shape[0])):
    # Define Function Defining log-likelihood of Gaussian dist.
    if problem_dim == 1:
        ## Count Data-set (outputed-samples) size
        n = Y_train.shape[1]
        ## Dummy Initialized Parameters
        initParams = [1, 1]
        def gaussian_log_like(parameters_in):
            mean = parameters_in[0]   
            sigma = parameters_in[1]

            # Calculate negative log likelihood
            negative_log_likelihood = -np.sum(stats.norm.logpdf(Y_train[i,], loc=mean, scale=sigma))
            return negative_log_likelihood
        
        # Search for MAE Gaussian Parameters
        results_loop = ((minimize(gaussian_log_like, initParams, method='Nelder-Mead')).x).reshape(1,-1)
    
    else:
        # Get Sample Means
        mean_loop = np.mean(Y_train[i,],axis=0)
        # Get (regularized Cholesky) Squareroot of Sample Covariance
        cov_loop = np.tril(np.linalg.cholesky(np.cov(Y_train[i,].T)+(10**-6)*np.diag(np.ones(output_dim)))).reshape(-1,)
        
        # Coercion
        results_loop = np.append(mean_loop,cov_loop).reshape(-1,dim_Gaussian_space)
    # Update Targets #
    #----------------#
    if i == 0:
        Y_train_var_emp = results_loop
    else:
        Y_train_var_emp = np.append(Y_train_var_emp,results_loop,axis=0)
# Stop timer:
time.Building_Training_Set_DGN = time.time() - timeBuilding_Training_Set_DGN
print("Done Getting Parameters for Deep Gaussian Network!")


# #### Train Deep Network on Infered Parameters

# In[11]:


print("===============================")
print("Training Deep Gaussian Network!")
print("===============================")
# Train simple deep classifier
timer_DGN = time.time()
if output_dim == 1:
    # Redefine (Dimension-related) Elements of Grid
    param_grid_Deep_Classifier['input_dim'] = [problem_dim]
    param_grid_Deep_Classifier['output_dim'] = [output_dim]
    Deep_Gaussian_train_parameters, Deep_Gaussian_test_parameters, N_params_deep_Gaussian, timer_output_Deep_Gaussian = build_ffNN_Gaussian(n_folds = CV_folds, 
                                                                                                                                            n_jobs = n_jobs, 
                                                                                                                                            n_iter = n_iter, 
                                                                                                                                            param_grid_in=param_grid_Deep_Classifier, 
                                                                                                                                            X_train = X_train, 
                                                                                                                                            y_train = Y_train_var_emp,
                                                                                                                                            X_test = X_test)
else:
    # Redefine (Dimension-related) Elements of Grid
    param_grid_Deep_Classifier['input_dim'] = [problem_dim]
    param_grid_Deep_Classifier['output_dim'] = [Y_train_var_emp.shape[1]]
    Deep_Gaussian_train_parameters, Deep_Gaussian_test_parameters, N_params_deep_Gaussian, timer_output_Deep_Gaussian = build_ffNN(n_folds = CV_folds, 
                                                                                                                                   n_jobs = n_jobs, 
                                                                                                                                   n_iter = n_iter, 
                                                                                                                                   param_grid_in=param_grid_Deep_Classifier, 
                                                                                                                                   X_train = X_train, 
                                                                                                                                   y_train = Y_train_var_emp,
                                                                                                                                   X_test = X_test)
# Format as float
Deep_Gaussian_train_parameters = np.array(Deep_Gaussian_train_parameters,dtype=float)
Deep_Gaussian_test_parameters = np.array(Deep_Gaussian_test_parameters,dtype=float)
timer_DGN = time.time() - timer_DGN
print("====================================")
print("Training Deep Gaussian Network!: END")
print("====================================")


# # Get Quality and Prediction Metrics
# ---

# #### Training Set

# In[12]:


N_Bootstraps = N_Boostraps_BCA
print("#---------------------------------------#")
print(" Get Training Errors for: Gaussian Models")
print("#---------------------------------------#")
for i in tqdm(range((X_train.shape[0]))):        
    if output_dim == 1:
        # Get Samples
        ## From: Deep Gaussian Network (DGN)
        hat_mu_DGaussianNet = Deep_Gaussian_train_parameters[i][0]
        hat_sd_DGaussianNet = np.sqrt(Deep_Gaussian_train_parameters[i][1])
        sample_DGaussianNet = np.random.normal(hat_mu_DGaussianNet,
                                               hat_sd_DGaussianNet,
                                               N_Monte_Carlo_Samples)

        ## From: Gaussian Process Regressor (GPR)
        hat_mu_GRP = GPR_means[i]
        hat_sd_GPR = np.sqrt(GPR_vars[i])
        sample_GRP = np.random.normal(hat_mu_GRP,
                                      hat_sd_GPR,
                                      N_Monte_Carlo_Samples)

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
        Mean_loop_DGN = np.sum(np.abs(hat_mu_DGaussianNet-Mu))
        Mean_loop_GPR = np.sum(np.abs(hat_mu_GRP-Mu_MC))
    else:
        ## From: Gaussian Process Regressor (GPR)
        hat_mu_GRP = GPR_means[i,]
        hat_sd_GPR = np.sqrt(GPR_vars[i])
        sample_GRP = np.random.multivariate_normal(hat_mu_GRP,
                                                   hat_sd_GPR*np.diag(np.ones(output_dim)),
                                                   N_Monte_Carlo_Samples)
        ## Get Multivariate Gaussian Process Regressor's Prediction
        sample_GRP = np.random.multivariate_normal(GPR_means[i,],
                                                   np.diag(np.repeat(GPR_vars[i],problem_dim)),
                                                   N_Monte_Carlo_Samples)
        ## Get Multivariate deep Gaussian Network's prediction
        # Extract Prediction(s)
        ## Get Mean
        mean_loop = Deep_Gaussian_train_parameters[0,:problem_dim]
        ## Get Covariance for Predicted Cholesky Root
        cov_sqrt_chol_loop = Deep_Gaussian_train_parameters[0,problem_dim:]
        cov_sqrt_chol_loop = cov_sqrt_chol_loop.reshape(output_dim,output_dim)
        cov_sqrt_chol_loop = (np.matmul(cov_sqrt_chol_loop,cov_sqrt_chol_loop.T))
        ## Get Empirical Samples
        sample_DGaussianNet = np.random.multivariate_normal(mean_loop,
                                                            cov_sqrt_chol_loop,
                                                            N_Monte_Carlo_Samples)
        
        ## W1
        W1_loop_GPR = ot.sliced.sliced_wasserstein_distance(X_s = sample_GRP, 
                                                            X_t = Y_train[i,],
                                                            seed = 2020)
        W1_loop_DGN = ot.sliced.sliced_wasserstein_distance(X_s = sample_DGaussianNet, 
                                                            X_t = Y_train[i,],
                                                            seed = 2020)
        
        ## M1
        Mu_MC = np.mean(np.array(Y_train[i,]))
        Mu = Mu_MC
        Mean_loop_GPR = np.sum(np.abs(hat_mu_GRP-Mu_MC))
        Mean_loop_DGN = np.sum(np.abs(cov_sqrt_chol_loop-Mu_MC))
   
    
    # Update
    if i == 0:
        W1_Errors_GPR = W1_loop_GPR
        W1_Errors_DGN = W1_loop_DGN
        # Moments
        ## GPR
        Mean_Errors_GPR =  Mean_loop_GPR
        ## DGN
        Mean_Errors_DGN =  Mean_loop_DGN
        
        
    else:
        W1_Errors_GPR = np.append(W1_Errors_GPR,W1_loop_GPR)
        W1_Errors_DGN = np.append(W1_Errors_DGN,W1_loop_DGN)
        # Moments
        ## GPR
        Mean_Errors_GPR =  np.append(Mean_Errors_GPR,Mean_loop_GPR)
        ## DGN
        Mean_Errors_DGN =  np.append(Mean_Errors_DGN,Mean_loop_DGN)
        
    
# Compute Error Metrics with Bootstrapped Confidence Intervals
W1_Errors_GPR = np.array(bootstrap(np.abs(W1_Errors_GPR),n=N_Bootstraps)(.95))
W1_Errors_DGN = np.array(bootstrap(np.abs(W1_Errors_DGN),n=N_Bootstraps)(.95))
M1_Errors_GPR = np.array(bootstrap(np.abs(Mean_Errors_GPR),n=N_Bootstraps)(.95))
M1_Errors_DGN = np.array(bootstrap(np.abs(Mean_Errors_DGN),n=N_Bootstraps)(.95))

print("#-------------------------#")
print(" Get Training Error(s): END")
print("#-------------------------#")


# #### Testing Set

# In[13]:


N_Bootstraps = N_Boostraps_BCA
print("#--------------------------------------#")
print(" Get Testing Errors for: Gaussian Models")
print("#--------------------------------------#")
for i in tqdm(range((X_test.shape[0]))):        
    if output_dim == 1:
        # Get Samples
        ## From: Deep Gaussian Network (DGN)
        hat_mu_DGaussianNet_test = Deep_Gaussian_test_parameters[i,][0]
        hat_sd_DGaussianNet_test = np.sqrt(Deep_Gaussian_test_parameters[i,][1])
        sample_DGaussianNet_test = np.random.normal(hat_mu_DGaussianNet_test,
                                               hat_sd_DGaussianNet_test,
                                               N_Monte_Carlo_Samples)

        ## From: Gaussian Process Regressor (GPR)
        hat_mu_GRP_test = GPR_means_test[i]
        hat_sd_GPR_test = np.sqrt(GPR_vars_test[i])
        sample_GRP_test = np.random.normal(hat_mu_GRP_test,
                                      hat_sd_GPR_test,
                                      N_Monte_Carlo_Samples)

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
        Mean_loop_DGN_test = np.sum(np.abs(hat_mu_DGaussianNet_test-Mu_test))
        Mean_loop_GPR_test = np.sum(np.abs(hat_mu_GRP_test-Mu_test))
    else:
        ## Get Multivariate Gaussian Process Regressor's Prediction
        hat_mu_GRP_test = GPR_means_test[i,]
        hat_sd_GPR_test = np.sqrt(GPR_vars_test[i])
        sample_GRP_test = np.random.multivariate_normal(GPR_means_test[i,],
                                                   np.diag(np.repeat(GPR_vars_test[i],problem_dim)),
                                                   N_Monte_Carlo_Samples)
        ## Get Multivariate deep Gaussian Network's prediction
        # Extract Prediction(s)
        ## Get Mean
        mean_loop = Deep_Gaussian_test_parameters[0,:problem_dim]
        ## Get Covariance for Predicted Cholesky Root
        cov_sqrt_chol_loop = Deep_Gaussian_test_parameters[0,problem_dim:]
        cov_sqrt_chol_loop = cov_sqrt_chol_loop.reshape(output_dim,output_dim)
        cov_sqrt_chol_loop = (np.matmul(cov_sqrt_chol_loop,cov_sqrt_chol_loop.T))
        ## Get Empirical Samples
        sample_DGaussianNet_test = np.random.multivariate_normal(mean_loop,
                                                                 cov_sqrt_chol_loop,
                                                                 N_Monte_Carlo_Samples)
        
        ## W1
        W1_loop_GPR_test = ot.sliced.sliced_wasserstein_distance(X_s = sample_GRP_test, 
                                                                 X_t = Y_test[i,],
                                                                 seed = 2020)
        W1_loop_DGN_test = ot.sliced.sliced_wasserstein_distance(X_s = sample_DGaussianNet_test, 
                                                                 X_t = Y_test[i,],
                                                                 seed = 2020)
        
        ## M1
        Mu_MC_test = np.mean(np.array(Y_test[i,]))
        Mu_test = Mu_MC_test
        Mean_loop_GPR_test = np.sum(np.abs(hat_mu_GRP_test-Mu_MC_test))
        Mean_loop_DGN_test = np.sum(np.abs(cov_sqrt_chol_loop-Mu_MC_test))
   
    
    # Update
    if i == 0:
        W1_Errors_GPR_test = W1_loop_GPR_test
        W1_Errors_DGN_test = W1_loop_DGN_test
        # Moments
        ## GPR
        Mean_Errors_GPR_test =  Mean_loop_GPR_test
        ## DGN
        Mean_Errors_DGN_test =  Mean_loop_DGN_test
        
        
    else:
        W1_Errors_GPR_test = np.append(W1_Errors_GPR_test,
                                       W1_loop_GPR_test)
        W1_Errors_DGN_test = np.append(W1_Errors_DGN_test,
                                       W1_loop_DGN_test)
        # Moments
        ## GPR
        Mean_Errors_GPR_test =  np.append(Mean_Errors_GPR_test,Mean_loop_GPR_test)
        ## DGN
        Mean_Errors_DGN_test =  np.append(Mean_Errors_DGN_test,Mean_loop_DGN_test)
        
    
# Compute Error Metrics with Bootstrapped Confidence Intervals
W1_Errors_GPR_test = np.array(bootstrap(np.abs(W1_Errors_GPR_test),n=N_Bootstraps)(.95))
W1_Errors_DGN_test = np.array(bootstrap(np.abs(W1_Errors_DGN_test),n=N_Bootstraps)(.95))
M1_Errors_GPR_test = np.array(bootstrap(np.abs(Mean_Errors_GPR_test),n=N_Bootstraps)(.95))
M1_Errors_DGN_test = np.array(bootstrap(np.abs(Mean_Errors_DGN_test),n=N_Bootstraps)(.95))

print("#-------------------------#")
print(" Get Training Error(s): END")
print("#-------------------------#")


# # Update Performance Metrics:
# NB, this means that this script *must* be run after the point-mass benchmarks script!
# 
# ## Update Prediction-Quality Metrics

# In[14]:


print("-------------------------------------------------")
print("Updating Performance Metrics Dataframe and Saved!")
print("-------------------------------------------------")
# Append Gaussian Process Regressor Performance
# Train
Summary_pred_Qual_models["GPR"] = pd.Series(np.append(np.append(W1_Errors_GPR,
                                                                M1_Errors_GPR),
                                                         np.array([0,
                                                                   GRP_time,
                                                                   (GPR_test_time_prediction/Test_Set_PredictionTime_MC)])), index=Summary_pred_Qual_models.index)
## Test
Summary_pred_Qual_models_test["GPR"] = pd.Series(np.append(np.append(W1_Errors_GPR_test,
                                                                M1_Errors_GPR_test),
                                                         np.array([0,
                                                                   GRP_time,
                                                                   (GPR_test_time_prediction/Test_Set_PredictionTime_MC)])), index=Summary_pred_Qual_models_test.index)
# Append Deep Gaussian Network Performance
Summary_pred_Qual_models["DGN"] = pd.Series(np.append(np.append(W1_Errors_DGN,
                                                                M1_Errors_DGN),
                                                      np.array([N_params_deep_Gaussian,
                                                                timer_DGN,
                                                                (timer_output_Deep_Gaussian/Test_Set_PredictionTime_MC)])), index=Summary_pred_Qual_models.index)
## Test
Summary_pred_Qual_models_test["DGN"] = pd.Series(np.append(np.append(W1_Errors_DGN_test,
                                                                     M1_Errors_DGN_test),
                                                           np.array([N_params_deep_Gaussian,
                                                                     timer_DGN,
                                                                     (timer_output_Deep_Gaussian/Test_Set_PredictionTime_MC)])), index=Summary_pred_Qual_models_test.index)
# Update Performance Metrics
## Train
Summary_pred_Qual_models.to_latex((results_tables_path+str(f_unknown_mode)+"Problemdimension"+str(problem_dim)+"__SUMMARY_METRICS.tex"))
print("Training Results to date:")
print(Summary_pred_Qual_models_test)
## Test
Summary_pred_Qual_models_test.to_latex((results_tables_path+str(f_unknown_mode)+"Problemdimension"+str(problem_dim)+"__SUMMARY_METRICS_test.tex"))
print("Test Results to date:")
print(Summary_pred_Qual_models_test)
print("------------------------------------------------")
print("Updated Performance Metrics Dataframe and Saved!")
print("------------------------------------------------")


# ---

# ---
# # Fin
# ---

# ---
