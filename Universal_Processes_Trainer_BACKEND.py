#!/usr/bin/env python
# coding: utf-8

# # Generic Conditional Laws for Random-Fields - via:
# 
# ## Universal $\mathcal{P}_1(\mathbb{R})$-Deep Neural Model (Type A)
# 
# ---
# 
# By: [Anastasis Kratsios](https://people.math.ethz.ch/~kratsioa/) - 2021.
# 
# ---

# ---
# # Training Algorithm:
# ---
# ## 1) Generate Data:
# Generates the empirical measure $\sum_{n=1}^N \delta_{X_T(\omega_n)}$ of $X_T$ conditional on $X_0=x_0\in \mathbb{R}$ *($x_0$ and $T>0$ are user-provided)*.
# 
# ## 2) Get "Sample Barycenters":
# Let $\{\mu_n\}_{n=1}^N\subset\mathcal{P}_1(\mathbb{R}^d)$.  Then, the *sample barycenter* is defined by:
# 1. $\mathcal{M}^{(0)}\triangleq \left\{\hat{\mu}_n\right\}_{n=1}^N$,
# 2. For $1\leq n\leq \mbox{N sample barycenters}$: 
#     - $
# \mu^{\star}\in \underset{\tilde{\mu}\in \mathcal{M}^{(n)}}{\operatorname{argmin}}\, \sum_{n=1}^N \mathcal{W}_1\left(\mu^{\star},\mu_n\right),
# $
#     - $\mathcal{M}^{(n)}\triangleq \mathcal{M}^{(n-1)} - \{\mu^{\star}\},$
# *i.e., the closest generated measure form the random sample to all other elements of the random sample.*
# 
# ---
# **Note:** *We simplify the computational burden of getting the correct classes by putting this right into this next loop.*
# 
# ## 3) Train Deep Classifier:
# $\hat{f}\in \operatorname{argmin}_{f \in \mathcal{NN}_{d:N}^{\star}} 
# \sum_{x \in \mathbb{X}}
# \, 
# \mathbb{H}
# \left(
#     \operatorname{Softmax}_N\circ f(x)_n| I\left\{W_1(\hat{\mu}_n,\mu_x),\inf_{m\leq N} W_1(\hat{\mu}_m,\mu_x)\right\}
# \right);
# $
# where $\mathbb{H}$ is the categorical cross-entropy.  
# 
# ---
# ---
# ---
# ## Notes - Why the procedure is so computationally efficient?
# ---
#  - The sample barycenters do not require us to solve for any new Wasserstein-1 Barycenters; which is much more computationally costly,
#  - Our training procedure never back-propages through $\mathcal{W}_1$ since steps 2 and 3 are full-decoupled.  Therefore, training our deep classifier is (comparatively) cheap since it takes values in the standard $N$-simplex.
# 
# ---

# ## Meta-Parameters

# ### Simulation

# #### Ground Truth:
# *The build-in Options:*
# - rSDE 
# - pfBM
# - 2lnflow

# In[2]:


# Option 1:
# groud_truth = "rSDE"
# Option 2:
# groud_truth = "2lnflow"
## Option 3:
# groud_truth = "pfBM"


# #### Grid Hyperparameter(s)

# In[3]:


# ## Monte-Carlo
# N_Euler_Maruyama_Steps = 100
# N_Monte_Carlo_Samples = 10**2
# N_Monte_Carlo_Samples_Test = 10**3 # How many MC-samples to draw from test-set?

# # End times for Time-Grid
# T_end = 1
# T_end_test = 1.1


# ## Grid
# N_Grid_Finess = 100
# Max_Grid = 1

# # Number of Centers (\hat{\mu}_s)
# N_Quantizers_to_parameterize = 10


# This option sets $\delta$ in $B_{\mathbb{R}\times [0,\infty)}(\hat{x}_n,\delta)$; where $\hat{x}_n\in \nu_{\cdot}^{-1}[\hat{\mu}]$.  N_measures_per_center sets the number of samples to draw in this ball...by construction the training set is $\delta$-bounded and $\nu_{(x,t)}$, for any such $x$ is $\omega_{\nu_{\cdot}}(\delta)$-bounded in $\mathcal{P}_1(\mathbb{R})$.

# In[4]:


# # Hyper-parameters of Cover
# delta = 0.1
# N_measures_per_center = 100


# **Note**: Setting *N_Quantizers_to_parameterize* prevents any barycenters and sub-sampling.

# ### Random Cover
# This is not an active option!

# In[5]:


# Set Minibatch Size
# Random_Cover_Mini_Batch_Size = 100


# #### Mode: Code-Testin Parameter(s)
# - True: $\Rightarrow$ cross validation through grid of very mild parameters just to test hardward or software.
# - False: $\Rightarrow$ run CV-grid.

# In[6]:


# trial_run = True


# ### Meta-parameters
# Ratio $\frac{\text{Testing Datasize}}{\text{Training Datasize}}$.

# In[7]:


# test_size_ratio = .25


# ## Simulation from Rough SDE
# Simulate via Euler-M method from:
# $$ 
# X_T^x = x + \int_0^T \alpha(s,X_s^x)ds + \int_0^T((1-\eta)\beta(s,X_s^s)+\eta\sigma_s^H)dW_s.
# $$

# ### Drift

# In[8]:


# def alpha(t,x):
#     return .1*(.1-.5*(.01**2))*t #+ np.cos(x)


# ### Volatility

# In[9]:


# def beta(t,x):
#     return 0.01#+t*np.cos(x)


# ### Roughness Meta-parameters
#  - Roughness is $H$,
#  - Ratio_fBM_to_typical_vol is $\eta$.

# In[10]:


# Rougness = 0.9 # Hurst Parameter
# Ratio_fBM_to_typical_vol = 0 # $\eta$ in equation above.


# ## Simulation from Measure-Valued $2$-Parameter Log-Gaussian Flow
# $$
# X_{t,x} \sim \log\text{-}\mathcal{N}\left(\alpha(t,x),\beta(t,x)\right).
# $$

# **Note:** *$\alpha$ and $\beta$ are specified below in the SDE Example*.

# In[48]:


# def field_dirction_x(x):
#     return x*np.cos(x)

# def finite_variation_t(t):
#     return t*(np.sin(math.pi*t) + np.exp(-t))


# In[49]:


# Run Backend
# exec(open('Universal_Processes_Trainer_BACKEND_Simulation_Based_Method.py').read())


# # INDEV-END

# ### Get Paths

# In[12]:


# load dataset
results_path = "./outputs/models/"
results_tables_path = "./outputs/results/"
raw_data_path_folder = "./inputs/raw/"
data_path_folder = "./inputs/data/"


# ### Import

# In[13]:


# Load Packages/Modules
exec(open('Init_Dump.py').read())
# Load Hyper-parameter Grid
exec(open('CV_Grid.py').read())
# Load Helper Function(s)
exec(open('Helper_Functions.py').read())
# Import time separately
import time


# ### Set Seed

# In[14]:


random.seed(2021)
np.random.seed(2021)
tf.random.set_seed(2021)


# ## Get Internal (Hyper)-Parameter(s)
# *Initialize the hyperparameters which are fully-specified by the user-provided hyperparameter(s).*

# ## Initialization of Auxiliary Internal-Variable(s)

# In[15]:


# Initialize (Empirical) Weight(s)
measure_weights = np.ones(N_Monte_Carlo_Samples)/N_Monte_Carlo_Samples
measure_weights_test = np.ones(N_Monte_Carlo_Samples_Test)/N_Monte_Carlo_Samples_Test

# Get number of centers
N_Centers_per_box = max(1,int(round(np.sqrt(N_Quantizers_to_parameterize))))


# ## Get Centers Grid

# Get x and t grid of "centers".  

# #### Build Coordinates Grid
# 
# *We separate the case of a $2$-parameter measure-valued flow from the SDE example as follows:*

# In[16]:


# Generate Grid of Barycenters
x_Grid_barycenters = np.arange(start=(-Max_Grid+x_0),
                               stop=(Max_Grid+x_0),
                               step = (2*Max_Grid/N_Centers_per_box))
if groud_truth == "2lnflow":
    t_Grid_barycenters = np.arange(start=0,
                                   stop=T_end,
                                   step = (T_end/N_Centers_per_box))


# **Note:** If we do not consider the 2-paramter (probability) measure-valued flow model; then we start all time points at the same place; so as to ensure that the "test set" consists exactly of future times corresponding to trained initial states!

# In[17]:


if not(groud_truth == "2lnflow"):
    t_Grid_barycenters = np.arange(start=0,
                               stop=T_end,
                               step = 1)


# #### Build Full-Grid

# In[18]:


for x_i in range(len(x_Grid_barycenters)):
    for t_j in range(len(t_Grid_barycenters)):
        new_grid_entry = np.array([t_Grid_barycenters[t_j],x_Grid_barycenters[x_i]]).reshape(1,-1)
        if (x_i==0 and t_j ==0):
            Grid_Barycenters = new_grid_entry
        else:
            Grid_Barycenters = np.append(Grid_Barycenters,new_grid_entry,axis=0)

# Update Number of Quantizers Generated
N_Quantizers_to_parameterize = Grid_Barycenters.shape[0]


# ### Generate Data
# This is $\mathbb{X}$ and it represents the grid of initial states.

# In[19]:


# LOAD Simulator (Backend)
# %run Simulator.ipynb
exec(open('Simulator.py').read())


# #### Start Timer (Model Type A)

# In[20]:


# Start Timer
Type_A_timer_Begin = time.time()


# ## $2$-Parameter $\log$-Gaussian Flow
# Generate data by sampling from a random-field $(X_t^x)_{t,x}$ distributed according to:
# $$
# X_{t,x} \sim \log\text{-}\mathcal{N}\left(\alpha(t,x),\beta(t,x)\right).
# $$
# 
# ### Get Training Data

# In[21]:


if groud_truth == "2lnflow":
    print("Building Training Set - 2-logNormal Ground-Truth")
    # Generate Training Data
    for i in tqdm(range(Grid_Barycenters.shape[0])):
        # Get output for center (mu-hat)
        center_current, trash = twoparameter_flow_sampler((Grid_Barycenters[i]).reshape(1,2),N_Monte_Carlo_Samples)

        # Get random sample in delta ball around ith center
        sub_grid_loop = np.random.uniform(0,delta,(N_measures_per_center,2)) + Grid_Barycenters[i]

        # Get Measures for this random sample
        measures_locations_list_current, measures_weights_list_current = twoparameter_flow_sampler(sub_grid_loop,N_Monte_Carlo_Samples)
        ##
        measures_locations_list_current = measures_locations_list_current + center_current
        measures_weights_list_current = measures_weights_list_current + trash
        # Update Classes
        Classifer_Wasserstein_Centers_loop = np.zeros([(N_measures_per_center+1),N_Quantizers_to_parameterize]) # The +1 is to account for the center which will be added to the random ball
        Classifer_Wasserstein_Centers_loop[:, i] =  1
        # Updates Classes
        if i==0:
            # INITIALIZE: Classifiers
            Classifer_Wasserstein_Centers = Classifer_Wasserstein_Centers_loop
            # INITIALIZE: Training Data
            X_train = np.append((Grid_Barycenters[i]).reshape(1,2),sub_grid_loop,axis=0)
            # INITIALIZE: Barycenters Array
            Barycenters_Array = (center_current[0]).reshape(-1,1)
            # INITIALIZE: Measures and locations
            measures_locations_list = measures_locations_list_current
            measures_weights_list = measures_weights_list_current
        else:
            # UPDATE: Classifer
            Classifer_Wasserstein_Centers = np.append(Classifer_Wasserstein_Centers,Classifer_Wasserstein_Centers_loop,axis=0)
            # UPDATE: Training Data
            X_train = np.append(X_train,np.append((Grid_Barycenters[i]).reshape(1,2),sub_grid_loop,axis=0),axis=0)
            # UPDATE: Populate Barycenters Array
            Barycenters_Array = np.append(Barycenters_Array,((center_current[0]).reshape(-1,1)),axis=-1)
            # UPDATE: Measures and locations
            measures_locations_list = measures_locations_list + measures_locations_list_current
            measures_weights_list = measures_locations_list + measures_weights_list_current


# ### Get Testing Data

# In[22]:


if groud_truth == "2lnflow":
    print("Building Test Set - 2-logNormal Ground-Truth")
    # Generate Testing Dataset (Inputs)
    x_tests = np.random.uniform(np.min(X_train[:,0]),np.max(X_train[:,0]),10)
    t_tests = np.arange(start=0,
                        stop=T_end,
                        step = (T_end_test/N_Euler_Maruyama_Steps))
    for x_i in tqdm(range(len(x_tests))):
        for t_j in range(len(t_tests)):
            test_set_entry = np.array([t_tests[t_j],x_tests[x_i]]).reshape(1,-1)
            if (x_i==0 and t_j ==0):
                X_test = test_set_entry
            else:
                X_test = np.append(X_test,test_set_entry,axis=0)

    # Generate Testing Dataset (Outputs)
    measures_locations_test_list, measures_weights_test_list = twoparameter_flow_sampler(X_test,N_Monte_Carlo_Samples_Test)


# ### Rough SDE:
# Simulation of the random-field:
# $$
# X_t^x = x + \int_0^t \alpha(s,X_t^x)ds + (1-\eta)\int_0^t \beta(s,X_t^x)dW_t + \int_0^t B_s^H dW_s;
# $$
# where: 
#  - $(B_t^H)_t$ is a [fractional Brownian Motion](https://arxiv.org/pdf/1406.1956.pdf) with [Hurst exponent](https://en.wikipedia.org/wiki/Hurst_exponent) $H\in (0,1)$,
#  - $(W_t)_t$ is a [Brownian Motion](https://en.wikipedia.org/wiki/Wiener_process),
#  - $\alpha$ and $\beta$ are uniformly [Lipschitz-functions](https://en.wikipedia.org/wiki/Lipschitz_continuity) of appropriate input/output dimension.

# ### Get Training and Testing Data

# In[23]:


# LOAD Simulator (Backend)
# %run Simulator.ipynb
exec(open('Simulator.py').read())


# In[24]:


# NEW?
if groud_truth == "rSDE":
    print("Building Training + Testing Set - rough-SDE Ground-Truth")
    
    # Initialize position Counter
    position_counter = 0
    # Iniitalize uniform weights vector
    measures_weights_list_loop = np.ones(N_Monte_Carlo_Samples)/N_Monte_Carlo_Samples

    # For simplicity override:
    N_Monte_Carlo_Samples_Test = N_Monte_Carlo_Samples
    
    # Overrine Number of Centers
    N_x = len(x_Grid_barycenters)
    N_t = len(t_Grid_barycenters)
    N_Quantizers_to_parameterize = N_x*N_t
    
    # Initialize number of training and testing to grab from each initial condition
    N_train = int(N_Euler_Maruyama_Steps*(1-test_size_ratio))
    N_test = N_Euler_Maruyama_Steps - N_train

    for x_i in tqdm(range(N_x)):
        for t_j in range(N_t):

            # Get Current Locations
            x_center = x_Grid_barycenters[x_i]
            t_center = t_Grid_barycenters[t_j]

            current_cover = Euler_Maruyama_Generator(x_0 = x_center,
                                                     N_Euler_Maruyama_Steps = N_Euler_Maruyama_Steps,
                                                     N_Monte_Carlo_Samples = N_Monte_Carlo_Samples,
                                                     T_begin = t_center,
                                                     T_end = (t_center+delta),
                                                     Hurst = Rougness,
                                                     Ratio_fBM_to_typical_vol = Ratio_fBM_to_typical_vol)
            # Get Barycenter
            barycenter_at_current_location = current_cover[0,:]
            
            # Subset
            ## Measure Location(s)
            measures_locations_list_current_train = (current_cover[:N_train]).tolist()
            measures_locations_list_current_test = (current_cover[:-N_train]).tolist()
            ## Measure Weight(s)
            measures_weights_list_current = list(itertools.repeat(measures_weights_list_loop,N_Monte_Carlo_Samples))

            
            # Get Current Training Data Positions
            t_grid_current = (np.linspace(start=t_center,
                                          stop=(t_center+delta),
                                          num=N_Euler_Maruyama_Steps)).reshape(1,-1)
            x_grid_current = (x_center*np.ones(N_Euler_Maruyama_Steps)).reshape(1,-1)

            X_train_current = (np.append(x_grid_current,t_grid_current,axis=0)).T
            ## Subset
            X_train_updater = X_train_current[:N_train,:] # Get top of array (including center)
            X_test_updater = X_train_current[-N_test:,:] # Get bottom of array (exclusing center)

            # Get Current Classes
            Classifer_Wasserstein_Centers_loop = np.zeros([N_train,N_Quantizers_to_parameterize])
            Classifer_Wasserstein_Centers_loop[:, position_counter] =  1


            # Updates Classes
            if (x_i==0 and t_j==0):
                # INITIALIZE: Classifiers
                Classifer_Wasserstein_Centers = Classifer_Wasserstein_Centers_loop
                # INITIALIZE: Training Data
                X_train = X_train_updater
                X_test = X_test_updater
                # INITIALIZE: Barycenters Array
                Barycenters_Array = barycenter_at_current_location.reshape(-1,1)
                # INITIALIZE: Measures and locations
                measures_locations_list = measures_locations_list_current_train
                measures_locations_test_list = measures_locations_list_current_test
                measures_weights_list = measures_weights_list_current
                measures_weights_test_list = measures_weights_list_current
            else:
                # UPDATE: Classifer
                Classifer_Wasserstein_Centers = np.append(Classifer_Wasserstein_Centers,Classifer_Wasserstein_Centers_loop,axis=0)
                # UPDATE: Training Data
                X_train = np.append(X_train,X_train_updater,axis=0)
                X_test = np.append(X_test,X_test_updater,axis=0)
                # UPDATE: Populate Barycenters Array
                Barycenters_Array = np.append(Barycenters_Array,(barycenter_at_current_location.reshape(-1,1)),axis=-1)
                # UPDATE: Measures and locations
                ## Train
                measures_locations_list = measures_locations_list + measures_locations_list_current_train
                measures_weights_list = measures_locations_list + measures_weights_list_current
                ## Test
                measures_locations_test_list = measures_locations_test_list + measures_locations_list_current_test
                measures_weights_test_list = measures_locations_test_list + measures_weights_list_current

            # Update Position
            position_counter = position_counter + 1


# ---

# ### Train Deep Classifier

# In this step, we train a deep (feed-forward) classifier:
# $$
# \hat{f}\triangleq \operatorname{Softmax}_N\circ W_J\circ \sigma \bullet \dots \sigma \bullet W_1,
# $$
# to identify which barycenter we are closest to.

# Re-Load Grid and Redefine Relevant Input/Output dimensions in dictionary.

# #### Train Deep Classifier

# In[25]:


# Re-Load Hyper-parameter Grid
exec(open('CV_Grid.py').read())
# Re-Load Classifier Function(s)
exec(open('Helper_Functions.py').read())


# In[26]:


print("==========================================")
print("Training Classifer Portion of Type-A Model")
print("==========================================")

# Redefine (Dimension-related) Elements of Grid
param_grid_Deep_Classifier['input_dim'] = [2]
param_grid_Deep_Classifier['output_dim'] = [N_Quantizers_to_parameterize]

# Train simple deep classifier
predicted_classes_train, predicted_classes_test, N_params_deep_classifier = build_simple_deep_classifier(n_folds = CV_folds, 
                                                                                                        n_jobs = n_jobs, 
                                                                                                        n_iter = n_iter, 
                                                                                                        param_grid_in=param_grid_Deep_Classifier, 
                                                                                                        X_train = X_train, 
                                                                                                        y_train = Classifer_Wasserstein_Centers,
                                                                                                        X_test = X_test)

print("=================================================")
print("Training Classifer Portion of Type-A Model: Done!")
print("=================================================")


# #### Get Predicted Quantized Distributions
# - Each *row* of "Predicted_Weights" is the $\beta\in \Delta_N$.
# - Each *Column* of "Barycenters_Array" denotes the $x_1,\dots,x_N$ making up the points of the corresponding empirical measures.

# In[27]:


# Format Weights
## Train
print("#---------------------------------------#")
print("Building Training Set (Regression): START")
print("#---------------------------------------#")
Predicted_Weights = np.array([])
for i in tqdm(range(N_Quantizers_to_parameterize)):    
    b = np.repeat(np.array(predicted_classes_train[:,i],dtype='float').reshape(-1,1),N_Monte_Carlo_Samples,axis=-1)
    b = b/N_Monte_Carlo_Samples
    if i ==0 :
        Predicted_Weights = b
    else:
        Predicted_Weights = np.append(Predicted_Weights,b,axis=1)
print("#-------------------------------------#")
print("Building Training Set (Regression): END")
print("#-------------------------------------#")

## Test
print("#-------------------------------------#")
print("Building Test Set (Predictions): START")
print("#-------------------------------------#")
Predicted_Weights_test = np.array([])
for i in tqdm(range(N_Quantizers_to_parameterize)):
    b_test = np.repeat(np.array(predicted_classes_test[:,i],dtype='float').reshape(-1,1),N_Monte_Carlo_Samples,axis=-1)
    b_test = b_test/N_Monte_Carlo_Samples
    if i ==0 :
        Predicted_Weights_test = b_test
    else:
        Predicted_Weights_test = np.append(Predicted_Weights_test,b_test,axis=1)
print("#-----------------------------------#")
print("Building Test Set (Predictions): END")
print("#-----------------------------------#")
        
# Format Points of Mass
print("#-----------------------------#")
print("Building Barycenters Set: START")
print("#-----------------------------#")
Barycenters_Array = Barycenters_Array.T.reshape(-1,)
print("#-----------------------------#")
print("Building Barycenters Set: END")
print("#-----------------------------#")


# #### Stop Timer

# In[28]:


# Stop Timer
Type_A_timer_end = time.time()
# Compute Lapsed Time Needed For Training
Time_Lapse_Model_A = Type_A_timer_end - Type_A_timer_Begin


# ## Get Moment Predictions

# #### Write Predictions

# ### Training-Set Result(s): 

# In[29]:


print("Building Training Set Performance Metrics")

# Initialize Wasserstein-1 Error Distribution
W1_errors = np.array([])
Mean_errors = np.array([])
Var_errors = np.array([])
Skewness_errors = np.array([])
Kurtosis_errors = np.array([])
predictions_mean = np.array([])
true_mean = np.array([])
#---------------------------------------------------------------------------------------------#

# Populate Error Distribution
for x_i in tqdm(range(len(measures_locations_list)-1)):    
    # Get Laws
    W1_loop = ot.emd2_1d(Barycenters_Array,
                         np.array(measures_locations_list[x_i]).reshape(-1,),
                         Predicted_Weights[x_i,].reshape(-1,),
                         (np.array(measures_weights_list[x_i])).reshape(-1,))
    W1_errors = np.append(W1_errors,W1_loop)
    # Get Means
    Mu_hat = np.sum((Predicted_Weights[x_i])*(Barycenters_Array))
    Mu = np.mean(np.array(measures_locations_list[x_i]))
    Mean_errors =  np.append(Mean_errors,(Mu_hat-Mu))
    ## Update Erros
    predictions_mean = np.append(predictions_mean,Mu_hat)
    true_mean = np.append(true_mean,Mu)
    # Get Var (non-centered)
    Var_hat = np.sum((Barycenters_Array**2)*(Predicted_Weights[x_i]))
    Var = np.mean(np.array(measures_locations_list[x_i])**2)
    Var_errors = np.append(Var_errors,(Var_hat-Var)**2)
    # Get skewness (non-centered)
    Skewness_hat = np.sum((Barycenters_Array**3)*(Predicted_Weights[x_i]))
    Skewness = np.mean(np.array(measures_locations_list[x_i])**3)
    Skewness_errors = np.append(Skewness_errors,(abs(Skewness_hat-Skewness))**(1/3))
    # Get skewness (non-centered)
    Kurtosis_hat = np.sum((Barycenters_Array**4)*(Predicted_Weights[x_i]))
    Kurtosis = np.mean(np.array(measures_locations_list[x_i])**4)
    Kurtosis_errors = np.append(Kurtosis_errors,(abs(Kurtosis_hat-Kurtosis))**.25)
    
#---------------------------------------------------------------------------------------------#
# Compute Error Statistics/Descriptors
W1_Performance = np.array([np.min(np.abs(W1_errors)),np.mean(np.abs(W1_errors)),np.max(np.abs(W1_errors))])
Mean_prediction_Performance = np.array([np.min(np.abs(Mean_errors)),np.mean(np.abs(Mean_errors)),np.max(np.abs(Mean_errors))])
Var_prediction_Performance = np.array([np.min(np.abs(Var_errors)),np.mean(np.abs(Var_errors)),np.max(np.abs(Var_errors))])
Skewness_prediction_Performance = np.array([np.min(np.abs(Skewness_errors)),np.mean(np.abs(Skewness_errors)),np.max(np.abs(Skewness_errors))])
Kurtosis_prediction_Performance = np.array([np.min(np.abs(Kurtosis_errors)),np.mean(np.abs(Kurtosis_errors)),np.max(np.abs(Kurtosis_errors))])

Type_A_Prediction = pd.DataFrame({"W1":W1_Performance,
                                  "E[X']-E[X]":Mean_prediction_Performance,
                                  "(E[X'^2]-E[X^2])^.5":Var_prediction_Performance,
                                  "(E[X'^3]-E[X^3])^(1/3)":Skewness_prediction_Performance,
                                  "(E[X'^4]-E[X^4])^.25":Kurtosis_prediction_Performance},index=["Min","MAE","Max"])

# Write Performance
Type_A_Prediction.to_latex((results_tables_path+str("Roughness_")+str(Rougness)+str("__RatiofBM_")+str(Ratio_fBM_to_typical_vol)+
 "__TypeAPrediction_Train.tex"))


#---------------------------------------------------------------------------------------------#
# Update User
Type_A_Prediction


# ---

# ### Test-Set Result(s): 

# In[30]:


print("Building Test Set Performance Metrics")

# Initialize Wasserstein-1 Error Distribution
W1_errors_test = np.array([])
Mean_errors_test = np.array([])
Var_errors_test = np.array([])
Skewness_errors_test = np.array([])
Kurtosis_errors_test = np.array([])
# Initialize Prediction Metrics
predictions_mean_test = np.array([])
true_mean_test = np.array([])
#---------------------------------------------------------------------------------------------#

# Populate Error Distribution
for x_i in tqdm(range(len(measures_locations_test_list))):    
    # Get Laws
    W1_loop_test = ot.emd2_1d(Barycenters_Array,
                         np.array(measures_locations_test_list[x_i]).reshape(-1,),
                         Predicted_Weights_test[x_i,].reshape(-1,),
                         (np.array(measures_weights_test_list[x_i])).reshape(-1,))
    W1_errors_test = np.append(W1_errors_test,W1_loop_test)
    # Get Means
    Mu_hat_test = np.sum((Predicted_Weights_test[x_i])*(Barycenters_Array))
    Mu_test = np.mean(np.array(measures_locations_test_list[x_i]))
    Mean_errors_test = np.append(Mean_errors_test,(Mu_hat_test-Mu_test))
    ## Update Predictions
    predictions_mean_test = np.append(predictions_mean_test,Mu_hat_test)
    true_mean_test = np.append(true_mean_test,Mu_test)
    # Get Var (non-centered)
    Var_hat_test = np.sum((Barycenters_Array**2)*(Predicted_Weights_test[x_i]))
    Var_test = np.mean(np.array(measures_locations_test_list[x_i])**2)
    Var_errors_test = np.append(Var_errors_test,(Var_hat_test-Var_test)**2)
    # Get skewness (non-centered)
    Skewness_hat_test = np.sum((Barycenters_Array**3)*(Predicted_Weights_test[x_i]))
    Skewness_test = np.mean(np.array(measures_locations_test_list[x_i])**3)
    Skewness_errors_test = np.append(Skewness_errors_test,(abs(Skewness_hat_test-Skewness_test))**(1/3))
    # Get skewness (non-centered)
    Kurtosis_hat_test = np.sum((Barycenters_Array**4)*(Predicted_Weights_test[x_i]))
    Kurtosis_test = np.mean(np.array(measures_locations_test_list[x_i])**4)
    Kurtosis_errors_test = np.append(Kurtosis_errors_test,(abs(Kurtosis_hat_test-Kurtosis_test))**.25)
    
#---------------------------------------------------------------------------------------------#
# Compute Error Statistics/Descriptors
W1_Performance_test = np.array([np.min(np.abs(W1_errors_test)),np.mean(np.abs(W1_errors_test)),np.max(np.abs(W1_errors_test))])
Mean_prediction_Performance_test = np.array([np.min(np.abs(Mean_errors_test)),np.mean(np.abs(Mean_errors_test)),np.max(np.abs(Mean_errors_test))])
Var_prediction_Performance_test = np.array([np.min(np.abs(Var_errors_test)),np.mean(np.abs(Var_errors_test)),np.max(np.abs(Var_errors_test))])
Skewness_prediction_Performance_test = np.array([np.min(np.abs(Skewness_errors_test)),np.mean(np.abs(Skewness_errors_test)),np.max(np.abs(Skewness_errors_test))])
Kurtosis_prediction_Performance_test = np.array([np.min(np.abs(Kurtosis_errors_test)),np.mean(np.abs(Kurtosis_errors_test)),np.max(np.abs(Kurtosis_errors_test))])

Type_A_Prediction_test = pd.DataFrame({"W1":W1_Performance_test,
                                  "E[X']-E[X]":Mean_prediction_Performance_test,
                                  "(E[X'^2]-E[X^2])^.5":Var_prediction_Performance_test,
                                  "(E[X'^3]-E[X^3])^(1/3)":Skewness_prediction_Performance_test,
                                  "(E[X'^4]-E[X^4])^.25":Kurtosis_prediction_Performance_test},index=["Min","MAE","Max"])

# Write Performance
Type_A_Prediction_test.to_latex((results_tables_path+str("Roughness_")+str(Rougness)+str("__RatiofBM_")+str(Ratio_fBM_to_typical_vol)+
 "__TypeAPrediction_Test.tex"))


# # Visualization

# #### Visualization of Training-Set Performance

# In[39]:


plt.plot(predictions_mean,label="prediction",color="purple")
plt.plot(true_mean,label="true",color="green")


# In[32]:


# # plt.plot(predictions_mean_test,color="purple")
# # plt.plot(true_mean_test)

# # Initialize Plot #
# #-----------------#
# plt.figure(num=None, figsize=(12, 12), dpi=80, facecolor='w', edgecolor='k')

# # Generate Plots #
# #----------------#
# ax = plt.axes(projection='3d')
# ax.plot_trisurf(X_train[:-1:,0], X_train[:-1:,1], true_mean, cmap='viridis',linewidth=0.5);
# ax.plot_trisurf(X_train[:-1:,0], X_train[:-1:,1], predictions_mean, cmap='inferno',linewidth=0.5);


# # Format Plot #
# #-------------#
# plt.legend(loc="upper left",prop={'size': 10})
# plt.title("Model Predictions")

# # Export #
# #--------#
# # SAVE Figure to .eps
# plt.savefig('./outputs/plots/Test.pdf', format='pdf')


# #### Visualization of Test-Set Performance

# In[33]:


# # plt.plot(predictions_mean_test,color="purple")
# # plt.plot(true_mean_test)

# # sns.set()
# # Initialize Plot #
# #-----------------#
# plt.figure(num=None, figsize=(12, 12), dpi=80, facecolor='w', edgecolor='k')

# # Generate Plots #
# #----------------#
# ax = plt.axes(projection='3d')
# ax.plot_trisurf(X_test[:-1:,0], X_test[:-1:,1], true_mean_test, cmap='viridis',linewidth=0.5);
# ax.plot_trisurf(X_train[:-1:,0], X_train[:-1:,1], predictions_mean_test, cmap='inferno',linewidth=0.5);


# # Format Plot #
# #-------------#
# plt.legend(loc="upper left",prop={'size': 10})
# plt.title("Model Predictions")

# # Export #
# #--------#
# # SAVE Figure to .eps
# plt.savefig('./outputs/plots/Test.pdf', format='pdf')


# ## Update User

# ### Print for Terminal Legibility

# In[34]:


print("#----------------------#")
print("Training-Set Performance")
print("#----------------------#")
print(Type_A_Prediction)
print(" ")
print(" ")
print(" ")

print("#------------------#")
print("Test-Set Performance")
print("#------------------#")
print(Type_A_Prediction_test)
print(" ")
print(" ")
print(" ")


# ### Training-Set Performance

# In[35]:


# Type_A_Prediction


# ### Test-Set Performance

# In[36]:


# Type_A_Prediction_test


# ---

# ---
# # Fin
# ---

# ---
