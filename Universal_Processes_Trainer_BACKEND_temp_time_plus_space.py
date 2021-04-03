#!/usr/bin/env python
# coding: utf-8

# # Universal $\mathcal{P}_1(\mathbb{R})$-Deep Neural Model (Type A)
# ---

# ---
# # Training Algorithm:
# ---
# ## 1) Generate Data:
# Generates the empirical measure $\sum_{n=1}^N \delta_{X_T(\omega_n)}$ of $X_T$ conditional on $X_0=x_0\in \mathbb{R}$ *($x_0$ and $T>0$ are user-provided)* by simulating from:
# $$ 
# X_T = x + \int_0^T \alpha(s,x)ds + \int_0^T(\beta(s,x)+\sigma_s^H)dW_s.
# $$
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

# ### Visualization

# In[59]:


# How many random polulations to visualize:
Visualization_Size = 4


# ### Quantization
# *This hyperparameter describes the proportion of the data used as sample-barycenters.*

# In[60]:


Quantization_Proportion = 0.75


# ### Simulation

# In[61]:


## Monte-Carlo
N_Euler_Maruyama_Steps = 5
N_Monte_Carlo_Samples = 10**1
N_Monte_Carlo_Samples_Test = 10**1 # How many MC-samples to draw from test-set?

# Roughness
Rougness = 0.1
Ratio_fBM_to_typical_vol = 0.1

T_end = 1
Direct_Sampling = True #This hyperparameter determines if we use a Euler-Maryama scheme or if we use something else.  

## Grid
N_Grid_Finess = 10
Max_Grid = 1


# **Note**: Setting *N_Quantizers_to_parameterize* prevents any barycenters and sub-sampling.

# #### Mode: Code-Testin Parameter(s)

# In[62]:


trial_run = True


# ### Meta-parameters

# In[63]:


# Test-size Ratio
test_size_ratio = .75


# ## SDE Simulation Hyper-Parameter(s)

# ### Drift

# In[64]:


def alpha(t,x):
    return 0#t*np.sin(math.pi*x) #+ np.exp(-t)


# ### Volatility

# In[65]:


def beta(t,x):
    return (1+t) + np.cos(x)


# ### Get Paths

# In[66]:


# load dataset
results_path = "./outputs/models/"
results_tables_path = "./outputs/results/"
raw_data_path_folder = "./inputs/raw/"
data_path_folder = "./inputs/data/"


# ### Import

# In[67]:


# Load Packages/Modules
exec(open('Init_Dump.py').read())
# Load Hyper-parameter Grid
exec(open('CV_Grid.py').read())
# Load Helper Function(s)
# %run ParaGAN_Backend.ipynb
exec(open('Helper_Functions.py').read())
# Import time separately
import time


# ### Set Seed

# In[68]:


random.seed(2021)
np.random.seed(2021)
tf.random.set_seed(2021)


# ## Get Internal (Hyper)-Parameter(s)
# *Initialize the hyperparameters which are fully-specified by the user-provided hyperparameter(s).*

# ### Initialize Grid
# This is $\mathbb{X}$ and it represents the grid of initial states.

# In[69]:


# Get Input Data
#----------------------------------------------------------#
## Train
x_Grid = np.arange(start=-Max_Grid,
                   stop=Max_Grid,
                   step=(2*Max_Grid/N_Grid_Finess))
## Get Number of Instances in Grid: Training
N_Grid_Instances = len(x_Grid)

#----------------------------------------------------------#
## Test
x_Grid_test = np.sort(np.random.uniform(low=-Max_Grid,
                                        high=Max_Grid,
                                        size = round(N_Grid_Instances*test_size_ratio)))
# Get Number of Instances in Grid: Test
N_Grid_Instances_test = len(x_Grid_test)
#----------------------------------------------------------#

# Updater User
print("\u2022 Grid Instances: ", N_Grid_Instances, "and :",N_Grid_Instances_test," Testing instances.")


# ### Initialize Counting Parameters
# Initialize the "conting" type parameters which will help us to determine the length of loops and to intialize object's size later on.  

# In[70]:


# Get Internal (Counting) Parameters
N_Quantizers_to_parameterize = round(Quantization_Proportion*N_Grid_Finess)
N_Elements_Per_Cluster = int(round(N_Grid_Instances/N_Quantizers_to_parameterize))

# Update User
print("\u2022",N_Quantizers_to_parameterize," Centers will be produced; from a total datasize of: ",N_Grid_Finess,
      "!  (That's ",Quantization_Proportion,
      " percent).")
print("\u2022 Each Wasserstein-1 Ball should contain: ",
      N_Elements_Per_Cluster, 
      "elements from the training set.")


# ---

# ### Simulate "Rough" SDE:
# $d X_t = \alpha(t,x)dt + (\beta(t,x)+\sigma_t^H)dW_t ;\qquad X_0 =x$
# Where $(\sigma_t^H)_t$ is a fBM with Hurst parameter $H=0.01$.  

# ### Define Sampler - Data-Generator

# Generates the empirical measure $\sum_{n=1}^N \delta_{X_T(\omega_n)}$ of $X_T$ conditional on $X_0=x_0\in \mathbb{R}$ *($x_0$ and $T>0$ are user-provided)*.

# In[71]:


def Euler_Maruyama_Generator(x_0,
                             N_Euler_Maruyama_Steps = 100,
                             N_Monte_Carlo_Samples = 100,
                             T = 1,
                             Hurst = 0.01,
                             Ratio_fBM_to_typical_vol = 0.5): 
    
    #----------------------------#    
    # DEFINE INTERNAL PARAMETERS #
    #----------------------------#
    # Initialize Empirical Measure
    X_T_Empirical = np.zeros(N_Monte_Carlo_Samples)


    # Internal Initialization(s)
    ## Initialize current state
    n_sample = 0
    ## Initialize Incriments
    dt = T/N_Euler_Maruyama_Steps
    sqrt_dt = np.sqrt(dt)

    #-----------------------------#    
    # Generate Monte-Carlo Sample #
    #-----------------------------#
    while n_sample < N_Monte_Carlo_Samples:
        # Reset Step Counter
        t = 1
        # Initialize Current State 
        X_current = x_0
        # Generate roughness
        sigma_rough = FBM(n=N_Euler_Maruyama_Steps, hurst=0.75, length=1, method='daviesharte').fbm()
        # Perform Euler-Maruyama Simulation
        while t<N_Euler_Maruyama_Steps:
            # Update Internal Parameters
            ## Get Current Time
            t_current = t*(T/N_Euler_Maruyama_Steps)

            # Update Generated Path
            drift_t = alpha(t_current,X_current)*dt
            vol_t = ((1-Ratio_fBM_to_typical_vol)*beta(t_current,X_current)+Ratio_fBM_to_typical_vol*(sigma_rough[t]))*np.random.normal(0,sqrt_dt)
            X_current = X_current + drift_t + vol_t

            # Update Counter (EM)
            t = t+1

        # Update Empirical Measure
        X_T_Empirical[n_sample] = X_current

        # Update Counter (MC)
        n_sample = n_sample + 1

    return X_T_Empirical


# ---

# ### Initializations

# In[72]:


# Initialize List of Barycenters
Wasserstein_Barycenters = []
# Initialize Terminal-Time Empirical Measures
## Training Outputs
measures_locations_list = []
measures_weights_list = []
## Testing Outputs
measures_locations_test_list = []
measures_weights_test_list = []

# Initialize (Empirical) Weight(s)
measure_weights = np.ones(N_Monte_Carlo_Samples)/N_Monte_Carlo_Samples
measure_weights_test = np.ones(N_Monte_Carlo_Samples_Test)/N_Monte_Carlo_Samples_Test
# Initialize Quantizer
Init_Quantizer_generic = np.ones(N_Monte_Carlo_Samples)/N_Monte_Carlo_Samples


# ## Generate $\{\hat{\nu}^{N}_{T,x}\}_{x \in \mathbb{X}}$ Build Wasserstein Cover

# #### Get Data

# In[73]:


# Update User
print("Current Monte-Carlo Step:")
if Direct_Sampling == True:
    print("Using Euler-Maruyama distritization + Monte-Carlo Sampling.")
else:
    print("Using Monte-Carlo Sampling directly from measure at time-T of X_T.")

print("===================================")
print("Start Simulation Step: Training Set")
print("===================================")
# Perform Monte-Carlo Data Generation
for i in tqdm(range(N_Grid_Instances)):
    # Get Terminal Distribution Shape
    ###
    
    if Direct_Sampling == True:
        # DIRECT SAMPLING
        measures_locations_loop = (np.random.normal(alpha(1,x_Grid[i]),
                                                    beta(1,x_Grid[i]),
                                                    N_Monte_Carlo_Samples).reshape(-1,))/N_Monte_Carlo_Samples
    else:
        measures_locations_loop = Euler_Maruyama_Generator(x_0=x_Grid[i],
                                                           N_Euler_Maruyama_Steps = N_Euler_Maruyama_Steps,
                                                           N_Monte_Carlo_Samples = N_Monte_Carlo_Samples,
                                                           T = T_end,
                                                           Hurst=Rougness,
                                                           Ratio_fBM_to_typical_vol=Ratio_fBM_to_typical_vol)
    
    # Append to List
    measures_locations_list.append(measures_locations_loop.reshape(-1,1))
    measures_weights_list.append(measure_weights)
    
# Update User
print("==================================")
print("Done Simulation Step: Training Set")
print("==================================")

#----------------------------------------------------------------------------------------------#

# Perform Monte-Carlo Data Generation
print("===============================")
print("Start Simulation Step: Test Set")
print("===============================")
for i in tqdm(range(N_Grid_Instances_test)):
    # Get Terminal Distribution Shape
    ###
    
     
    if Direct_Sampling == True:
        # DIRECT SAMPLING
        measures_locations_test_loop = (np.random.normal(alpha(1,x_Grid[i]),
                                                    beta(1,x_Grid[i]),
                                                    N_Monte_Carlo_Samples_Test).reshape(-1,))/N_Monte_Carlo_Samples_Test
    else:
        measures_locations_test_loop = Euler_Maruyama_Generator(x_0=x_Grid[i],
                                                                N_Euler_Maruyama_Steps = N_Euler_Maruyama_Steps,
                                                                N_Monte_Carlo_Samples = N_Monte_Carlo_Samples_Test,
                                                                T = T_end,
                                                                Hurst=Rougness,
                                                                Ratio_fBM_to_typical_vol=Ratio_fBM_to_typical_vol)
    
    
    # Append to List
    measures_locations_test_list.append(measures_locations_test_loop.reshape(-1,1))
    measures_weights_test_list.append(measure_weights_test)
    
# Update User
print("===============================")
print("Start Simulation Step: Test Set")
print("===============================")


# #### Start Timer (Model Type A)

# In[74]:


# Start Timer
Type_A_timer_Begin = time.time()


# ## Get "Sample Barycenters":
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
# ---

# ## Build Dissimilarity (Distance) Matrix
# *In this step we build a dissimularity matrix of the dataset on the Wasserstein-1 space.  Namely:*
# $$
# \operatorname{Mat}_{\# \mathbb{X},\# \mathbb{X}}\left(\mathbb{R}\right)\ni D; \text{ where}\qquad \, D_{i,j}\triangleq \mathcal{W}_1\left(f(x_i),f(x_j)\right)
# ;
# $$
# *where $f\in C\left((\mathcal{X},\mathcal{P}_1(\mathcal{Y})\right)$ is the "target" function we are learning.*
# 
# **Note**: *Computing the dissimularity matrix is the most costly part of the entire algorithm with a complexity of at-most $\mathcal{O}\left(E_{W} \# \mathbb{X})^2\right)$ where $E_W$ denotes the complexity of a single Wasserstein-1 evaluation between two elements of the dataset.*

# In[75]:


# Initialize Disimilarity Matrix
Dissimilarity_matrix_ot = np.zeros([N_Grid_Instances,N_Grid_Instances])


# Update User
print("\U0001F61A"," Begin Building Distance Matrix"," \U0001F61A")
# Build Disimilarity Matrix
for i in tqdm(range(N_Grid_Instances)):
    for j in range(N_Grid_Instances):
        Dissimilarity_matrix_ot[i,j] = ot.emd2_1d(measures_locations_list[j],
                                                  measures_locations_list[i])
# Update User
print("\U0001F600"," Done Building Distance Matrix","\U0001F600","!")


# ## Initialize Quantities to Loop Over

# ## Get "Sample Barycenters" and Generate Classes

# In[76]:


# Initialize Locations Matrix (Internal to Loop)
measures_locations_list_current = copy.copy(measures_locations_list)
Dissimilarity_matrix_ot_current = copy.copy(Dissimilarity_matrix_ot)

# Initialize masker vector
masker = np.ones(N_Grid_Instances)

# Initialize Sorting Reference Vector (This helps us efficiently scroll through the disimularity matrix to identify the barycenter without having to re-compute the dissimultarity matrix of a sub-saple at every iteration (which is the most costly part of the algorithm!))
Distances_Loop = Dissimilarity_matrix_ot_current.sum(axis=1)

# Initialize Classes (In-Sample)
Classifer_Wasserstein_Centers = np.zeros([N_Quantizers_to_parameterize,N_Grid_Instances])


# In[77]:


# Update User
print("\U0001F61A"," Begin Identifying Sample Barycenters"," \U0001F61A")

# Identify Sample Barycenters
for i in tqdm(range(N_Quantizers_to_parameterize)):    
    # GET BARYCENTER #
    #----------------#
    ## Identify row with minimum total distance
    Barycenter_index = int(Distances_Loop.argsort()[:1][0])
    ## Get Barycenter
    ## Update Barycenters Array ##
    #----------------------------#
    ### Get next Barycenter
    new_barycenter_loop = measures_locations_list_current[Barycenter_index].reshape(-1,1)
    ### Update Array of Barycenters
    if i == 0:
        # Initialize Barycenters Array
        Barycenters_Array = new_barycenter_loop
    else:
        # Populate Barycenters Array
        Barycenters_Array = np.append(Barycenters_Array,new_barycenter_loop,axis=-1)

    # GET CLUSTER #
    #-------------#
    # Identify Cluster for this barycenter (which elements are closest to it)
    Cluster_indices = (masker*Dissimilarity_matrix_ot_current[:,Barycenter_index]).argsort()[:N_Elements_Per_Cluster]
    ## UPDATES Set  M^{(n)}  ##
    #-------------------------#
    Dissimilarity_matrix_ot_current[Cluster_indices,:] = 0
    # Distance-Based Sorting
    Distances_Loop[Cluster_indices] = math.inf

    # Update Cluster
    masker[Cluster_indices] = math.inf
    
    # Update Classes
    Classifer_Wasserstein_Centers[i,Cluster_indices] = 1
#     print(Cluster_indices)

# Update User
print("\U0001F600"," Done Identifying Sample Barycenters","\U0001F600","!")
print(Classifer_Wasserstein_Centers)


# ---

# ### Train Deep Classifier

# In this step, we train a deep (feed-forward) classifier:
# $$
# \hat{f}\triangleq \operatorname{Softmax}_N\circ W_J\circ \sigma \bullet \dots \sigma \bullet W_1,
# $$
# to identify which barycenter we are closest to.

# Re-Load Grid and Redefine Relevant Input/Output dimensions in dictionary.

# #### Train Deep Classifier

# In[78]:


# Re-Load Hyper-parameter Grid
exec(open('CV_Grid.py').read())
# Re-Load Classifier Function(s)
exec(open('Helper_Functions.py').read())


# In[79]:


# Redefine (Dimension-related) Elements of Grid
# param_grid_Deep_Classifier['input_dim'] = [1]
param_grid_Deep_Classifier['output_dim'] = [N_Quantizers_to_parameterize]

# Train simple deep classifier
predicted_classes_train, predicted_classes_test, N_params_deep_classifier = build_simple_deep_classifier(n_folds = CV_folds, 
                                                                                                        n_jobs = n_jobs, 
                                                                                                        n_iter = n_iter, 
                                                                                                        param_grid_in=param_grid_Deep_Classifier, 
                                                                                                        X_train = x_Grid, 
                                                                                                        y_train = Classifer_Wasserstein_Centers.T,
                                                                                                        X_test = x_Grid_test)


# #### Get Predicted Quantized Distributions
# - Each *row* of "Predicted_Weights" is the $\beta\in \Delta_N$.
# - Each *Column* of "Barycenters_Array" denotes the $x_1,\dots,x_N$ making up the points of the corresponding empirical measures.

# In[80]:


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

# In[81]:


# Stop Timer
Type_A_timer_end = time.time()
# Compute Lapsed Time Needed For Training
Time_Lapse_Model_A = Type_A_timer_end - Type_A_timer_Begin


# ## Get Moment Predictions

# #### Write Predictions

# ### Training-Set Result(s): 

# In[82]:


print("Building Training Set Performance Metrics")

# Initialize Wasserstein-1 Error Distribution
W1_errors = np.array([])
Mean_errors = np.array([])
Var_errors = np.array([])
Skewness_errors = np.array([])
Kurtosis_errors = np.array([])
#---------------------------------------------------------------------------------------------#

# Populate Error Distribution
for x_i in tqdm(range(len(measures_locations_list)-1)):    
    # Get Laws
    W1_loop = ot.emd2_1d(Barycenters_Array,
                         measures_locations_list[x_i].reshape(-1,),
                         Predicted_Weights[x_i,].reshape(-1,),
                         measures_weights_list[x_i].reshape(-1,))
    W1_errors = np.append(W1_errors,W1_loop)
    # Get Means
    Mu_hat = np.sum((Predicted_Weights[x_i])*(Barycenters_Array))
    Mu = np.mean(measures_locations_list[x_i])
    Mean_errors =  np.append(Mean_errors,(Mu_hat-Mu))
    # Get Var (non-centered)
    Var_hat = np.sum((Barycenters_Array**2)*(Predicted_Weights[x_i]))
    Var = np.mean(measures_locations_list[x_i]**2)
    Var_errors = np.append(Var_errors,(Var_hat-Var)**2)
    # Get skewness (non-centered)
    Skewness_hat = np.sum((Barycenters_Array**3)*(Predicted_Weights[x_i]))
    Skewness = np.mean(measures_locations_list[x_i]**3)
    Skewness_errors = np.append(Skewness_errors,(abs(Skewness_hat-Skewness))**(1/3))
    # Get skewness (non-centered)
    Kurtosis_hat = np.sum((Barycenters_Array**4)*(Predicted_Weights[x_i]))
    Kurtosis = np.mean(measures_locations_list[x_i]**4)
    Kurtosis_errors = np.append(Kurtosis_errors,(abs(Kurtosis_hat-Kurtosis))**.25)
    
#---------------------------------------------------------------------------------------------#
# Compute Error Statistics/Descriptors
W1_Performance = np.array([np.mean(np.abs(W1_errors)),np.mean(W1_errors**2)])
Mean_prediction_Performance = np.array([np.mean(np.abs(Mean_errors)),np.mean(Mean_errors**2)])
Var_prediction_Performance = np.array([np.mean(np.abs(Var_errors)),np.mean(Var_errors**2)])
Skewness_prediction_Performance = np.array([np.mean(np.abs(Skewness_errors)),np.mean(Skewness_errors**2)])
Kurtosis_prediction_Performance = np.array([np.mean(np.abs(Kurtosis_errors)),np.mean(Kurtosis_errors**2)])

Type_A_Prediction = pd.DataFrame({"W1":W1_Performance,
                                  "E[X']-E[X]":Mean_prediction_Performance,
                                  "(E[X'^2]-E[X^2])^.5":Var_prediction_Performance,
                                  "(E[X'^3]-E[X^3])^(1/3)":Skewness_prediction_Performance,
                                  "(E[X'^4]-E[X^4])^.25":Kurtosis_prediction_Performance},index=["MAE","MSE"])

# Write Performance
Type_A_Prediction.to_latex((results_tables_path+str("Roughness_")+str(Rougness)+str("__RatiofBM_")+str(Ratio_fBM_to_typical_vol)+
 "__TypeAPrediction_Train.tex"))


#---------------------------------------------------------------------------------------------#
# Update User
print(Type_A_Prediction)


# ---

# ### Test-Set Result(s): 

# In[83]:


print("Building Test Set Performance Metrics")

# Initialize Wasserstein-1 Error Distribution
W1_errors_test = np.array([])
Mean_errors_test = np.array([])
Var_errors_test = np.array([])
Skewness_errors_test = np.array([])
Kurtosis_errors_test = np.array([])
#---------------------------------------------------------------------------------------------#

# Populate Error Distribution
for x_i in tqdm(range(len(measures_locations_test_list)-1)):    
    # Get Laws
    W1_loop_test = ot.emd2_1d(Barycenters_Array,
                         measures_locations_test_list[x_i].reshape(-1,),
                         Predicted_Weights_test[x_i,].reshape(-1,),
                         measures_weights_test_list[x_i].reshape(-1,))
    W1_errors_test = np.append(W1_errors_test,W1_loop_test)
    # Get Means
    Mu_hat_test = np.sum((Predicted_Weights_test[x_i])*(Barycenters_Array))
    Mu_test = np.mean(measures_locations_test_list[x_i])
    Mean_errors_test =  np.append(Mean_errors_test,(Mu_hat_test-Mu_test))
    # Get Var (non-centered)
    Var_hat_test = np.sum((Barycenters_Array**2)*(Predicted_Weights_test[x_i]))
    Var_test = np.mean(measures_locations_test_list[x_i]**2)
    Var_errors_test = np.append(Var_errors_test,(Var_hat_test-Var_test)**2)
    # Get skewness (non-centered)
    Skewness_hat_test = np.sum((Barycenters_Array**3)*(Predicted_Weights_test[x_i]))
    Skewness_test = np.mean(measures_locations_test_list[x_i]**3)
    Skewness_errors_test = np.append(Skewness_errors_test,(abs(Skewness_hat_test-Skewness_test))**(1/3))
    # Get skewness (non-centered)
    Kurtosis_hat_test = np.sum((Barycenters_Array**4)*(Predicted_Weights_test[x_i]))
    Kurtosis_test = np.mean(measures_locations_test_list[x_i]**4)
    Kurtosis_errors_test = np.append(Kurtosis_errors_test,(abs(Kurtosis_hat_test-Kurtosis_test))**.25)
    
#---------------------------------------------------------------------------------------------#
# Compute Error Statistics/Descriptors
W1_Performance_test = np.array([np.mean(np.abs(W1_errors_test)),np.mean(W1_errors_test**2)])
Mean_prediction_Performance_test = np.array([np.mean(np.abs(Mean_errors_test)),np.mean(Mean_errors_test**2)])
Var_prediction_Performance_test = np.array([np.mean(np.abs(Var_errors_test)),np.mean(Var_errors_test**2)])
Skewness_prediction_Performance_test = np.array([np.mean(np.abs(Skewness_errors_test)),np.mean(Skewness_errors_test**2)])
Kurtosis_prediction_Performance_test = np.array([np.mean(np.abs(Kurtosis_errors_test)),np.mean(Kurtosis_errors_test**2)])

Type_A_Prediction_test = pd.DataFrame({"W1":W1_Performance_test,
                                  "E[X']-E[X]":Mean_prediction_Performance_test,
                                  "(E[X'^2]-E[X^2])^.5":Var_prediction_Performance_test,
                                  "(E[X'^3]-E[X^3])^(1/3)":Skewness_prediction_Performance_test,
                                  "(E[X'^4]-E[X^4])^.25":Kurtosis_prediction_Performance_test},index=["MAE","MSE"])

# Write Performance
Type_A_Prediction_test.to_latex((results_tables_path+str("Roughness_")+str(Rougness)+str("__RatiofBM_")+str(Ratio_fBM_to_typical_vol)+
 "__TypeAPrediction_Test.tex"))


# ## Update User

# ### Training-Set Performance

# In[84]:


Type_A_Prediction


# ### Test-Set Performance

# In[85]:


Type_A_Prediction_test


# ### Print for Terminal Legibility

# In[58]:


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


# # Visualization of Performance
# Randomly subsample from output space and visualize empirical measures!

# In[30]:


# # Adjust if is number of plots to visualizes is larger than number of output distributions (But only if there is not enough data!)
# if N_Grid_Instances <= Visualization_Size**2:
#         Visualization_Size = int(round(np.sqrt(min(N_Grid_Instances,Visualization_Size**2)))-1)


# # Initialize Random Sample of input-output pairs to visualize
# plotting_distribution_indices = random.sample(range(N_Grid_Instances), (Visualization_Size)**2)

# # Generate Plot
# f, axarr = plt.subplots(Visualization_Size,Visualization_Size,figsize=(6, 6), dpi=80, facecolor='w', edgecolor='k')
# plt.suptitle("Sample of Predictions")
# for i in range(Visualization_Size):
#     for j in range(Visualization_Size):
#         # Get Current (Randomly chosen (uniformly)) Index
#         current_index = (i*Visualization_Size + j)
#         current_random_index = plotting_distribution_indices[current_index]
#         # Generate Current Plot
#         axarr[i,j].bar(Barycenters_Array,(Predicted_Weights[current_random_index].reshape(-1,)), alpha=0.5,label="Prediction",color="chartreuse")
#         axarr[i,j].bar(measures_locations_list[current_random_index].reshape(-1,),measures_weights_list[current_random_index], alpha=0.5,label="Target",color="purple")


# ---

# ---
# # Fin
# ---

# ---
