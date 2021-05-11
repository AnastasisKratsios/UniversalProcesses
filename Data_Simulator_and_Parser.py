#!/usr/bin/env python
# coding: utf-8

# # Data Simulator and/or Parser
# **Description:** *This scripts prepares, preprocesses, and parses and any datasets used in for the main script*.
# 
# ---

# In[1]:


print("---------------------------------------")
print("Beginning Data-Parsing/Simulation Phase")
print("---------------------------------------")


# #### For Debugging

# In[2]:


trial_run = True

problem_dim = 3


train_test_ratio = .2
N_train_size = 5


## Monte-Carlo
N_Monte_Carlo_Samples = 10**2
N_Euler_Steps = 50
Hurst_Exponent = .6


# Hyper-parameters of Cover
delta = 0.01
Proportion_per_cluster = .5


# Random DNN
# f_unknown_mode = "Heteroskedastic_NonLinear_Regression"

# Random DNN internal noise
# Real-world data version
# f_unknown_mode = "Extreme_Learning_Machine"
dataset_option = 'crypto'
N_Random_Features = 10**2
# Simulated Data version
# f_unknown_mode = "DNN_with_Random_Weights"
Depth_Bayesian_DNN = 2
width = 20

# Random Dropout applied to trained DNN
# f_unknown_mode = "DNN_with_Bayesian_Dropout"
Dropout_rate = 0.1

# Rough SDE (time 1)
# f_unknown_mode = "Rough_SDE"
f_unknown_mode = "Rough_SDE_Vanilla"

# GD with Randomized Input
# f_unknown_mode = "GD_with_randomized_input"
GD_epochs = 2



exec(open('Loader.py').read())
# Load Packages/Modules
exec(open('Init_Dump.py').read())
trial_run = True
# Load Hyper-parameter Grid
exec(open('CV_Grid.py').read())
# Load Helper Function(s)
exec(open('Helper_Functions.py').read())
# Architecture Builder
exec(open('Benchmarks_Model_Builder.py').read())
# Import time separately
import time
#os.environ['CUDA_VISIBLE_DEVICES'] = '0'


# load dataset
results_path = "./outputs/models/"
results_tables_path = "./outputs/results/"
raw_data_path_folder = "./inputs/raw/"
data_path_folder = "./inputs/data/"


### Set Seed
random.seed(2021)
np.random.seed(2021)
tf.random.set_seed(2021)

N_test_size = int(np.round(N_train_size*train_test_ratio,0))




#--------------------------#
# Define Process' Dynamics #
#--------------------------#
drift_constant = 0.1
volatility_constant = 0.01

# Define DNN Applier
def f_unknown_drift_vanilla(x):
    x_internal = x
    x_internal = drift_constant*np.ones(problem_dim)
    return x_internal
def f_unknown_vol_vanilla(x):
    x_internal = volatility_constant*np.diag(np.ones(problem_dim))
    return x_internal


# In[3]:


Train_Set_PredictionTime_MC = time.time()


# # Decide on Which Simulator/Parser To Load:

# In[4]:


print("Deciding on Which Simulator/Parser To Load")


# ## Heteroskedastic_NonLinear_Regression:
# $$
# Y_x \sim f(x) + \text{Laplace}\left(\tilde{f}(x),\|x\|\right).
# $$

# In[5]:


if f_unknown_mode == "Heteroskedastic_NonLinear_Regression":
    #-----------#
    # Build DNN #
    #-----------#
    W_feature = np.random.uniform(size=np.array([width,problem_dim]),low=-.5,high=.5)
    W_readout = np.random.uniform(size=np.array([1,width]),low=-.5,high=.5)
    # Generate Matrices
    for i_weights in range(Depth_Bayesian_DNN):
        W_hidden_loop = np.random.uniform(size=np.array([width,width]),low=-.5,high=.5)
        if i_weights == 0:
            W_hidden_list = [W_hidden_loop]
        else:
            W_hidden_list.append(W_hidden_loop)
    # Define DNN Applier
    def f_unknown(x):
        x_internal = x.reshape(-1,)
        x_internal = np.matmul(W_feature,x)
        #Deep Layer(s)
        for i in range(Depth_Bayesian_DNN):
            W_internal = W_hidden_list[i]
            x_internal = np.matmul(W_internal,x_internal)
            x_internal = np.maximum(0,x_internal)    
        # Readout Layer
        x_internal = np.matmul(W_readout,x_internal)
        return x_internal

    # Define Simulator
    def Simulator(x_in):
        var = np.sqrt(np.sum(x_in**2))
        # Pushforward
        f_x = f_unknown(x_in)
        # Apply Noise After
        noise = np.random.laplace(0,var,N_Monte_Carlo_Samples)
        f_x_noise = np.cos(f_x) + noise
        return f_x_noise


# ## Bayesian DNN

# In[6]:


if f_unknown_mode == "DNN_with_Random_Weights":
    def f_unknown(x):
        x_internal = x.reshape(-1,) 
        # Feature Map Layer
        W_feature = np.random.uniform(size=np.array([width,problem_dim]),low=-.5,high=.5)
        x_internal = np.matmul(W_feature,x)
    #     Deep Layer(s)
        for i in range(Depth_Bayesian_DNN):
            W_internal = np.random.uniform(size=np.array([width,width]),low=-.5,high=.5)
            x_internal = np.matmul(W_internal,x_internal)
            x_internal = np.maximum(0,x_internal)    
        # Readout Layer
        W_readout = np.random.uniform(size=np.array([1,width]),low=-.5,high=.5)
        x_internal = np.matmul(W_readout,x_internal)
        return x_internal


    def Simulator(x_in):
        for i_MC in range(N_Monte_Carlo_Samples):
            y_MC_loop = f_unknown(x_in)
            if i_MC == 0:
                y_MC = y_MC_loop
            else:
                y_MC = np.append(y_MC,y_MC_loop)
        return y_MC


# ## Vanilla DNN with MC-Droupout

# In[7]:


if f_unknown_mode == "DNN_with_Bayesian_Dropout":
    # Initialize Drouput Parameters
    N_Dropout = int(np.maximum(1,round(width*Dropout_rate)))
    
    #-----------#
    # Build DNN #
    #-----------#
    W_feature = np.random.uniform(size=np.array([width,problem_dim]),low=-.5,high=.5)
    W_readout = np.random.uniform(size=np.array([1,width]),low=-.5,high=.5)
    # Generate Matrices
    for i_weights in range(Depth_Bayesian_DNN):
        W_hidden_loop = np.random.uniform(size=np.array([width,width]),low=-.5,high=.5)
        if i_weights == 0:
            W_hidden_list = [W_hidden_loop]
        else:
            W_hidden_list.append(W_hidden_loop)
    # Define DNN Applier
    def f_unknown(x):
        x_internal = x.reshape(-1,)
        x_internal = np.matmul(W_feature,x)
        #Deep Layer(s)
        for i in range(Depth_Bayesian_DNN):
            W_internal = W_hidden_list[i]
            # Apply Random Dropout
            random_mask_coordinates_i = np.random.choice(range(width),N_Dropout)
            random_mask_coordinates_j = np.random.choice(range(width),N_Dropout)
            W_internal[random_mask_coordinates_i,random_mask_coordinates_j] = 0
            # Apply Dropped-out layer
            x_internal = np.matmul(W_internal,x_internal)
            x_internal = np.maximum(0,x_internal)    
        # Readout Layer
        x_internal = np.matmul(W_readout,x_internal)
        return x_internal

    def Simulator(x_in):
        for i_MC in range(N_Monte_Carlo_Samples):
            y_MC_loop = f_unknown(x_in)
            if i_MC == 0:
                y_MC = y_MC_loop
            else:
                y_MC = np.append(y_MC,y_MC_loop)
        return y_MC


# ## (fractional) SDE:
# $$
# (x,t)\mapsto \frac1{S}\sum_{s=1}^S\, \delta_{X_t^{x:s}}; 
# $$
# where $H\in (0,1)$, $\alpha,\beta$ are DNNs of correct dimensions, and the $X_t^{x,s}$ are i.i.d. copies of: 
# $$
# X_t^x \triangleq x + \int_0^t\alpha(s,X_s)ds + \int_0^t \beta(s,X_s)dB_s^H
# .
# $$

# In[8]:


if f_unknown_mode == "Rough_SDE":
    #-------------------#
    # Build DNN (Drift) #
    #-------------------#
    W_feature = np.random.uniform(size=np.array([width,problem_dim]),low=-.5,high=.5)
    W_readout = np.random.uniform(size=np.array([problem_dim,width]),low=-.5,high=.5)
    # Generate Matrices
    for i_weights in range(Depth_Bayesian_DNN):
        W_hidden_loop = np.random.uniform(size=np.array([width,width]),low=-.5,high=.5)
        if i_weights == 0:
            W_hidden_list = [W_hidden_loop]
        else:
            W_hidden_list.append(W_hidden_loop)
    # Define DNN Applier
    def f_unknown_drift(x):
        x_internal = x.reshape(-1,)
        x_internal = np.matmul(W_feature,x)
        #Deep Layer(s)
        for i in range(Depth_Bayesian_DNN):
            W_internal = W_hidden_list[i]
            x_internal = np.matmul(W_internal,x_internal)
            x_internal = np.maximum(0,x_internal)    
        # Readout Layer
        x_internal = np.matmul(W_readout,x_internal)
        return x_internal
    
    #-----------------#
    # Build DNN (Vol) #
    #-----------------#
    W_feature_vol = np.random.uniform(size=np.array([width,problem_dim]),low=-.5,high=.5)
    W_readout_vol = np.random.uniform(size=np.array([problem_dim,width]),low=-.5,high=.5)
    # Generate Matrices
    for i_weights in range(Depth_Bayesian_DNN):
        W_hidden_loop_vol = np.random.uniform(size=np.array([width,width]),low=-.5,high=.5)
        if i_weights == 0:
            W_hidden_list_vol = [W_hidden_loop_vol]
        else:
            W_hidden_list_vol.append(W_hidden_loop_vol)
    def f_unknown_vol(x):
        x_internal = x.reshape(-1,)
        x_internal = np.matmul(W_feature,x)
        #Deep Layer(s)
        for i in range(Depth_Bayesian_DNN):
            W_internal = W_hidden_list[i]
            x_internal = np.matmul(W_internal,x_internal)
            x_internal = np.maximum(0,x_internal)    
        # Readout Layer
        x_internal = np.matmul(W_readout,x_internal)
        x_internal = np.outer(x_internal,x_internal)
        x_internal = np.tanh(x_internal)
        return x_internal
    
    
    
#------------------------------------------------------------------------------#   
#------------------------------------------------------------------------------#   
# Note: The simulator is a bit more complicated in this case that the others.
    def Simulator(x):
        #-------------------#
        # Initialization(s) #
        #-------------------#
        x_init = x.reshape(-1,)

        #--------------------------------#
        # Perform Monte-Carlo Simulation #
        #--------------------------------#
        for i_MC in range(N_Monte_Carlo_Samples):
            # (re) Coerce input_data fBM Path
            x_internal = x_init
            # Get fBM path
            for d in range(problem_dim):
                fBM_gen_loop = (((FBM(n=N_Euler_Steps, hurst=Hurst_Exponent, length=1, method='daviesharte')).fbm())[1:]).reshape(-1,1)
                if d == 0:
                    fBM_gen = fBM_gen_loop
                else:
                    fBM_gen = np.append(fBM_gen,fBM_gen_loop,axis=-1)


            #---------------#
            # Generate Path #
            #---------------#
            for t in range(N_Euler_Steps):
                # Coerce
                x_internal = x_internal.reshape(-1,)
                # Evolve Path
                drift_update = f_unknown_drift(x_internal)/N_Euler_Steps
                vol_update = f_unknown_vol(x_internal)
                x_internal = (x_internal + drift_update + np.matmul(vol_update,fBM_gen[t,])).reshape(1,-1,problem_dim)
                # Coerce
                x_internal = x_internal.reshape(1,-1,problem_dim)
                # Update Sample path
                if t == 0:
                    x_sample_path_loop = x_internal
                else:
                    x_sample_path_loop = np.append(x_sample_path_loop,x_internal,axis=0)
            # Update Sample Path
            if i_MC == 0:
                x_sample_path = x_sample_path_loop
            else:
                x_sample_path = np.append(x_sample_path,x_sample_path_loop,axis=1)

        #------------------------------------------#
        # Get Inputs for These Monte-Carlo Outputs #
        #------------------------------------------#
        ## Generate Path in time
        t_steps = (np.linspace(start = 0, stop = 1, num = N_Euler_Steps)).reshape(-1,1)
        ## Generate x paired with this t
        x_position_initialization = (np.repeat(x.reshape(1,-1),N_Euler_Steps,axis=0)).reshape(-1,problem_dim)
        ## Create (t,x) pairs
        X_inputs_to_return = np.append(t_steps,x_position_initialization,axis=1)


        #------------------------------------------------------#
        # Return Monte-Carlo Sample and Dataset update to User #
        #------------------------------------------------------#
        return X_inputs_to_return, x_sample_path


# ## (fractional) SDE - Vanilla Version:
# $$
# (x,t)\mapsto \frac1{S}\sum_{s=1}^S\, \delta_{X_t^{x:s}}; 
# $$
# where $H\in (0,1)$, $\alpha,\beta$ are known "classical" functions and the $X_t^{x,s}$ are i.i.d. copies of: 
# $$
# X_t^x \triangleq x + \int_0^t\alpha(s,X_s)ds + \int_0^t \beta(s,X_s)dB_s^H
# .
# $$

# In[9]:


if f_unknown_mode == "Rough_SDE_Vanilla": 
    #------------------------------------------------------------------------------#   
    #------------------------------------------------------------------------------#   
    # Note: The simulator is a bit more complicated in this case that the others.
    def Simulator(x):
        #-------------------#
        # Initialization(s) #
        #-------------------#
        x_init = x.reshape(-1,)

        #--------------------------------#
        # Perform Monte-Carlo Simulation #
        #--------------------------------#
        for i_MC in range(N_Monte_Carlo_Samples):
            # (re) Coerce input_data fBM Path
            x_internal = x_init
            # Get fBM path
            if Hurst_Exponent != 0.5:
                for d in range(problem_dim):
                    fBM_gen_loop = (((FBM(n=N_Euler_Steps, hurst=Hurst_Exponent, length=1, method='daviesharte')).fbm())[1:]).reshape(-1,1)
                    
                    if d == 0:
                        fBM_gen = fBM_gen_loop
                    else:
                        fBM_gen = np.append(fBM_gen,fBM_gen_loop,axis=-1)


            #---------------#
            # Generate Path #
            #---------------#
            for t in range(N_Euler_Steps):
                # Coerce
                x_internal = x_internal.reshape(-1,)
                # Evolve Path
                drift_update = f_unknown_drift_vanilla(x_internal)/N_Euler_Steps
                vol_update = f_unknown_vol_vanilla(x_internal)
                if Hurst_Exponent != 0.5:
                    current_noise = fBM_gen[t,]
                else:
                    current_noise = (np.random.normal(1,np.sqrt(1/N_Euler_Steps),problem_dim)).reshape(1,-1)
                x_internal = (x_internal + drift_update + np.matmul(vol_update,current_noise)).reshape(1,-1,problem_dim)
                # Coerce
                x_internal = x_internal.reshape(1,-1,problem_dim)
                # Update Sample path
                if t == 0:
                    x_sample_path_loop = x_internal
                else:
                    x_sample_path_loop = np.append(x_sample_path_loop,x_internal,axis=0)
            # Update Sample Path
            if i_MC == 0:
                x_sample_path = x_sample_path_loop
            else:
                x_sample_path = np.append(x_sample_path,x_sample_path_loop,axis=1)

        #------------------------------------------#
        # Get Inputs for These Monte-Carlo Outputs #
        #------------------------------------------#
        ## Generate Path in time
        t_steps = (np.linspace(start = 0, stop = 1, num = N_Euler_Steps)).reshape(-1,1)
        ## Generate x paired with this t
        x_position_initialization = (np.repeat(x.reshape(1,-1),N_Euler_Steps,axis=0)).reshape(-1,problem_dim)
        ## Create (t,x) pairs
        X_inputs_to_return = np.append(t_steps,x_position_initialization,axis=1)


        #------------------------------------------------------#
        # Return Monte-Carlo Sample and Dataset update to User #
        #------------------------------------------------------#
        return X_inputs_to_return, x_sample_path

    # Set Model to rough_SDE since the rest of the code is identical in that case:
    f_unknown_mode = "Rough_SDE"
    #Done


# # Set/Define: Internal Parameters

# In[10]:


print("Setting/Defining: Internal Parameters")


# ### Dimension of outputs space $\mathcal{Y}=\mathbb{R}^D$.
# 
# **Note:** *This is only relevant for (fractional) SDE Example which is multi-dimensional in the output space.*

# In[11]:


if f_unknown_mode != "Rough_SDE":
    output_dim = 1
else: 
    output_dim = problem_dim


# ## Decide on Testing Set's Size

# In[12]:


N_test_size = int(np.round(N_train_size*train_test_ratio,0))


# ---
# # Decide on Which Type of Data to Get/Simulate
# ---

# In[13]:


print("Deciding on Which Type of Data to Get/Simulate")


# ## Initialize Inputs (Training & Testing) for: 
# *Non-SDE and non GD with random inputs examples*.

# In[14]:


if f_unknown_mode != "GD_with_randomized_input":
    # Get Training Set
    X_train = np.random.uniform(size=np.array([N_train_size,problem_dim]),low=.5,high=1.5)

    # Get Testing Set
    test_set_indices = np.random.choice(range(X_train.shape[0]),N_test_size)
    X_test = X_train[test_set_indices,]
    X_test = X_test + np.random.uniform(low=-(delta/np.sqrt(problem_dim)), 
                                        high = -(delta/np.sqrt(problem_dim)),
                                        size = X_test.shape)


# #### Relabel if fSDE is used instead
# **Explanation:** *The "lowercase x" is used to highlight that the X is made of time-space pairs: (t,x).*

# In[15]:


if f_unknown_mode == "Rough_SDE":
    x_train = X_train
    x_test = X_test


# ## Prase Inputs for: 
# ### Gradient Descent with random initialization:
# $$
# Y_x\triangleq \hat{f}_{\theta_T}(x),\qquad \theta_{t+1} \triangleq \theta_t - \nabla \sum_{x\in \mathbb{X}} \|\hat{f}_{\theta_t}(x)-f(x)\|, \qquad \theta_0 \sim N_d(0,1).
# $$

# In[16]:


if f_unknown_mode == "GD_with_randomized_input":
    # Auxiliary Initialization(s)
    Train_step_proportion = 1-train_test_ratio

    
    if dataset_option == "crypto":
        #--------------#
        # Prepare Data #
        #--------------#
        # Read Dataset
        crypto_data = pd.read_csv('inputs/data/cryptocurrencies/Cryptos_All_in_one.csv')
        # Format Date-Time
        crypto_data['Date'] = pd.to_datetime(crypto_data['Date'],infer_datetime_format=True)
        crypto_data.set_index('Date', drop=True, inplace=True)
        crypto_data.index.names = [None]

        # Remove Missing Data
        crypto_data = crypto_data[crypto_data.isna().any(axis=1)==False]

        # Get Returns
        crypto_returns = crypto_data.diff().iloc[1:]

        # Parse Regressors from Targets
        ## Get Regression Targets
        crypto_target_data = pd.DataFrame({'BITCOIN-closing':crypto_returns['BITCOIN-Close']})
        ## Get Regressors
        crypto_data_returns = crypto_returns.drop('BITCOIN-Close', axis=1)  

        #-------------#
        # Subset Data #
        #-------------#
        # Get indices
        N_train_step = int(round(crypto_data_returns.shape[0]*Train_step_proportion,0))
        N_test_set = int(crypto_data_returns.shape[0] - round(crypto_data_returns.shape[0]*Train_step_proportion,0))
        # # Get Datasets
        X_train = crypto_data_returns[:N_train_step]
        X_test = crypto_data_returns[-N_test_set:]

        ## Coerce into format used in benchmark model(s)
        data_x = X_train
        data_x_test = X_test
        # Get Targets 
        data_y = crypto_target_data[:N_train_step]
        data_y_test = crypto_target_data[-N_test_set:]

        # Scale Data
        scaler = StandardScaler()
        data_x = scaler.fit_transform(data_x)
        data_x_test = scaler.transform(data_x_test)

        # # Update User
        print('#================================================#')
        print(' Training Datasize: '+str(X_train.shape[0])+' and test datasize: ' + str(X_test.shape[0]) + '.  ')
        print('#================================================#')
    
    if dataset_option == "SnP":
        #--------------#
        # Get S&P Data #
        #--------------#
        #=# SnP Constituents #=#
        # Load Data
        snp_data = pd.read_csv('inputs/data/snp500_data/snp500-adjusted-close.csv')
        # Format Data
        ## Index by Time
        snp_data['date'] = pd.to_datetime(snp_data['date'],infer_datetime_format=True)
        #-------------------------------------------------------------------------------#

        #=# SnP Index #=#
        ## Read Regression Target
        snp_index_target_data = pd.read_csv('inputs/data/snp500_data/GSPC.csv')
        ## Get (Reference) Dates
        dates_temp = pd.to_datetime(snp_data['date'],infer_datetime_format=True).tail(600)
        ## Format Target
        snp_index_target_data = pd.DataFrame({'SnP_Index': snp_index_target_data['Close'],'date':dates_temp.reset_index(drop=True)})
        snp_index_target_data['date'] = pd.to_datetime(snp_index_target_data['date'],infer_datetime_format=True)
        snp_index_target_data.set_index('date', drop=True, inplace=True)
        snp_index_target_data.index.names = [None]
        #-------------------------------------------------------------------------------#

        ## Get Rid of Rubbish
        snp_data.set_index('date', drop=True, inplace=True)
        snp_data.index.names = [None]
        ## Get Rid of NAs and Expired Trends
        snp_data = (snp_data.tail(600)).dropna(axis=1).fillna(0)

        # Apple
        snp_index_target_data = snp_data[{'AAPL'}]
        snp_data = snp_data[{'IBM','QCOM','MSFT','CSCO','ADI','MU','MCHP','NVR','NVDA','GOOGL','GOOG'}]
        # Get Return(s)
        snp_data_returns = snp_data.diff().iloc[1:]
        snp_index_target_data_returns = snp_index_target_data.diff().iloc[1:]
        #--------------------------------------------------------#

        #-------------#
        # Subset Data #
        #-------------#
        # Get indices
        N_train_step = int(round(snp_index_target_data_returns.shape[0]*Train_step_proportion,0))
        N_test_set = int(snp_index_target_data_returns.shape[0] - round(snp_index_target_data_returns.shape[0]*Train_step_proportion,0))
        # # Get Datasets
        X_train = snp_data_returns[:N_train_step]
        X_test = snp_data_returns[-N_test_set:]
        ## Coerce into format used in benchmark model(s)
        data_x = X_train
        data_x_test = X_test
        # Get Targets 
        data_y = snp_index_target_data_returns[:N_train_step]
        data_y_test = snp_index_target_data_returns[-N_test_set:]

        # Scale Data
        scaler = StandardScaler()
        data_x = scaler.fit_transform(data_x)
        data_x_test = scaler.transform(data_x_test)

        # # Update User
        print('#================================================#')
        print(' Training Datasize: '+str(X_train.shape[0])+' and test datasize: ' + str(X_test.shape[0]) + '.  ')
        print('#================================================#')

    
    # # Set First Run to Off
    First_run = False

#     #-----------#
#     # Plot Data #
#     #-----------#
#     fig = crypto_data_returns.plot(figsize=(16, 16))
#     fig.get_legend().remove()
#     plt.title("Crypto_Market Returns")

#     # SAVE Figure to .eps
#     plt.savefig('./outputs/plots/'+str(dataset_option)+'_returns.pdf', format='pdf')

    # Redefine Meta-Parameters #
    #--------------------------#
    # Redefine Training Set inputs and ys to train DNN:
    data_y_to_train_DNN_on = (data_y.to_numpy()).reshape(-1,)
    X_train = data_x
    X_test = data_x_test
    problem_dim=data_x.shape[1]



    # Initialize Target Function #
    #----------------------------#
    # Initialize DNN to train
    f_model = get_ffNN(width, Depth_Bayesian_DNN, 0.001, problem_dim, 1)

    # Define Stochastic Prediction Function:
    def f_unknown():
        f_model.fit(data_x,data_y_to_train_DNN_on,epochs = GD_epochs)
        f_x_trained_with_random_initialization_x_train = f_model.predict(X_train)
        f_x_trained_with_random_initialization_x_test = f_model.predict(X_test)
        return f_x_trained_with_random_initialization_x_train, f_x_trained_with_random_initialization_x_test

    def Simulator(x_in):
        for i_MC in range(N_Monte_Carlo_Samples):
            y_MC_loop = f_unknown(x_in)
            if i_MC == 0:
                y_MC = y_MC_loop
            else:
                y_MC = np.append(y_MC,y_MC_loop)
        return y_MC


# ### Extreme Learning-Machine Version

# In[17]:


if f_unknown_mode == "Extreme_Learning_Machine":
    # Auxiliary Initialization(s)
    Train_step_proportion = 1-train_test_ratio
    
    # Vectorized Sigmoid
    
    # custom function
    def sigmoid_univariate(x):
        return 1 / (1 + math.exp(-x))
    sigmoid = np.vectorize(sigmoid_univariate)
    
    # Get Data
    if dataset_option == "crypto":
        #--------------#
        # Prepare Data #
        #--------------#
        # Read Dataset
        crypto_data = pd.read_csv('inputs/data/cryptocurrencies/Cryptos_All_in_one.csv')
        # Format Date-Time
        crypto_data['Date'] = pd.to_datetime(crypto_data['Date'],infer_datetime_format=True)
        crypto_data.set_index('Date', drop=True, inplace=True)
        crypto_data.index.names = [None]

        # Remove Missing Data
        crypto_data = crypto_data[crypto_data.isna().any(axis=1)==False]

        # Get Returns
        crypto_returns = crypto_data.diff().iloc[1:]

        # Parse Regressors from Targets
        ## Get Regression Targets
        crypto_target_data = pd.DataFrame({'BITCOIN-closing':crypto_returns['BITCOIN-Close']})
        ## Get Regressors
        crypto_data_returns = crypto_returns.drop('BITCOIN-Close', axis=1)  

        #-------------#
        # Subset Data #
        #-------------#
        # Get indices
        N_train_step = int(round(crypto_data_returns.shape[0]*Train_step_proportion,0))
        N_test_set = int(crypto_data_returns.shape[0] - round(crypto_data_returns.shape[0]*Train_step_proportion,0))
        # # Get Datasets
        X_train = crypto_data_returns[:N_train_step]
        X_test = crypto_data_returns[-N_test_set:]

        ## Coerce into format used in benchmark model(s)
        data_x = X_train
        data_x_test = X_test
        # Get Targets 
        data_y = crypto_target_data[:N_train_step]
        data_y_test = crypto_target_data[-N_test_set:]

        # Scale Data
        scaler = StandardScaler()
        data_x = scaler.fit_transform(data_x)
        data_x_test = scaler.transform(data_x_test)

        # # Update User
        print('#================================================#')
        print(' Training Datasize: '+str(X_train.shape[0])+' and test datasize: ' + str(X_test.shape[0]) + '.  ')
        print('#================================================#')
    
    if dataset_option == "SnP":
        #--------------#
        # Get S&P Data #
        #--------------#
        #=# SnP Constituents #=#
        # Load Data
        snp_data = pd.read_csv('inputs/data/snp500_data/snp500-adjusted-close.csv')
        # Format Data
        ## Index by Time
        snp_data['date'] = pd.to_datetime(snp_data['date'],infer_datetime_format=True)
        #-------------------------------------------------------------------------------#

        #=# SnP Index #=#
        ## Read Regression Target
        snp_index_target_data = pd.read_csv('inputs/data/snp500_data/GSPC.csv')
        ## Get (Reference) Dates
        dates_temp = pd.to_datetime(snp_data['date'],infer_datetime_format=True).tail(600)
        ## Format Target
        snp_index_target_data = pd.DataFrame({'SnP_Index': snp_index_target_data['Close'],'date':dates_temp.reset_index(drop=True)})
        snp_index_target_data['date'] = pd.to_datetime(snp_index_target_data['date'],infer_datetime_format=True)
        snp_index_target_data.set_index('date', drop=True, inplace=True)
        snp_index_target_data.index.names = [None]
        #-------------------------------------------------------------------------------#

        ## Get Rid of Rubbish
        snp_data.set_index('date', drop=True, inplace=True)
        snp_data.index.names = [None]
        ## Get Rid of NAs and Expired Trends
        snp_data = (snp_data.tail(600)).dropna(axis=1).fillna(0)

        # Apple
        snp_index_target_data = snp_data[{'AAPL'}]
        snp_data = snp_data[{'IBM','QCOM','MSFT','CSCO','ADI','MU','MCHP','NVR','NVDA','GOOGL','GOOG'}]
        # Get Return(s)
        snp_data_returns = snp_data.diff().iloc[1:]
        snp_index_target_data_returns = snp_index_target_data.diff().iloc[1:]
        #--------------------------------------------------------#

        #-------------#
        # Subset Data #
        #-------------#
        # Get indices
        N_train_step = int(round(snp_index_target_data_returns.shape[0]*Train_step_proportion,0))
        N_test_set = int(snp_index_target_data_returns.shape[0] - round(snp_index_target_data_returns.shape[0]*Train_step_proportion,0))
        # # Get Datasets
        X_train = snp_data_returns[:N_train_step]
        X_test = snp_data_returns[-N_test_set:]
        ## Coerce into format used in benchmark model(s)
        data_x = X_train
        data_x_test = X_test
        # Get Targets 
        data_y = snp_index_target_data_returns[:N_train_step]
        data_y_test = snp_index_target_data_returns[-N_test_set:]

        # Scale Data
        scaler = StandardScaler()
        data_x = scaler.fit_transform(data_x)
        data_x_test = scaler.transform(data_x_test)

        # # Update User
        print('#================================================#')
        print(' Training Datasize: '+str(X_train.shape[0])+' and test datasize: ' + str(X_test.shape[0]) + '.  ')
        print('#================================================#')

    
    # # Set First Run to Off
    First_run = False

#     #-----------#
#     # Plot Data #
#     #-----------#
#     fig = crypto_data_returns.plot(figsize=(16, 16))
#     fig.get_legend().remove()
#     plt.title("Crypto_Market Returns")

#     # SAVE Figure to .eps
#     plt.savefig('./outputs/plots/'+str(dataset_option)+'_returns.pdf', format='pdf')

    # Redefine Meta-Parameters #
    #--------------------------#
    # Redefine Training Set inputs and ys to train DNN:
    data_y_to_train_DNN_on = (data_y.to_numpy()).reshape(-1,)
    X_train = data_x
    X_test = data_x_test
    problem_dim=data_x.shape[1]



    # Initialize Target Function #
    #----------------------------#
    # Initialize DNN to train
    f_model = get_ffNN(width, Depth_Bayesian_DNN, 0.001, problem_dim, 1)

    # Define Stochastic Prediction Function:
    def f_unknown():
        f_model.fit(data_x,data_y_to_train_DNN_on,epochs = GD_epochs)
        f_x_trained_with_random_initialization_x_train = f_model.predict(X_train)
        f_x_trained_with_random_initialization_x_test = f_model.predict(X_test)
        return f_x_trained_with_random_initialization_x_train, f_x_trained_with_random_initialization_x_test

    def Simulator(x_in):
        for i_MC in range(N_Monte_Carlo_Samples):
            y_MC_loop = f_unknown(x_in)
            if i_MC == 0:
                y_MC = y_MC_loop
            else:
                y_MC = np.append(y_MC,y_MC_loop)
        return y_MC


# ---

# # Get Output Data

# In[18]:


print("Simulating Output Data for given input data")


# ## Get outputs for all cases besides Gradient-Descent or fractional SDEs:
# ### Training:

# In[19]:


if (f_unknown_mode != "Rough_SDE") and (f_unknown_mode != "GD_with_randomized_input") and (f_unknown_mode != 'Extreme_Learning_Machine'):
    for i in tqdm(range(X_train.shape[0])):
        # Put Datum
        x_loop = X_train[i,]
        # Product Monte-Carlo Sample for Input
        y_loop = (Simulator(x_loop)).reshape(1,-1)

        # Update Dataset
        if i == 0:
            Y_train = y_loop
            Y_train_mean_emp = np.mean(y_loop)
    #         Y_train_var_emp = np.mean((y_loop - np.mean(y_loop))**2)
        else:
            Y_train = np.append(Y_train,y_loop,axis=0)
            Y_train_mean_emp = np.append(Y_train_mean_emp,np.mean(y_loop))
    #         Y_train_var_emp = np.append(Y_train_var_emp,np.mean((y_loop - np.mean(y_loop))**2))
    # Join mean and Variance Training Data
    Y_train_var_emp = np.append(Y_train_mean_emp.reshape(-1,1),Y_train_var_emp.reshape(-1,1),axis=1)


# ### Testing:

# In[20]:


if (f_unknown_mode != "Rough_SDE") and (f_unknown_mode != "GD_with_randomized_input") and (f_unknown_mode != 'Extreme_Learning_Machine'):
    # Start Timer
    Test_Set_PredictionTime_MC = time.time()

    # Generate Data
    for i in tqdm(range(X_test.shape[0])):
        # Put Datum
        x_loop = X_test[i,]
        # Product Monte-Carlo Sample for Input
        y_loop = (Simulator(x_loop)).reshape(1,-1)

        # Update Dataset
        if i == 0:
            Y_test = y_loop
        else:
            Y_test = np.append(Y_test,y_loop,axis=0)

    # End Timer
    Test_Set_PredictionTime_MC = time.time() - Test_Set_PredictionTime_MC


# ## Special Cases (Simulated Differently for higher efficiency...since it is possible):

# ### For: "GD_with_randomized_input":
# This variant is more efficient in the case of the gradient-descent with randomized initializations

# In[21]:


# This implemention of the GD algorithm is more efficient (but this only holds for the GD Monte-Carlo method):
if f_unknown_mode == "GD_with_randomized_input":
    # Start Timer
    Test_Set_PredictionTime_MC = time.time()
    for j_MC in range(N_Monte_Carlo_Samples):
        # MC of SGD
        Y_train_loop,Y_test_loop = f_unknown()
        # Update Dataset
        if j_MC == 0:
            Y_train = Y_train_loop
            Y_test = Y_test_loop
        else:
            Y_train = np.append(Y_train,Y_train_loop,axis=1)
            Y_test = np.append(Y_test,Y_test_loop,axis=1)
    # End Timer
    Test_Set_PredictionTime_MC = time.time() - Test_Set_PredictionTime_MC
    
## Get means for mean-prediction models
    ## Training
    for i in tqdm(range(X_train.shape[0])):
        # Product Monte-Carlo Sample for Input
        y_loop = Y_train[i,]

        # Update Dataset
        if i == 0:
            Y_train_mean_emp = np.mean(y_loop)
        else:
            Y_train_mean_emp = np.append(Y_train_mean_emp,np.mean(y_loop))
    ## Testing
    ### Continue Timer
    Test_Set_PredictionTime_MC2 = time.time()
    for i in tqdm(range(X_test.shape[0])):
        # Product Monte-Carlo Sample for Input
        y_loop_test = Y_test[i,]

        # Update Dataset
        if i == 0:
            Y_test_mean_emp = np.mean(y_loop_test)
        else:
            Y_test_mean_emp = np.append(Y_test_mean_emp,np.mean(y_loop_test))
    
    # End Timer
    Test_Set_PredictionTime_MC = (time.time() - Test_Set_PredictionTime_MC2) + Test_Set_PredictionTime_MC


# In[22]:


if f_unknown_mode == 'Extreme_Learning_Machine':
    
    # Initialization(s) #
    #-------------------#
    # Timer(s)
    Test_time_elapse = 0
    Train_time_elapse = 0
    # Features
    X_train_rand_features = X_train
    X_test_rand_features = X_test

    for j_loop in tqdm(range(N_Monte_Carlo_Samples)):
        #--------------------#
        ## Perform Learning ##
        #--------------------#    
        for d_loop in range(Depth_Bayesian_DNN):
            # Timer
            Update_train_time_elapse = time.time()
            # Get Random Features
            #---------------------------------------------------------------------------------------------------#
            Weights_rand = randsp(m=(X_train_rand_features.shape[1]),n=N_Random_Features,density = 0.75)
            biases_rand = np.random.uniform(low=-.5,high=.5,size = N_Random_Features)
            ## Get Random Features
            #---------------------------------------------------------------------------------------------------#
            ### Training
            #### Apply Random (hidden) Weights
            X_train_rand_features = sparse.csr_matrix.dot(X_train_rand_features,Weights_rand)
            #### Apply Random (hidden) Biases
            X_train_rand_features = X_train_rand_features + biases_rand
            #### Apply Discontinuous (Step) Activation function
            if activation_function == 'thresholding':
                X_train_rand_features[X_train_rand_features>0] = 1
                X_train_rand_features[X_train_rand_features<=0] = 0
            else:
                X_train_rand_features = sigmoid(X_train_rand_features)
            #### Compress
            X_train_rand_features = sparse.csr_matrix(X_train_rand_features)
            # TIMER
            Update_train_time_elapse = time.time() - Update_train_time_elapse
            Train_time_elapse = Train_time_elapse + Update_train_time_elapse
            #---------------------------------------------------------------------------------------------------#
            ### Testing

            # TIMER
            Update_test_time_elapse = time.time()

            #### Apply Random (hidden) Weights
            X_test_rand_features = sparse.csr_matrix.dot(X_test_rand_features,Weights_rand) 
            #### Apply Random (hidden) Biases
            X_test_rand_features = X_test_rand_features + biases_rand
            #### Apply Discontinuous (Step) Activation function
            if activation_function == 'thresholding':
                X_test_rand_features[X_test_rand_features>0] = 1
                X_test_rand_features[X_test_rand_features<=0] = 0
            else:
                X_test_rand_features = sigmoid(X_test_rand_features)
            #### Compress
            X_train_rand_features = sparse.csr_matrix(X_train_rand_features)

            # TIMER
            Update_test_time_elapse = time.time() - Update_test_time_elapse
            Test_time_elapse = Test_time_elapse + Update_test_time_elapse

        #---------------------------------------------------------------------------------------------------#
        # Timer
        Update_train_time_elapse = time.time()
        # Train Extreme Learning Machine
        ExLM_reg = Ridge(alpha=(np.random.uniform(low=0,high=1,size=1)[0]))
        ExLM_reg.fit(X_train_rand_features,data_y)
        # Get Predictions
        ## Training Set
        ExLM_predict_train = ExLM_reg.predict(X_train_rand_features)
        # TIMER
        Update_train_time_elapse = time.time() - Update_train_time_elapse
        Train_time_elapse = Train_time_elapse + Update_train_time_elapse



        # TIMER
        Update_test_time_elapse = time.time()
        ## Get Test-Set Prediction(s)
        ExLM_predict_test = ExLM_reg.predict(X_test_rand_features)
        # TIMER
        Update_test_time_elapse = time.time() - Update_test_time_elapse
        Test_time_elapse = Test_time_elapse + Update_test_time_elapse


        # Update Prediction(s) #
        #----------------------#
        if j_loop == 0:
            Y_train = ExLM_predict_train
            Y_test = ExLM_predict_test
        else:
            Y_train = np.append(Y_train,ExLM_predict_train,axis=1)
            Y_test = np.append(Y_test,ExLM_predict_test,axis=1)
        
    # Update MC Training Time
    Train_Set_PredictionTime_MC = Train_time_elapse
    Test_Set_PredictionTime_MC = Test_time_elapse
    
    # Get Mean Training Data
    Y_train_mean_emp = np.mean(Y_train,axis=1)
    Y_test_mean_emp = np.mean(Y_test,axis=1)


# ## Prepare Data for (f)SDE Case

# #### Build Training Set

# In[23]:


if f_unknown_mode == "Rough_SDE":
    for x_i in tqdm(range(x_train.shape[0])):
        # Extrain current initial condition
        x_init_loop = x_train[x_i,]
        # Monte-Carlo Simulate
        X_inputs_to_return_loop, x_sample_path_loop = Simulator(x_init_loop)

        # Update Training dataset (both input(s) and output(s))
        if x_i == 0:
            # Update Input(s)
            X_train = X_inputs_to_return_loop
            # Update Output(s)
            Y_train = x_sample_path_loop
        else:
            # Update Input(s)
            X_train = np.append(X_train,X_inputs_to_return_loop,axis=0)
            # Update Output(s)
            Y_train = np.append(Y_train,x_sample_path_loop,axis=0)
    # Get Mean Training Data
    Y_train_mean_emp = np.mean(Y_train,axis=1)


# #### Build Testing Set

# In[24]:


if f_unknown_mode == "Rough_SDE":
    # End Timer
    Test_Set_PredictionTime_MC = time.time()
    for x_i in tqdm(range(x_test.shape[0])):
        # Extrain current initial condition
        x_init_loop = x_test[x_i,]
        # Monte-Carlo Simulate
        X_inputs_to_return_loop, x_sample_path_loop = Simulator(x_init_loop)
        
        # Update Training dataset (both input(s) and output(s))
        if x_i == 0:
            # Update Input(s)
            X_test = X_inputs_to_return_loop
            # Update Output(s)
            Y_test = x_sample_path_loop
        else:
            # Update Input(s)
            X_test = np.append(X_test,X_inputs_to_return_loop,axis=0)
            # Update Output(s)
            Y_test = np.append(Y_test,x_sample_path_loop,axis=0)
    # Get Testing Mean Data
    Y_test_mean_emp = np.mean(Y_test,axis=1)
    # End Timer
    Test_Set_PredictionTime_MC = time.time() - Test_Set_PredictionTime_MC
    
    f_unknown_mode = "GD_with_randomized_input"


# ## Extra Parsing:

# In[25]:


if f_unknown_mode == "Rough_SDE":
    Y_train_mean_emp = np.sum(Y_train,axis=1)


# In[26]:


print("----------------------------------")
print("Done Data-Parsing/Simulation Phase")
print("----------------------------------")


# In[27]:


Train_Set_PredictionTime_MC = time.time() - Train_Set_PredictionTime_MC


# ---

# # Fin

# ---
