#!/usr/bin/env python
# coding: utf-8

# # Simulator

# ## Euler-Maruyama +
# For generating rough SDEs:
# 
# Generates the empirical measure $\sum_{n=1}^N \delta_{X_T(\omega_n)}$ of $X_T$ conditional on $X_0=x_0\in \mathbb{R}$ *($x_0$ and $T>0$ are user-provided)*.

# In[ ]:


def Euler_Maruyama_Generator(x_0,
                             N_Euler_Maruyama_Steps = 10,
                             N_Monte_Carlo_Samples = 100,
                             T_begin = 0,
                             T_end = 1,
                             Hurst = 0.1,
                             Ratio_fBM_to_typical_vol = 0.5): 
    #----------------------------#    
    # DEFINE INTERNAL PARAMETERS #
    #----------------------------#
    # Initialize Empirical Measure
    X_T_Empirical = np.zeros([N_Euler_Maruyama_Steps,N_Monte_Carlo_Samples])


    # Internal Initialization(s)
    ## Initialize current state
    n_sample = 0
    ## Initialize Incriments
    dt = (T_end-T_begin)/N_Euler_Maruyama_Steps
    sqrt_dt = np.sqrt(dt)
    
    #-----------------------------#    
    # Generate Monte-Carlo Sample #
    #-----------------------------#
    for n_sample in range(N_Monte_Carlo_Samples):
        # Initialize Current State 
        X_current = x_0
        # Generate roughness
        sigma_rough = FBM(n=N_Euler_Maruyama_Steps, hurst=0.75, length=1, method='daviesharte').fbm()
        # Perform Euler-Maruyama Simulation
        for t in range(N_Euler_Maruyama_Steps):
            # Update Internal Parameters
            ## Get Current Time
            t_current = t*((T_end - T_begin)/N_Euler_Maruyama_Steps)

            # Update Generated Path
            drift_t = alpha(t_current,X_current)*dt
            W_t = np.random.normal(0,sqrt_dt)
            vol_t = (1-Ratio_fBM_to_typical_vol)*beta(t_current,X_current) #+ Ratio_fBM_to_typical_vol*sigma_rough[t]
            vol_t = vol_t*W_t
            X_current = X_current + drift_t + vol_t

            # Update Empirical Measure
            X_T_Empirical[t,n_sample] = (X_current+ Ratio_fBM_to_typical_vol*sigma_rough[t])

    return X_T_Empirical


# ## Euler-Maruyama Simulator
# Using the above Euler-Maruyama scheme, the next function generates a bunch of paths at the given space-time points.  

# In[7]:


def Euler_Maruyama_simulator(Grid_in,
                             N_Monte_Carlo_Samples=10,
                             Rougness=0.01,
                             N_Euler_Maruyama_Steps =10,
                             Ratio_fBM_to_typical_vol=0.5):
    # Internal Parameters
    t_Grid = Grid_in[:,0]
    x_Grid = Grid_in[:,1]
    N_Grid_Instances_x = len(x_Grid)
    # Initializations
    measure_weights = np.ones(N_Monte_Carlo_Samples)/N_Monte_Carlo_Samples
    measures_locations_list_internal = []
    measures_weights_list_internal = []
    

    # Perform Euler-Maruyama distritization + Monte-Carlo Sampling.
    #----------------------------------------------------------------------------------------------#

    # Initialize fBM Generator
    fBM_Generator = FBM(n=N_Euler_Maruyama_Steps, hurst=0.75, length=1, method='daviesharte')

    # Perform Monte-Carlo Data Generation
    for i in tqdm(range(N_Grid_Instances_x)):
        x_loop = x_Grid[i]
        # Get x
        field_loop_x = field_dirction_x(x_loop)
        # Simulate Paths
        paths_loop = Euler_Maruyama_Generator(x_0=x_loop,
                                              N_Euler_Maruyama_Steps = N_Euler_Maruyama_Steps,
                                              N_Monte_Carlo_Samples = N_Monte_Carlo_Samples,
                                              T = (t_Grid[(N_Euler_Maruyama_Steps-1)]),
                                              Hurst=Rougness,
                                              Ratio_fBM_to_typical_vol=Ratio_fBM_to_typical_vol)

        # Map numpy to list
        measures_locations_loop = paths_loop.tolist()

        # Append to List
        measures_locations_list_internal = measures_locations_list_internal + measures_locations_loop
        measures_weights_list_internal.append(measure_weights)
        
        # Get Positions
        X_loop = np.append(np.repeat(x_loop,N_Euler_Maruyama_Steps).reshape(-1,1),
                           t_Grid.reshape(-1,1),
                           axis=1)    
        # Update Inputs
        if i==0:
            X_internal = X_loop
        else:
            X_internal = np.append(X_train,X_loop,axis=0)
    
        
    
    return measures_locations_list_internal, measures_weights_list_internal, X_internal, N_Euler_Maruyama_Steps


# ## 2-Parameter Measure-Valued Flow

# In[6]:


def twoparameter_flow_sampler(Grid_in,N_Monte_Carlo_Samples=10):
    ## Get Dimensions
    N_Grid_Instances_x = Grid_in.shape[0]
    N_Grid_Instances_t = Grid_in.shape[0]
    # Initializations
    measure_weights = np.ones(N_Monte_Carlo_Samples)/N_Monte_Carlo_Samples
    measures_locations_list_internal = []
    measures_weights_list_internal = []

    #----------------------------------------------------------------------------------------------#
    # Perform Monte-Carlo Data Generation
    for i in range(N_Grid_Instances_x):
        t_loop = Grid_in[i,][0]
        x_loop = Grid_in[i,][1]
        # Generate finite-variation path (since it stays unchanged)
        measures_locations_loop = np.random.lognormal(alpha(t_loop,x_loop),
                                                          beta(t_loop,x_loop),
                                                          N_Monte_Carlo_Samples)

        # Append to List
        measures_locations_list_internal = measures_locations_list_internal + [measures_locations_loop]
        measures_weights_list_internal.append(measure_weights)
    
    return measures_locations_list_internal, measures_weights_list_internal#, X_internal


# # Perturbed fBM Simulator

# In[ ]:


def perturbed_fBM_simulator(x_Grid,t_Grid,N_Monte_Carlo_Samples=10):
    ## Get Dimensions
    N_Grid_Instances_x = x_Grid.shape[0]
    N_Grid_Instances_t = t_Grid.shape[0]
    # Initializations
    measure_weights = np.ones(N_Monte_Carlo_Samples)/N_Monte_Carlo_Samples
    measures_locations_list_internal = []
    measures_weights_list_internal = []


    print("===================================")
    print("Start Simulation Step: Training Set")
    print("===================================")
    # Initialize fBM Generator
    fBM_Generator = FBM(n=N_Euler_Maruyama_Steps, hurst=0.75, length=1, method='daviesharte')

    # Perform Monte-Carlo Data Generation
    for i in tqdm(range(N_Grid_Instances_x)):
        t_loop = Grid_in[i,][0]
        x_loop = Grid_in[i,][1]
        # Get x
        field_loop_x = field_dirction_x(x_loop)
        # Get omega and t
        # Generate finite-variation path (since it stays unchanged)
        finite_variation_path = finite_variation_t(t_loop).reshape(-1,1) +field_loop_x
        for n_MC in range(N_Monte_Carlo_Samples):
            fBM_variation_path_loop = fBM_Generator.fbm().reshape(-1,1)
            generated_path_loop = finite_variation_path + fBM_variation_path_loop
            if n_MC == 0:
                paths_loop = generated_path_loop
            else:
                paths_loop = np.append(paths_loop,generated_path_loop,axis=-1)
        
        # Map numpy to list
        measures_locations_loop = paths_loop.tolist()
        # Get inputs
        X_loop = np.append(np.repeat(x_Grid[i],(N_Euler_Maruyama_Steps+1)).reshape(-1,1),
                                 t_Grid.reshape(-1,1),
                                 axis=1)
        
        # Append to List
        measures_locations_list_internal = measures_locations_list_internal + measures_locations_loop
        measures_weights_list_internal.append(measure_weights)
        
        # Update Inputs
        if i==0:
            X_internal = X_loop
        else:
            X_internal = np.append(X_train,X_loop,axis=0)
    
    # Update User
    print("======================")
    print(" Done Simulation Step ")
    print("======================")
    
    return measures_locations_list_internal, measures_weights_list_internal, X_internal

