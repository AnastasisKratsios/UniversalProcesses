#!/usr/bin/env python
# coding: utf-8

# # Simulate and Parse Data Generated from SDE 
# ## with: *fractional Brownian Driver*

# ---

# # Get Path(s)

# In[1]:


# load dataset
results_path = "./outputs/models/"
results_tables_path = "./outputs/results/"
raw_data_path_folder = "./inputs/raw/"
data_path_folder = "./inputs/data/"


# ## Set: Seeds

# In[ ]:


random.seed(2021)
np.random.seed(2021)
tf.random.set_seed(2021)


# ## Get Internal (Hyper)-Parameter(s)
# *Initialize the hyperparameters which are fully-specified by the user-provided hyperparameter(s).*

# ## Initialization of Auxiliary Internal-Variable(s)

# In[1]:


# Get Number of Quantizers
# N_Quantizers_to_parameterize = int(round(N_Grid_Finess*Proportion_per_cluster))

# Initialize (Empirical) Weight(s)
measure_weights = np.ones(N_Monte_Carlo_Samples)/N_Monte_Carlo_Samples
measure_weights_test = np.ones(N_Monte_Carlo_Samples_Test)/N_Monte_Carlo_Samples_Test

# Get number of centers
N_Centers_per_box = max(1,int(round(N_Quantizers_to_parameterize)))
N_points_per_barycenter = 10


# ## Get Barycenters

# In[ ]:


x_Grid_barycenters = np.random.uniform(low=-Max_Grid,
                                       high = Max_Grid, 
                                       size = np.array([N_Centers_per_box,problem_dim]))


# # Get Training and Testing Data

# In[ ]:


print("Building Training + Testing Set - rough-SDE Ground-Truth")

# Initialize position Counter
position_counter = 0
# Barycenter Counter
barycenter_counter = 0
# Iniitalize uniform weights vector
measures_weights_list_loop = np.ones(N_Monte_Carlo_Samples)/N_Monte_Carlo_Samples

# For simplicity override:
N_Monte_Carlo_Samples_Test = N_Monte_Carlo_Samples

# Overrine Number of Centers
N_x = x_Grid_barycenters.shape[0]
N_t = N_Euler_Maruyama_Steps
N_Quantizers_to_parameterize = N_x*N_t
Q_How_many_time_steps_to_sample_per_x = int(round(N_Euler_Maruyama_Steps*Proportion_per_cluster))

# Initialize number of training and testing to grab from each initial condition
N_train = int(N_Euler_Maruyama_Steps*(1-test_size_ratio))
N_test = N_Euler_Maruyama_Steps - N_train

# Initialize Times List
t_Grid = np.linspace(T_begin,T_end,N_Euler_Maruyama_Steps).reshape(-1,1)

for x_bary_i in range(1):#tqdm(range(N_x)): This works; no need to loop
    # Get Current Locations
    x_barycenter = x_Grid_barycenters[x_bary_i,]

    for x_i in tqdm(range(N_points_per_barycenter),leave=True):
        # timer
        if x_i == 1:
            Test_Set_PredictionTime_MC_loop = time.time()
        else:
            Train_Set_PredictionTime_MC_loop = time.time()

        # Get Current x_init by sampling near the barycenter
        if x_i > 0:
            # This case represents the genuine barycenter (which must lie in the dataset eh)
            x_center = x_barycenter
        else:
            x_center = x_barycenter + np.random.uniform(low=delta/2,high=delta/2,size = problem_dim)

        # Update X Grid of Loop
        X_grid_loop = np.repeat((x_center.reshape(1,-1)),
                                N_Euler_Maruyama_Steps,axis=0)
        X_grid_loop = np.append(t_Grid,X_grid_loop,axis=1)

        # Get Sample Paths for this x_init
        current_cover = Euler_Maruyama_Generator(x_0 = x_center,
                                                 N_Euler_Maruyama_Steps = N_Euler_Maruyama_Steps,
                                                 N_Monte_Carlo_Samples = N_Monte_Carlo_Samples,
                                                 T_begin = T_begin,
                                                 T_end = T_end,
                                                 Hurst = Rougness,
                                                 Ratio_fBM_to_typical_vol = Ratio_fBM_to_typical_vol)


        # Update(s) #
        #-----------#
        # Identify Which Elements to Add to Barycenters Array
        ## Identify Which Rows Belong to this Barycenter
        t_indices_barycenters_loop = np.sort(np.random.choice(range(N_Euler_Maruyama_Steps),size = Q_How_many_time_steps_to_sample_per_x, replace=False))
        X_grid_barycenters_loop = X_grid_loop[t_indices_barycenters_loop,:]
        # Get Barycenters for this loop
        Barycenter_update_loop = current_cover[t_indices_barycenters_loop,:,:]

        # Get Current Barycenter Index      
        ## Initializations(Loop)
        barycenter_loop_index = 0
        current_associated_centers_index = t_indices_barycenters_loop[barycenter_loop_index]
        max_possible_loop = max(t_indices_barycenters_loop)
        current_barycenter_index = np.array(range(N_Euler_Maruyama_Steps)) 
        ## Get Clusters
        for loop_index in range(N_Euler_Maruyama_Steps):
            if (current_barycenter_index[loop_index] >= current_associated_centers_index) and (current_barycenter_index[loop_index] < max_possible_loop):
                # Update Active Barycenter
                current_associated_centers_index = t_indices_barycenters_loop[barycenter_loop_index]
                barycenter_loop_index = barycenter_loop_index + 1
                barycenter_counter = barycenter_counter + 1
            # Update Dummy
            current_barycenter_index[loop_index] = barycenter_counter
        
        # Append
        ## Decide if we should initialize or append?...
        if x_bary_i == 0:
            if x_i == 0:
                # Initialize Barycenters
                Barycenters_Array = Barycenter_update_loop
                # Initialize Training Set
                X_train = X_grid_loop
                Y_train = current_cover
                # Initialize Training Timer
                Train_Set_PredictionTime_MC = time.time() - Train_Set_PredictionTime_MC_loop
                # Update Barycenters Array (For training the deep classifier)
                Train_classes = current_barycenter_index
            # Initialize Test Set
            if x_i == 1:
                X_test = X_grid_loop
                Y_test = current_cover
                # Initialize Test Timer
                Test_Set_PredictionTime_MC = time.time() - Test_Set_PredictionTime_MC_loop
        # Update arrays (Now that they're all nice and initialized)
        else:
            if x_i != 1:
                # Update Barycenters
                Barycenters_Array = np.append(Barycenters_Array,Barycenter_update_loop,axis=0)
                # Update Training Data
                X_train = np.append(X_train,X_grid_loop,axis = 0)
                Y_train = np.append(Y_train,current_cover,axis = 0)
                # Update Training Timer
                Train_Set_PredictionTime_MC = (time.time() - Train_Set_PredictionTime_MC) + Train_Set_PredictionTime_MC
                # Update Barycenters Array (For training the deep classifier)
                Train_classes = np.append(Train_classes,current_barycenter_index)
            else:
                # Update Testing Data
                X_test = np.append(X_test,X_grid_loop,axis = 0)
                Y_test = np.append(Y_test,current_cover,axis = 0)
                # Update Test Timer
                Test_Set_PredictionTime_MC = (time.time() - Test_Set_PredictionTime_MC_loop) + Test_Set_PredictionTime_MC
                
# Get Numpy Classes
Train_classes = (pd.get_dummies(Train_classes)).to_numpy()

# Update Number of Centers
N_Quantizers_to_parameterize = Train_classes.shape[1]


# ## Get Mean Data for Benchmark Models
# *(When applicable)*

# In[ ]:


# Get Mean Training Data
Y_train_mean_emp = np.mean(Y_train,axis=1)


# ---
# # Fin
# ---
