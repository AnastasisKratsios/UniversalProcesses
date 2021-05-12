#!/usr/bin/env python
# coding: utf-8

# # Generic Conditional Laws for Random-Fields - via:
# 
# ## Universal $\mathcal{P}_1(\mathbb{R})$-Deep Neural Model $\mathcal{NN}_{1_{\mathbb{R}^n},\mathcal{D}}^{\sigma:\star}$.
# 
# ---
# 
# By: [Anastasis Kratsios](https://people.math.ethz.ch/~kratsioa/) - 2021.
# 
# ---
# 
# This is the main component of this implementation; namely, it implements and trains the paper's main *deep neural model* $\mathcal{NN}_{1_{\mathbb{R}^n},\mathcal{D}}^{\sigma:\star}$.

# ### Meta-Parameter Dump (easy access for debugging):

# In[12]:


# trial_run = True
# exec(open('Init_Dump.py').read())
# exec(open('Loader.py').read())
# exec(open('Debug_Menu.py').read())
# %run Debug_Menu.ipynb
# Proportion_per_cluster = 0.1


# ---
# # Begin Implementation of $\mathcal{NN}_{1_{\mathbb{R}^d},\mathcal{D}}^{\max\{0,\cdot\}:\star}$:
# ---

# ### Get the measures $\hat{\mu}_n$ via Barycenters...*aka "K-Means"*!
# - We first identify N-balls in the input space (which is equivalent to identifying N balls in the output space by uniform continuity)
# - We then project each of the centers of these balls onto the nearest element of the training set.
# - The corresponing (observed) $f(x)\in \mathcal{P}_1(\mathbb{R})$ are our $\hat{\mu}_n$ (for $n=1,\dots,N$).
# 
# 
# **NB:** *This is essentially what is done in the proof, exect there, we have access to the correct N and the optimal balls (outside the training dataset)...which we clearly do not here...*

# #### Index and identify: $\{f^{-1}[\hat{\mu}_{n=1}^N]\}_{n=1}^N\subset \mathbb{X}!$

# In[17]:


# if (f_unknown_mode != 'Rough_SDE') and (f_unknown_mode != 'Rough_SDE_Vanilla'):
# Initialize k_means
N_Quantizers_to_parameterize = int(np.maximum(2,round(Proportion_per_cluster*X_train.shape[0])))
kmeans = KMeans(n_clusters=N_Quantizers_to_parameterize, random_state=0).fit(X_train)
# Get Classes
Train_classes = np.array(pd.get_dummies(kmeans.labels_))
# Get Center Measures
Barycenters_Array_x = kmeans.cluster_centers_


# ### Get $\{\hat{\mu}_{n=1}^{N}\}$!

# In[46]:


# if (f_unknown_mode != 'Rough_SDE') and (f_unknown_mode != 'Rough_SDE_Vanilla'):
for i in tqdm(range(Barycenters_Array_x.shape[0])):
    # Identify Nearest Datapoint to a ith Barycenter
    #------------------------------------------------------------------------------------------------------#
    ## Get Barycenter "out of sample" in X (NB there is no data-leakage since we know nothing about Y!)
    Bar_x_loop = Barycenters_Array_x[i,]
    ## Project Barycenter onto testset
    distances = np.sum(np.abs(X_train-Bar_x_loop.reshape(-1,)),axis=1)

    # Update Subsetting Index
    if i == 0:
        Barycenters_index = np.array(np.argmin(distances))
    else:
        Barycenters_index = np.append(Barycenters_index,np.array(np.argmin(distances)))

# Subset Training Set-Outputs
if (f_unknown_mode != 'Rough_SDE') and (f_unknown_mode != 'Rough_SDE_Vanilla'):
    Barycenters_Array = Y_train[Barycenters_index,]
else:
    Barycenters_Array = Y_train[Barycenters_index,:,:]
    # Update Problem Dimension (include time)
    problem_dim = problem_dim + 1
    # Save number of cluster produced
    N_Quantizers_to_parameterize = Train_classes.shape[1]


# # Train Model

# #### Start Timer

# In[8]:


# Start Timer
Type_A_timer_Begin = time.time()


# ### Train Deep Classifier

# In this step, we train a deep (feed-forward) classifier:
# $$
# \hat{f}\triangleq \operatorname{Softmax}_N\circ W_J\circ \sigma \bullet \dots \sigma \bullet W_1,
# $$
# to identify which barycenter we are closest to.

# #### Train Deep Classifier

# Re-Load Packages and CV Grid

# In[9]:


# Re-Load Hyper-parameter Grid
exec(open('CV_Grid.py').read())
# Re-Load Classifier Function(s)
exec(open('Helper_Functions.py').read())


# Train Deep Classifier

# In[10]:


print("==========================================")
print("Training Classifer Portion of Type-A Model")
print("==========================================")

# Redefine (Dimension-related) Elements of Grid
param_grid_Deep_Classifier['input_dim'] = [problem_dim]
param_grid_Deep_Classifier['output_dim'] = [N_Quantizers_to_parameterize]

# Train simple deep classifier
predicted_classes_train, predicted_classes_test, N_params_deep_classifier, timer_output = build_simple_deep_classifier(n_folds = CV_folds, 
                                                                                                        n_jobs = n_jobs, 
                                                                                                        n_iter = n_iter, 
                                                                                                        param_grid_in=param_grid_Deep_Classifier, 
                                                                                                        X_train = X_train, 
                                                                                                        y_train = Train_classes,
                                                                                                        X_test = X_test)

print("===============================================")
print("Training Classifer Portion of Type Model: Done!")
print("===============================================")


# #### Get Predicted Quantized Distributions
# - Each *row* of "Predicted_Weights" is the $\beta\in \Delta_N$.
# - Each *Column* of "Barycenters_Array" denotes the $x_1,\dots,x_N$ making up the points of the corresponding empirical measures.

# In[35]:


if (f_unknown_mode != "Rough_SDE") and (f_unknown_mode != "Rough_SDE_Vanilla"):
    for i in range(Barycenters_Array.shape[0]):
        if i == 0:
            points_of_mass = Barycenters_Array[i,]
        else:

            points_of_mass = np.append(points_of_mass,Barycenters_Array[i,])
# Janky way but works
if (f_unknown_mode == "Rough_SDE") or (f_unknown_mode == "Rough_SDE_Vanilla"):
    for i in range(Barycenters_Array.shape[0]):
        if i == 0:
            points_of_mass = Barycenters_Array[i,:,:]
        else:

            points_of_mass = np.append(points_of_mass,Barycenters_Array[i,:,:],axis=0)


# In[36]:


if (f_unknown_mode != "GD_with_randomized_input") and (f_unknown_mode != "Rough_SDE") and (f_unknown_mode != "Extreme_Learning_Machine") and (f_unknown_mode != "Rough_SDE_Vanilla"):
    # Get Noisless Mean
    direct_facts = np.apply_along_axis(f_unknown, 1, X_train)
    direct_facts_test = np.apply_along_axis(f_unknown, 1, X_test)


# ### Get Error(s)

# In[37]:


# %run Evaluation.ipynb
exec(open('Evaluation.py').read())


# #### Initialize Relevant Solvers
# Solve using either:
# - Sinkhorn Regularized Wasserstein Distance of: [Cuturi - Sinkhorn Distances: Lightspeed Computation of Optimal Transport (2016)](https://papers.nips.cc/paper/2013/hash/af21d0c97db2e27e13572cbf59eb343d-Abstract.html)
# - Slices Wasserstein Distance of: [Bonneel, Nicolas, et al. “Sliced and radon wasserstein barycenters of measures.” Journal of Mathematical Imaging and Vision 51.1 (2015): 22-45](https://dl.acm.org/doi/10.1007/s10851-014-0506-3)

# In[38]:


# Transport-problem initializations #
#-----------------------------------#
if output_dim != 1:
    ## Multi-dimensional
    # Externally Update Empirical Weights for multi-dimensional case
    empirical_weights = np.full((N_Monte_Carlo_Samples,),1/N_Monte_Carlo_Samples)
    # Also Initialize
    Sinkhorn_regularization = 0.1
else:
    ## Single-Dimensional
    # Initialize Empirical Weights
    empirical_weights = (np.ones(N_Monte_Carlo_Samples)/N_Monte_Carlo_Samples).reshape(-1,)

#-------------------------#
# Define Transport Solver #
#-------------------------#
def transport_dist(x_source,w_source,x_sink,w_sink,output_dim,OT_method="Sliced",n_projections = 10):
    # Decide which problem to solve (1D or multi-D)?
    if output_dim == 1:
        OT_out = ot.emd2_1d(x_source,
                            x_sink,
                            w_source,
                            w_sink)
    else:
        # COERCSION
        ## Update Source Distribution
        x_source = points_of_mass.reshape(-1,output_dim)
        ## Update Sink Distribution
        x_sink = np.array(Y_train[i,]).reshape(-1,output_dim)
        
        if OT_method == "Sinkhorn":
            OT_out = ot.bregman.empirical_sinkhorn2(X_s = x_source, 
                                                    X_t = x_sink,
                                                    a = w_source, 
                                                    b = w_sink, 
                                                    reg=0.01, 
                                                    verbose=False,
                                                    method = "sinkhorn_stabilized")
            # COERSION
            OT_out = float(OT_out[0])
        else:
            OT_out = ot.sliced.sliced_wasserstein_distance(X_s = x_source, 
                                                           X_t = x_sink,
                                                           a = w_source, 
                                                           b = w_sink, 
                                                           seed = 2020,
                                                           n_projections = n_projections)
            # COERSION
            OT_out = float(OT_out)
    # Return (regularized?) Transport Distance
    return OT_out


# #### Compute *Training* Error(s)

# In[39]:


# print("#--------------------#")
# print(" Get Training Error(s)")
# print("#--------------------#")
# for i in tqdm(range((X_train.shape[0]))):
#     for j in range(N_Quantizers_to_parameterize):
#         b_loop = np.repeat(predicted_classes_train[i,j],N_Monte_Carlo_Samples)
#         if j == 0:
#             b = b_loop
#         else:
#             b = np.append(b,b_loop)
#         b = b.reshape(-1,1)
#         b = b
#     b = np.array(b,dtype=float).reshape(-1,)
#     b = b/N_Monte_Carlo_Samples
    
#     # Compute Error(s)
#     ## W1
#     W1_loop = transport_dist(x_source = points_of_mass,
#                              w_source = b,
#                              x_sink = np.array(Y_train[i,]).reshape(-1,),
#                              w_sink = empirical_weights,
#                              output_dim = output_dim)
    
#     ## M1
#     Mu_hat = np.matmul(points_of_mass.T,b).reshape(-1,)
#     Mu_MC = np.mean(np.array(Y_train[i,]),axis=0).reshape(-1,)
#     if f_unknown_mode == "Heteroskedastic_NonLinear_Regression":
#         Mu = direct_facts[i,]
#     else:
#         Mu = Mu_MC
#     ## Tally W1-Related Errors
#     ## Mu
#     Mean_loop = np.sum(np.abs((Mu_hat-Mu)))
#     Mean_loop_MC = np.sum(np.abs((Mu-Mu_MC)))
    
#     if f_unknown_mode != "Rough_SDE":
#         ## Variance
#         Var_hat = np.sum(((points_of_mass-Mu_hat)**2)*b)
#         Var_MC = np.mean(np.array(Y_train[i]-Mu_MC)**2)
#         if f_unknown_mode == "Heteroskedastic_NonLinear_Regression":
#             Var = 2*np.sum(X_train[i,]**2)
#         else:
#             Var = Var_MC     

#         # Skewness
#         Skewness_hat = np.sum((((points_of_mass-Mu_hat)/Var_hat)**3)*b)
#         Skewness_MC = np.mean((np.array(Y_train[i]-Mu_MC)/Var_MC)**3)
#         if f_unknown_mode == "Heteroskedastic_NonLinear_Regression":
#             Skewness = 0
#         else:
#             Skewness = Skewness_MC

#         # Excess Kurtosis
#         Ex_Kurtosis_hat = np.sum((((points_of_mass-Mu_hat)/Var_hat)**4)*b) - 3
#         Ex_Kurtosis_MC = np.mean((np.array(Y_train[i]-Mu_MC)/Var_MC)**4) - 3
#         if f_unknown_mode == "Heteroskedastic_NonLinear_Regression":
#             Ex_Kurtosis = 3
#         else:
#             Ex_Kurtosis = Ex_Kurtosis_MC
#         # Tally Higher-Order Error(s)
#         ## Var
#         Var_loop = np.sum(np.abs(Var_hat-Var))
#         Var_loop_MC = np.sum(np.abs(Var_MC-Var))
#         ## Skewness
#         Skewness_loop = np.abs(Skewness_hat-Skewness)
#         Skewness_loop_MC = np.abs(Skewness_MC-Skewness)
#         ## Excess Kurtosis
#         Ex_Kurtosis_loop = np.abs(Ex_Kurtosis-Ex_Kurtosis_hat)
#         Ex_Kurtosis_loop_MC = np.abs(Ex_Kurtosis-Ex_Kurtosis_MC)
    
    
#     # Update
#     if i == 0:
#         W1_errors = W1_loop
#         ## DNM
#         Mean_errors =  Mean_loop
#         ## Monte-Carlo
#         Mean_errors_MC =  Mean_loop_MC
#         # Higher-Order Moments
#         if f_unknown_mode != "Rough_SDE":
#             ## DNM
#             Var_errors = Var_loop
#             Skewness_errors = Skewness_loop
#             Ex_Kurtosis_errors = Ex_Kurtosis_loop
#             ## Monte-Carlo
#             Mean_errors_MC =  Mean_loop_MC
#             Var_errors_MC = Var_loop_MC
#             Skewness_errors_MC = Skewness_loop_MC
#             Ex_Kurtosis_errors_MC = Ex_Kurtosis_loop_MC
        
        
#     else:
#         W1_errors = np.append(W1_errors,W1_loop)
#         # Moments
#         ## DNM
#         Mean_errors =  np.append(Mean_errors,Mean_loop)
#         ## Monte-Carlo
#         Mean_errors_MC =  np.append(Mean_errors_MC,Mean_loop_MC)
#         ## Higher-Order Moments
#         if f_unknown_mode != "Rough_SDE":
#             ## DNM
#             Var_errors = np.append(Var_errors,Var_loop)
#             Skewness_errors = np.append(Skewness_errors,Skewness_loop)
#             Ex_Kurtosis_errors = np.append(Ex_Kurtosis_errors,Ex_Kurtosis_loop)
#             ## Monte-Carlo
#             Var_errors_MC = np.append(Var_errors_MC,Var_loop_MC)
#             Skewness_errors_MC = np.append(Skewness_errors_MC,Skewness_loop_MC)
#             Ex_Kurtosis_errors_MC = np.append(Ex_Kurtosis_errors_MC,Ex_Kurtosis_loop_MC)
            

# ## Get Error Statistics
# W1_Errors = np.array(bootstrap(np.abs(W1_errors),n=N_Boostraps_BCA)(.95))
# Mean_Errors =  np.array(bootstrap(np.abs(Mean_errors),n=N_Boostraps_BCA)(.95))
# Mean_Errors_MC =  np.array(bootstrap(np.abs(Mean_errors_MC),n=N_Boostraps_BCA)(.95))
# print("#-------------------------#")
# print(" Get Training Error(s): END")
# print("#-------------------------#")


# #### Compute *Testing* Errors

# In[59]:


# print("#----------------#")
# print(" Get Test Error(s)")
# print("#----------------#")
# for i in tqdm(range((X_test.shape[0]))):
#     for j in range(N_Quantizers_to_parameterize):
#         b_loop_test = np.repeat(predicted_classes_test[i,j],N_Monte_Carlo_Samples)
#         if j == 0:
#             b_test = b_loop_test
#         else:
#             b_test = np.append(b,b_loop)
#         b_test = b_test.reshape(-1,1)
#     b_test = np.array(b,dtype=float).reshape(-1,)
#     b_test = b/N_Monte_Carlo_Samples
    
#     # Compute Error(s)
#     ## W1
#     W1_loop_test = transport_dist(x_source = points_of_mass,
#                                   w_source = b,
#                                   x_sink = np.array(Y_test[i,]).reshape(-1,),
#                                   w_sink = empirical_weights,
#                                   output_dim = output_dim)
    
#     ## M1
#     Mu_hat_test = np.matmul(points_of_mass.T,b).reshape(-1,)
#     Mu_MC_test = np.mean(np.array(Y_test[i,]),axis=0).reshape(-1,)
#     if f_unknown_mode == "Heteroskedastic_NonLinear_Regression":
#         Mu_test = direct_facts_test[i,]
#     else:
#         Mu_test = Mu_MC_test
#     ## Tally W1-Related Errors
#     ## Mu
#     Mean_loop_test = np.sum(np.abs((Mu_hat_test-Mu_test)))
#     Mean_loop_MC_test = np.sum(np.abs((Mu_test-Mu_MC_test)))
    
#     if f_unknown_mode != "Rough_SDE":
#         ## M2
#         Var_hat_test = np.sum(((points_of_mass-Mu_hat_test)**2)*b)
#         Var_MC_test = np.mean(np.array(Y_test[i]-Mu_MC)**2)
#         if f_unknown_mode == "Rough_SDE":
#             Var_test = 2*np.sum(X_test[i,]**2)
#         else:
#             Var_test = Var_MC

#         ### Error(s)
#         Var_loop_test = np.abs(Var_hat_test-Var_test)
#         Var_loop_MC_test = np.abs(Var_MC_test-Var_test)

#         # Skewness
#         Skewness_hat_test = np.sum((((points_of_mass-Mu_hat_test)/Var_hat_test)**3)*b)
#         Skewness_MC_test = np.mean((np.array(Y_test[i]-Mu_MC_test)/Var_MC_test)**3)
#         if f_unknown_mode == "Heteroskedastic_NonLinear_Regression":
#             Skewness_test = 0
#         else:
#             Skewness_test = Skewness_MC_test
#         ### Error(s)
#         Skewness_loop_test = np.abs(Skewness_hat_test-Skewness_test)
#         Skewness_loop_MC_test = np.abs(Skewness_MC_test-Skewness_test)

#         # Skewness
#         Ex_Kurtosis_hat_test = np.sum((((points_of_mass-Mu_hat_test)/Var_hat_test)**4)*b) - 3
#         Ex_Kurtosis_MC_test = np.mean((np.array(Y_test[i]-Mu_MC_test)/Var_MC_test)**4) - 3
#         if f_unknown_mode == "Heteroskedastic_NonLinear_Regression":
#             Ex_Kurtosis_test = 3
#         else:
#             Ex_Kurtosis_test = Ex_Kurtosis_MC_test
#         ### Error(s)
#         Ex_Kurtosis_loop_test = np.abs(Ex_Kurtosis_test-Ex_Kurtosis_hat_test)
#         Ex_Kurtosis_loop_MC_test = np.abs(Ex_Kurtosis_test-Ex_Kurtosis_MC_test)
    
    
#     # Update
#     if i == 0:
#         W1_errors_test = W1_loop_test
#         ## DNM
#         Mean_errors_test =  Mean_loop_test
#         ## Monte-Carlo
#         Mean_errors_MC_test =  Mean_loop_MC_test
#         ### Get Higher-Moments
#         if f_unknown_mode != "Rough_SDE":
#             ## DNM
#             Var_errors_test = Var_loop_test
#             Skewness_errors_test = Skewness_loop_test
#             Ex_Kurtosis_errors_test = Ex_Kurtosis_loop_test
#             ## Monte-Carlo
#             Var_errors_MC_test = Var_loop_MC_test
#             Skewness_errors_MC_test = Skewness_loop_MC_test
#             Ex_Kurtosis_errors_MC_test = Ex_Kurtosis_loop_MC_test
            
        
#     else:
#         W1_errors_test = np.append(W1_errors_test,W1_loop_test)
#         ## DNM
#         Mean_errors_test =  np.append(Mean_errors_test,Mean_loop_test)
#         ## Monte-Carlo
#         Mean_errors_MC_test =  np.append(Mean_errors_MC_test,Mean_loop_MC_test)
#         ### Get Higher Moments
#         if f_unknown_mode != "Rough_SDE":
#             Var_errors_test = np.append(Var_errors_test,Var_loop_test)
#             Skewness_errors_test = np.append(Skewness_errors_test,Skewness_loop_test)
#             Ex_Kurtosis_errors_test = np.append(Ex_Kurtosis_errors_test,Ex_Kurtosis_loop_test)
#             ## Monte-Carlo
#             Var_errors_MC_test = np.append(Var_errors_MC_test,Var_loop_MC_test)
#             Skewness_errors_MC_test = np.append(Skewness_errors_MC_test,Skewness_loop_MC_test)
#             Ex_Kurtosis_errors_MC_test = np.append(Ex_Kurtosis_errors_MC_test,Ex_Kurtosis_loop_MC_test)

            
# ## Get Error Statistics
# W1_Errors_test = np.array(bootstrap(np.abs(W1_errors_test),n=N_Boostraps_BCA)(.95))
# Mean_Errors_test =  np.array(bootstrap(np.abs(Mean_errors_test),n=N_Boostraps_BCA)(.95))
# Mean_Errors_MC_test =  np.array(bootstrap(np.abs(Mean_errors_MC_test),n=N_Boostraps_BCA)(.95))
# print("#------------------------#")
# print(" Get Testing Error(s): END")
# print("#------------------------#")


# #### Stop Timer

# In[14]:


# # Stop Timer
# Type_A_timer_end = time.time()
# # Compute Lapsed Time Needed For Training
# Time_Lapse_Model_DNM = Type_A_timer_end - Type_A_timer_Begin


# ## Update Tables

# #### Predictive Performance Metrics

# In[69]:


# Summary_pred_Qual_models = pd.DataFrame({"DNM":np.append(np.append(W1_Errors,
#                                                                    Mean_Errors),
#                                                          np.array([N_params_deep_classifier,
#                                                                    Time_Lapse_Model_DNM,
#                                                                    (timer_output/Test_Set_PredictionTime_MC)])),
#                                     "MC-Oracle":np.append(np.append(np.repeat(0,3),
#                                                                    Mean_Errors_MC),
#                                                          np.array([0,
#                                                                    Train_Set_PredictionTime_MC,
#                                                                    (Test_Set_PredictionTime_MC/Test_Set_PredictionTime_MC)])),
#                                    },index=["W1-95L","W1","W1-95R","M-95L","M","M-95R","N_Par","Train_Time","Test_Time/MC-Oracle_Test_Time"])

# Summary_pred_Qual_models_test = pd.DataFrame({"DNM":np.append(np.append(W1_Errors_test,
#                                                                    Mean_Errors_test),
#                                                          np.array([N_params_deep_classifier,
#                                                                    Time_Lapse_Model_DNM,
#                                                                    (timer_output/Test_Set_PredictionTime_MC)])),
#                                     "MC-Oracle":np.append(np.append(np.repeat(0,3),
#                                                                    Mean_Errors_MC_test),
#                                                          np.array([0,
#                                                                    Train_Set_PredictionTime_MC,
#                                                                    (Test_Set_PredictionTime_MC/Test_Set_PredictionTime_MC)])),
#                                    },index=["W1-95L","W1","W1-95R","M-95L","M","M-95R","N_Par","Train_Time","Test_Time/MC-Oracle_Test_Time"])
# ## Get Worst-Case
# Summary_pred_Qual_models_train = Summary_pred_Qual_models
# Summary_pred_Qual_models_internal = Summary_pred_Qual_models.copy()
# Summary_pred_Qual_models = np.maximum(Summary_pred_Qual_models_internal,Summary_pred_Qual_models_test)
# ## Write Performance Metrics
# Summary_pred_Qual_models.to_latex((results_tables_path+"Performance_metrics_Problem_Type_"+str(f_unknown_mode)+"Problemdimension"+str(problem_dim)+"__SUMMARY_METRICS.tex"))
# Summary_pred_Qual_models_train.to_latex((results_tables_path+"Performance_metrics_Problem_Type_"+str(f_unknown_mode)+"Problemdimension"+str(problem_dim)+"__SUMMARY_METRICS_train.tex"))
# Summary_pred_Qual_models_test.to_latex((results_tables_path+"Performance_metrics_Problem_Type_"+str(f_unknown_mode)+"Problemdimension"+str(problem_dim)+"__SUMMARY_METRICS_test.tex"))


# # Update User

# In[70]:


# print(Summary_pred_Qual_models_test)
# Summary_pred_Qual_models_test


# ---

# ---
# # Fin
# ---

# ---
