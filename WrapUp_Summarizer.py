#!/usr/bin/env python
# coding: utf-8

# ---
# # This Script Wraps-Ups and Saves the Final Experiment's Final Results 
# ---

# ## Initialize Final Path

# In[1]:


# Final Outputs
results_tables_path_final = "./outputs/results/Final_Results/"
Path(results_tables_path_final).mkdir(parents=True, exist_ok=True)


# ## Save and Summarize Model Complexitie(s)

# #### Compute

# In[ ]:


# Coercion
Summary_Complexity_models = Summary_Complexity_models.T
# Compute Complexity Metrics for GPR
Deep_Neural_Model_Facts = np.array([(N_Monte_Carlo_Samples*(X_train.shape[0]+X_test.shape[0])),Test_Set_PredictionTime_MC,1])
MCOracle_Facts = np.array([(N_Monte_Carlo_Samples*(X_train.shape[0]+X_test.shape[0])),Test_Set_PredictionTime_MC,1])
# Update Model Complexities
Summary_Complexity_models["DNM"] = pd.Series(Deep_Neural_Model_Facts, index=Summary_Complexity_models.index)
Summary_Complexity_models["MC_Oracle"] = pd.Series(MCOracle_Facts, index=Summary_Complexity_models.index)
# Coercion
Summary_Complexity_models = Summary_Complexity_models
Model_Complexities_Final = Summary_Complexity_models


# #### Write

# In[ ]:


# Save #
pd.set_option('display.float_format', '{:.4E}'.format)
Model_Complexities_Final.to_latex((results_tables_path_final+"Latent_Width_NSDE"+str(width)+"Problemdimension"+str(problem_dim)+"__ModelComplexities.tex"))


# ## Prediction Metrics

# ### Compute

# In[ ]:


# Get DNM Prediction Quality Metrics
## Train
W1_Errors_DNM_train = np.array([np.mean(np.abs(W1_errors)),
                                np.mean(np.abs(Mean_errors)),
                                np.mean(np.abs(Var_errors)),
                                np.mean(np.abs(Skewness_errors)),
                                np.mean(np.abs(Ex_Kurtosis_errors))])
## Test
W1_Errors_DNM_test =np.array([np.mean(np.abs(W1_errors_test)),
                              np.mean(np.abs(Mean_errors_test)),
                              np.mean(np.abs(Var_errors_test)),
                              np.mean(np.abs(Skewness_errors_test)),
                              np.mean(np.abs(Ex_Kurtosis_errors_test))])
# Get MC-Oracle Quality Metrics
## Train
W1_Errors_MCOracle_train = np.array([0,
                                np.mean(np.abs(Mean_errors_MC)),
                                np.mean(np.abs(Var_errors_MC)),
                                np.mean(np.abs(Skewness_errors_MC)),
                                np.mean(np.abs(Ex_Kurtosis_errors_MC))])
## Test
W1_Errors_MCOracle_test =np.array([0,
                              np.mean(np.abs(Mean_errors_MC_test)),
                              np.mean(np.abs(Var_errors_MC_test)),
                              np.mean(np.abs(Skewness_errors_MC_test)),
                              np.mean(np.abs(Ex_Kurtosis_errors_MC_test))])


# ### Summarize and Write

# In[ ]:


print("-------------------------------------------------")
print("Updating Performance Metrics Dataframe and Saved!")
print("-------------------------------------------------")
# Append Gaussian Process Regressor Performance
## Train
Summary_pred_Qual_models["MC-Oracle"] = pd.Series((W1_Errors_MCOracle_train), index=Summary_pred_Qual_models.index)
## Test
Summary_pred_Qual_models_test["MC-Oracle"] = pd.Series((W1_Errors_MCOracle_test), index=Summary_pred_Qual_models_test.index)

# Append Deep Gaussian Network Performance
## Train
Summary_pred_Qual_models["DNM"] = pd.Series((W1_Errors_DNM_train), index=Summary_pred_Qual_models.index)
## Test
Summary_pred_Qual_models_test["DNM"] = pd.Series((W1_Errors_DNM_test), index=Summary_pred_Qual_models_test.index)

# Rename
PredictivePerformance_Metrics_Train = Summary_pred_Qual_models
PredictivePerformance_Metrics_Test = Summary_pred_Qual_models_test

# Update Performance Metrics
## Train
PredictivePerformance_Metrics_Train.to_latex((results_tables_path_final+str(f_unknown_mode)+"Problemdimension"+str(problem_dim)+"__SUMMARY_METRICS.tex"))
## Test
PredictivePerformance_Metrics_Test.to_latex((results_tables_path_final+str(f_unknown_mode)+"Problemdimension"+str(problem_dim)+"__SUMMARY_METRICS_test.tex"))

print("------------------------------------------------")
print("Updated Performance Metrics Dataframe and Saved!")
print("------------------------------------------------")


# ## Write Which Kernel Was Used 
# *(to count parameters by hand for GPR)...*

# In[ ]:


print("Kernel_Used_in_GPR: "+str(GPR_trash.kernel))
# Write Kernel Information to File
path_kernel_facts = (results_tables_path_final+"KERNELUSEDFORGRP_Latent_Width_NSDE"+str(width)+"Problemdimension"+str(problem_dim)+"__KernelUsed.tex")
text_file = open(path_kernel_facts, "w")
text_file.write("Kernel_Used_in_GPR: "+str(GPR_trash.kernel))
text_file.close()


# ---
# # Fin
# ---
