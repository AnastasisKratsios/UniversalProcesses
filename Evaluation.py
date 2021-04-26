#!/usr/bin/env python
# coding: utf-8

# # Evaluation Metrics and Scripts

# Initialize "Higher Moments" Loss Metric
# $$
# \sum_{k=1}^K \frac{|\sum_{x \in \mathbb{X}}x^k\hat{\nu}(x) - \mu_k|}{k!}
# $$

# In[ ]:


def Higher_Moments_Loss(Mu_hat_input,Mu_hat_MC_input):
    for k in range(10):
        moment_hat_loop = np.sum((Mu_hat_input**k)*points_of_mass)/np.math.factorial(k)
        moment_MC_hat_loop = np.mean(Mu_hat_MC_input**k)/np.math.factorial(k)
        if k == 0:
            moment_hat = moment_hat_loop
            moment_MC_hat = moment_MC_hat_loop
        else:
            moment_hat = np.append(moment_hat,moment_hat_loop)
            moment_MC_hat = np.append(moment_MC_hat,moment_MC_hat_loop)

    return moment_hat,moment_MC_hat


# ## One-Dimensional Error Metrics

# In[ ]:


def get_deterministic_errors(X_inputs, mean_predictions,Y_targets,N_Bootstraps=10):
    print("#------------#")
    print(" Get Error(s) ")
    print("#------------#")
    for i in tqdm(range((mean_predictions.shape[0]))):    
        # Compute Error(s)
        ## W1
        W1_loop = ot.emd2_1d(np.array([mean_predictions[i]]),
                             np.array(Y_targets[i,]).reshape(-1,),
                             np.ones(1),
                             empirical_weights)

        ## M1
        Mu_hat = mean_predictions[i]
        Mu_MC = np.mean(np.array(Y_targets[i,]))
        if f_unknown_mode == "Heteroskedastic_NonLinear_Regression":
            Mu = direct_facts[i,]
        else:
            Mu = Mu_MC

        ### Error(s)
        Mean_loop = (Mu_hat-Mu)
        Mean_loop_MC = (Mu_hat-Mu_MC)

        ## Variance
        Var_hat = np.sum((((mean_predictions[i])-Mu_hat)**2)*b)
        Var_MC = np.mean(np.array(Y_targets[i]-Mu_MC)**2)
        if f_unknown_mode == "Heteroskedastic_NonLinear_Regression":
            Var = 2*np.sum(X_inputs[i,]**2)
        else:
            Var = Var_MC     
        ### Error(s)
        Var_loop = np.abs(Var_hat-Var)
        Var_loop_MC = np.abs(Var_MC-Var)

        # Skewness
        Skewness_hat = np.sum(((((mean_predictions[i])-Mu_hat)/Var)**3)*b)
        Skewness_MC = np.mean((np.array(Y_targets[i]-Mu_MC)/Var_MC)**3)
        if f_unknown_mode == "Heteroskedastic_NonLinear_Regression":
            Skewness = 0
        else:
            Skewness = Skewness_MC
        ### Error(s)
        Skewness_loop = np.abs(Skewness_hat-Skewness)
        Skewness_loop_MC = np.abs(Skewness_MC-Skewness)

        # Skewness
        Ex_Kurtosis_hat = np.sum(((((mean_predictions[i])-Mu_hat)/Var)**4)*b) - 3
        Ex_Kurtosis_MC = np.mean((np.array(Y_targets[i]-Mu_MC)/Var_MC)**4) - 3
        if f_unknown_mode == "Heteroskedastic_NonLinear_Regression":
            Ex_Kurtosis = 3
        else:
            Ex_Kurtosis = Ex_Kurtosis_MC
        ### Error(s)
        Ex_Kurtosis_loop = np.abs(Ex_Kurtosis-Ex_Kurtosis_hat)
        Ex_Kurtosis_loop_MC = np.abs(Ex_Kurtosis-Ex_Kurtosis_MC)



        # Get Higher Moments Loss
        Higher_momentserrors_loop,Higher_MC_momentserrors_loop = Higher_Moments_Loss(b,np.array(Y_targets[i,]))
        Higher_Moments_Errors_loop = np.abs(Higher_momentserrors_loop-Higher_MC_momentserrors_loop)


        # Update
        if i == 0:
            W1_errors = W1_loop
            # Moments
            ## DNM
            Mean_errors =  Mean_loop
            Var_errors = Var_loop
            Skewness_errors = Skewness_loop
            Ex_Kurtosis_errors = Ex_Kurtosis_loop
            # Higher Moments
            Higher_Moments_Errors = Higher_Moments_Errors_loop


        else:
            W1_errors = np.append(W1_errors,W1_loop)
            # Moments
            ## DNM
            Mean_errors =  np.append(Mean_errors,Mean_loop)
            Var_errors = np.append(Var_errors,Var_loop)
            Skewness_errors = np.append(Skewness_errors,Skewness_loop)
            Ex_Kurtosis_errors = np.append(Ex_Kurtosis_errors,Ex_Kurtosis_loop)
            # Higher Moments
            Higher_Moments_Errors = np.append(Higher_Moments_Errors,Higher_Moments_Errors_loop)

            
    # Compute Error Metrics with Bootstrapped Confidence Intervals
    W1_Errors = np.array(bootstrap(np.abs(W1_errors),n=N_Bootstraps)(.95))
    Mean_Errors = np.array(bootstrap(np.abs(Mean_errors),n=N_Bootstraps)(.95))
    Var_Errors = np.array(bootstrap(np.abs(Var_errors),n=N_Bootstraps)(.95))
    Skewness_Errors = np.array(bootstrap(np.abs(Skewness_errors),n=N_Bootstraps)(.95))
    Ex_Kurtosis_Errors = np.array(bootstrap(np.abs(Ex_Kurtosis_errors),n=N_Bootstraps)(.95))
    Higher_Moment_Errors = np.array(bootstrap(np.abs(Higher_Moments_Errors),n=N_Bootstraps)(.95))
    
    # Format Error Metrics
    output = np.array([W1_Errors,Mean_Errors,Var_Errors,Skewness_Errors,Ex_Kurtosis_Errors,Higher_Moment_Errors])
    print("#-----------------#")
    print(" Get Error(s): END ")
    print("#-----------------#")
    return output


# ### Get Loss Statistics
