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


# # Get OT Comparer

# In[ ]:


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
def transport_dist(x_source,w_source,x_sink,w_sink,output_dim,OT_method="Sliced"):
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
                                                    seed = 2020)
            # COERSION
            OT_out = float(OT_out)
    # Return (regularized?) Transport Distance
    return OT_out


# ## One-Dimensional Error Metrics

# In[ ]:


def get_deterministic_errors(X_inputs, mean_predictions,Y_targets,N_Bootstraps=10):
    print("#------------#")
    print(" Get Error(s) ")
    print("#------------#")
    for i in tqdm(range((mean_predictions.shape[0]))):    
        # Compute Error(s)
        ## W1
        if output_dim > 1:
            W1_loop = ot.sliced.sliced_wasserstein_distance(X_s = mean_predictions[i,].reshape(1,-1),
                                                            X_t = (Y_targets[i,]))
            ## M1
            Mu_hat = mean_predictions[i,]
            Mu_MC = np.mean(Y_targets[i,],axis=0)
            Mu = Mu_MC
            
        else:
            W1_loop = ot.emd2_1d(np.array([mean_predictions[i]]),
                                     np.array(Y_targets[i,]).reshape(-1,),
                                     np.ones(1),
                                     empirical_weights)

            ## M1
            Mu_hat = mean_predictions[i]
            Mu_MC = np.mean(np.array(Y_targets[i,]))
            ###
            if f_unknown_mode == "Heteroskedastic_NonLinear_Regression":
                Mu = direct_facts[i,]
            else:
                Mu = Mu_MC
            ### Error(s)
        Mean_loop = np.sum(np.abs((Mu_hat-Mu)))
        Mean_loop_MC = np.sum(np.abs((Mu_hat-Mu_MC)))
        # Update
        if i == 0:
            W1_errors = W1_loop
            # Moments
            ## DNM
            Mean_errors =  Mean_loop
            
        else:
            W1_errors = np.append(W1_errors,W1_loop)
            # Moments
            ## DNM
            Mean_errors =  np.append(Mean_errors,Mean_loop)
            
            
    # Compute Error Metrics with Bootstrapped Confidence Intervals
    W1_Errors = np.array(bootstrap(np.abs(W1_errors),n=N_Bootstraps)(.95))
    Mean_Errors = np.array(bootstrap(np.abs(Mean_errors),n=N_Bootstraps)(.95))
    
    print("#-----------------#")
    print(" Get Error(s): END ")
    print("#-----------------#")
    return W1_Errors, Mean_Errors

