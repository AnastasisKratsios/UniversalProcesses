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
# X_T = x + \int_0^T \alpha(s,x)ds + \int_0^T\beta(s,x)dW_s.
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

# ### Quantization
# *This hyperparameter describes the proportion of the data used as sample-barycenters.*

# In[1]:


Quantization_Proportion = 1


# ### Simulation

# In[2]:


## Monte-Carlo
N_Euler_Maruyama_Steps = 100
N_Monte_Carlo_Samples = 10**5

N_Monte_Carlo_Samples_Test = 1000 # How many MC-samples to draw from test-set?
T_end = 1
Direct_Sampling = False #This hyperparameter determines if we use a Euler-Maryama scheme or if we use something else.  

## Grid
N_Grid_Finess = 100
Max_Grid = 1


# **Note**: Setting *N_Quantizers_to_parameterize* prevents any barycenters and sub-sampling.

# #### Mode: Code-Testin Parameter(s)

# In[3]:


trial_run = True


# ### Meta-parameters

# In[4]:


# Test-size Ratio
test_size_ratio = .75


# ## SDE Simulation Hyper-Parameter(s)

# ### Drift

# In[5]:


def alpha(t,x):
    return np.sin(t*math.pi) + x


# ### Volatility

# In[6]:


def beta(t,x):
    return 1


# ### Visualization

# In[7]:


# How many random polulations to visualize:
Visualization_Size = 4


# ---
# # Run Backend:
# *Runs the backend with these pre-specified hyperparameters*.

# In[8]:


# Use is user prefers GUI
# %run Universal_Processes_Trainer_BACKEND.ipynb
# Use for faster/shorter execution times (note: more amenable to terminal)
exec(open('Universal_Processes_Trainer_BACKEND.py').read())


# ---
