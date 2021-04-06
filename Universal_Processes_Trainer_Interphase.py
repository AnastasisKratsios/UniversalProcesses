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

# In[1]:


# Option 1:
groud_truth = "rSDE"
# Option 2:
# groud_truth = "2lnflow"
## Option 3:
# groud_truth = "pfBM"


# #### Grid Hyperparameter(s)

# In[2]:


## Monte-Carlo
N_Euler_Maruyama_Steps = 100
N_Monte_Carlo_Samples = 10**2
N_Monte_Carlo_Samples_Test = 10**3 # How many MC-samples to draw from test-set?

# End times for Time-Grid
T_end = 1
T_end_test = 1.1


## Grid
N_Grid_Finess = 100
Max_Grid = 0.5
x_0 = 1

# Number of Centers (\hat{\mu}_s)
N_Quantizers_to_parameterize = 10


# This option sets $\delta$ in $B_{\mathbb{R}\times [0,\infty)}(\hat{x}_n,\delta)$; where $\hat{x}_n\in \nu_{\cdot}^{-1}[\hat{\mu}]$.  N_measures_per_center sets the number of samples to draw in this ball...by construction the training set is $\delta$-bounded and $\nu_{(x,t)}$, for any such $x$ is $\omega_{\nu_{\cdot}}(\delta)$-bounded in $\mathcal{P}_1(\mathbb{R})$.

# In[3]:


# Hyper-parameters of Cover
delta = 0.1
N_measures_per_center = 100


# **Note**: Setting *N_Quantizers_to_parameterize* prevents any barycenters and sub-sampling.

# ### Random Cover
# This is not an active option!

# In[4]:


# Set Minibatch Size
# Random_Cover_Mini_Batch_Size = 100


# #### Mode: Code-Testin Parameter(s)
# - True: $\Rightarrow$ cross validation through grid of very mild parameters just to test hardward or software.
# - False: $\Rightarrow$ run CV-grid.

# In[5]:


trial_run = True


# ### Meta-parameters
# Ratio $\frac{\text{Testing Datasize}}{\text{Training Datasize}}$.

# In[6]:


test_size_ratio = .25


# ## Simulation from Rough SDE
# Simulate via Euler-M method from:
# $$ 
# X_T^x = x + \int_0^T \alpha(s,X_s^x)ds + \int_0^T((1-\eta)\beta(s,X_s^s)+\eta\sigma_s^H)dW_s.
# $$

# ### Drift

# In[7]:


def alpha(t,x):
    return .1*(.1-.5*(.01**2))*t #+ np.cos(x)


# ### Volatility

# In[8]:


def beta(t,x):
    return 0.01#+t*np.cos(x)


# ### Roughness Meta-parameters
#  - Roughness is $H$,
#  - Ratio_fBM_to_typical_vol is $\eta$.

# In[9]:


Rougnessa = 0.9 # Hurst Parameter
Ratio_fBM_to_typical_vol = 0 # $\eta$ in equation above.


# ## Simulation from Measure-Valued $2$-Parameter Log-Gaussian Flow
# $$
# X_{t,x} \sim \log\text{-}\mathcal{N}\left(\alpha(t,x),\beta(t,x)\right).
# $$

# **Note:** *$\alpha$ and $\beta$ are specified below in the SDE Example*.

# In[10]:


# Run Backend
# get_ipython().run_line_magic('run', 'Universal_Processes_Trainer_BACKEND.ipynb')
exec(open('Universal_Processes_Trainer_BACKEND.py').read())


# # Visualization
# *From hereon out...do nothing and just let the backend sript run...the images and tables will load :).*

# In[11]:


# Run Backend
# %run Universal_Processes_Trainer_Visuals.ipynb
exec(open('Universal_Processes_Trainer_Visuals.py').read())


# ## Update User

# ### Training-Set Performance

# In[ ]:


Type_A_Prediction


# ### Test-Set Performance

# In[ ]:


Type_A_Prediction_test


# ---

# ---
# # Fin
# ---

# ---
