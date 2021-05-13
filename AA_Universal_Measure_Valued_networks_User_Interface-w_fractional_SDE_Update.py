#!/usr/bin/env python
# coding: utf-8

# # Deep Universal Regular Conditional Expectations:
# 
# ---
# This implements the universal deep neural model of $\mathcal{NN}_{1_{\mathbb{R}^n},\mathcal{D}}^{\sigma:\star}$ [Anastasis Kratsios](https://people.math.ethz.ch/~kratsioa/) - 2021.
# 
# ---
# 
# ## What does this code do?
# 1. Learn Heteroskedastic Non-Linear Regression Problem
#      - $Y\sim f_{\text{unkown}}(x) + \epsilon$ where $f$ is an known function and $\epsilon\sim Laplace(0,\|x\|)$
# 2. Learn Random Bayesian Network's Law:
#     - $Y = W_J Y^{J-1}, \qquad Y^{j}\triangleq \sigma\bullet A^{j}Y^{j-1} + b^{j}, \qquad Y^0\triangleq x$
# 
# 3. In the above example if $A_j = M_j\odot \tilde{A_j}$ where $\tilde{A}_j$ is a deterministic matrix and $M_j$ is a "mask", that is, a random matrix with binary entries and $\odot$ is the Hadamard product then we recover the dropout framework.
# 4. Learn the probability distribution that the unique strong solution to the rough SDE with uniformly Lipschitz drivers driven by a factional Brownian motion with Hurst exponent $H \in [\frac1{2},1)$:
# $$
# X_t^x = x + \int_0^t \alpha(s,X_s^x)ds + \int_0^t \beta(s,X_s^x)dB_s^H
# $$
# belongs, at time $t=1$, to a ball about the initial point $x$ of random radius given by an independant exponential random-variable with shape parameter $\lambda=2$
# 5. Train a DNN to predict the returns of bitcoin with GD.  Since this has random initialization then each prediction of a given $x$ is stochastic...We learn the distribution of this conditional RV (conditioned on x in the input space).
# $$
# Y_x \triangleq \hat{f}_{\theta_{T}}(x), \qquad \theta_{(t+1)}\triangleq \theta_{(t)} + \lambda \sum_{x \in \mathbb{X}} \nabla_{\theta}\|\hat{f}_{\theta_t}(x) - f(x)\|, \qquad \theta_0 \sim N_d(0,1);
# $$
# $T\in \mathbb{N}$ is a fixed number of "SGD" iterations (typically identified by cross-validation on a single SGD trajectory for a single initialization) and where $\theta \in \mathbb{R}^{(d_{J}+1)+\sum_{j=0}^{J-1} (d_{j+1}d_j + 1)}$ and $d_j$ is the dimension of the "bias" vector $b_j$ defining each layer of the DNN with layer dimensions:
# $$
# \hat{f}_{\theta}(x)\triangleq A^{(J)}x^{(J)} + b^{(J)},\qquad x^{(j+1)}\triangleq \sigma\bullet A^{j}x^{(j)} + b^{j},\qquad x^{(0)}\triangleq x
# .
# $$
# 
# 6. Extreme Learning Machines: 
#     Just like the Bayesian network but then last layer is trained on the training set using KRidge!

# #### Mode:
# Software/Hardware Testing or Real-Deal?

# In[1]:


trial_run = True


# ### Simulation Method:

# In[2]:


# Random DNN
# f_unknown_mode = "Heteroskedastic_NonLinear_Regression"

# Random DNN internal noise
## Real-world data version
# f_unknown_mode = "Extreme_Learning_Machine"
### General Parameters
# activation_function == 'thresholding'
activation_function = 'sigmoid'
### Dataset Option 1
dataset_option = 'SnP'
### Dataset Option 2
# dataset_option = 'crypto'
Depth_Bayesian_DNN = 1
N_Random_Features = 10**2
## Simulated Data version
# f_unknown_mode = "DNN_with_Random_Weights"
width = 10**2

# Random Dropout applied to trained DNN
# f_unknown_mode = "DNN_with_Bayesian_Dropout"
Dropout_rate = 0.75

# GD with Randomized Input
# f_unknown_mode = "GD_with_randomized_input"
# GD_epochs = 50

# SDE with fractional Driver
f_unknown_mode = "Rough_SDE"
N_Euler_Steps = 50
Hurst_Exponent = 0.9

f_unknown_mode = "Rough_SDE_Vanilla"
## Define Process' dynamics in (2) cell(s) below.


# ---
# #### More Meta-Parameters for "Vanilla" fractional SDE

# In[3]:


## Monte-Carlo
N_Euler_Maruyama_Steps = 10**2

# End times for Time-Grid
T_end = 1
T_end_test = 1.1


## Grid
N_Grid_Finess = 1
Max_Grid = 0.5
x_0 = 1

# Number of Centers (\hat{\mu}_s)
N_Quantizers_to_parameterize = 1
N_Clusters = 2
Hurst_Exponent = 0.5
uniform_noise_level = 0

# Hyper-parameters of Cover
delta = 0.1
N_measures_per_center = 10**2


# #### Vanilla Drift

# In[4]:


def alpha(t,x):
    return .1*np.ones(problem_dim)#(.1-.5*(.01**2))*t + np.cos(x)


# #### Vanilla Vol

# In[5]:


def beta(t,x):
    return 0.01*np.ones(problem_dim)


# ---

# ## Problem Dimension

# In[6]:


problem_dim = 2
if f_unknown_mode != 'Extreme_Learning_Machine':
    width = int(2*(problem_dim+1))


# #### Vanilla fractional SDE:
# If f_unknown_mode == "Rough_SDE_Vanilla" is selected, then we can specify the process's dynamics.  

# In[7]:


# Depricated
# #--------------------------#
# # Define Process' Dynamics #
# #--------------------------#
# drift_constant = 0.1
# volatility_constant = 0.01

# # Define DNN Applier
# def f_unknown_drift_vanilla(x):
# #     x_internal = drift_constant*np.ones(problem_dim)
#     x_internal = drift_constant*np.sin(x)
#     return x_internal
# def f_unknown_vol_vanilla(x):
# #     x_internal = volatility_constant*np.diag(np.ones(problem_dim))
#     x_internal = volatility_constant*np.diag(np.cos(x))
#     return x_internal


# ## Note: *Why the procedure is so computationally efficient*?
# ---
#  - The sample barycenters do not require us to solve for any new Wasserstein-1 Barycenters; which is much more computationally costly,
#  - Our training procedure never back-propages through $\mathcal{W}_1$ since steps 2 and 3 are full-decoupled.  Therefore, training our deep classifier is (comparatively) cheap since it takes values in the standard $N$-simplex.
# 
# ---

# #### Grid Hyperparameter(s)
# - Ratio $\frac{\text{Testing Datasize}}{\text{Training Datasize}}$.
# - Number of Training Points to Generate

# In[8]:


train_test_ratio = .1
N_train_size = 10**3


# Monte-Carlo Paramters

# In[9]:


## Monte-Carlo
N_Monte_Carlo_Samples = 10**3
N_Monte_Carlo_Samples_Test = 10**2 # How many MC-samples to draw from test-set?


# Initial radis of $\delta$-bounded random partition of $\mathcal{X}$!

# In[10]:


# Hyper-parameters of Cover
delta = 0.1
Proportion_per_cluster = .75


# ## Dependencies and Auxiliary Script(s)

# In[11]:


# %run Loader.ipynb
exec(open('Loader.py').read())
# Load Packages/Modules
exec(open('Init_Dump.py').read())
import time as time #<- Note sure why...but its always seems to need 'its own special loading...'


# # Simulate or Parse Data

# In[12]:


#-------------------------------------------------------#
print("Generating/Prasing Data and Training MC-Oracle")
#-------------------------------------------------------#
if (f_unknown_mode != 'Rough_SDE') and (f_unknown_mode != 'Rough_SDE_Vanilla'):
    # %run Data_Simulator_and_Parser.ipynb
    exec(open('Data_Simulator_and_Parser.py').read())
else:
    # Renaming Some internal Parameter(s)
    groud_truth = "rSDE"
    test_size_ratio = train_test_ratio
    Ratio_fBM_to_typical_vol = 1
    output_dim = problem_dim
    T_begin = 0
    T_end = 1
    # Run da code
    get_ipython().run_line_magic('run', 'Fractional_SDE/fractional_SDE_Simulator.ipynb')
    get_ipython().run_line_magic('run', 'Fractional_SDE/Data_Simulator_and_Parser___fractional_SDE.ipynb')
#     exec(open('Fractional_SDE/fractional_SDE_Simulator.py').read())
#     exec(open('Fractional_SDE/Data_Simulator_and_Parser___fractional_SDE.py').read())

# Verbosity is nice
print("Generated Data:")
print("Number of Training Datums:"+str(X_train.shape[0]))
print("Number of Testing Datums:"+str(X_test.shape[0]))


# ### Rescale

# In[13]:


# Rescale
scaler = StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


# ## Run: Main (DNM) and Oracle Models

# In[14]:


print("------------------------------")
print("Running script for main model!")
print("------------------------------")
get_ipython().run_line_magic('run', 'Universal_Measure_Valued_Networks_Backend.ipynb')
# exec(open('Universal_Measure_Valued_Networks_Backend.py').read())
print("------------------------------------")
print("Done: Running script for main model!")
print("------------------------------------")


# ### Evaluate Main and Oracle Model's Performance

# In[15]:


get_ipython().run_line_magic('run', 'Universal_Measure_Valued_Networks_Backend_EVALUATOR.ipynb')
# exec(open(Universal_Measure_Valued_Networks_Backend_EVALUATOR.py).read())


# ## 1) *Pointmass Benchmark(s)*
# These benchmarks consist of subsets of $C(\mathbb{R}^d,\mathbb{R})$ which we lift to models in $C(\mathbb{R}^d,\cap_{1\leq q<\infty}\mathscr{P}_{q}(\mathbb{R}))$ via:
# $$
# \mathbb{R}^d \ni x \to f(x) \to \delta_{f(x)}\in \cap_{1\leq q<\infty}\mathcal{P}_{q}(\mathbb{R}).
# $$

# In[16]:


exec(open('CV_Grid.py').read())
# Notebook Mode:
# %run Evaluation.ipynb
# %run Benchmarks_Model_Builder_Pointmass_Based.ipynb
# Terminal Mode (Default):
exec(open('Evaluation.py').read())
exec(open('Benchmarks_Model_Builder_Pointmass_Based.py').read())


# # Summary of Point-Mass Regression Models

# #### Training Model Facts

# In[17]:


print(Summary_pred_Qual_models)
Summary_pred_Qual_models


# #### Testing Model Facts

# In[18]:


print(Summary_pred_Qual_models_test)
Summary_pred_Qual_models_test


# ## 2) *Gaussian Benchmarks*

# - Bencharm 1: [Gaussian Process Regressor](https://scikit-learn.org/stable/modules/gaussian_process.html)
# - Benchmark 2: Deep Gaussian Networks:
# These models train models which assume Gaussianity.  We may view these as models in $\mathcal{P}_2(\mathbb{R})$ via:
# $$
# \mathbb{R}^d \ni x \to (\hat{\mu}(x),\hat{\Sigma}(x)\hat{\Sigma}^{\top})\triangleq f(x) \in \mathbb{R}\times [0,\infty) \to 
# (2\pi)^{-\frac{d}{2}}\det(\hat{\Sigma}(x))^{-\frac{1}{2}} \, e^{ -\frac{1}{2}(\cdot - \hat{\mu}(x))^{{{\!\mathsf{T}}}} \hat{\Sigma}(x)^{-1}(\cdot - \hat{\mu}(x)) } \mu \in \mathcal{G}_d\subset \mathcal{P}_2(\mathbb{R});
# $$
# where $\mathcal{G}_1$ is the set of Gaussian measures on $\mathbb{R}$ equipped with the relative Wasserstein-1 topology.
# 
# Examples of this type of architecture are especially prevalent in uncertainty quantification; see ([Deep Ensembles](https://arxiv.org/abs/1612.01474)] or [NOMU: Neural Optimization-based Model Uncertainty](https://arxiv.org/abs/2102.13640).  Moreover, their universality in $C(\mathbb{R}^d,\mathcal{G}_2)$ is known, and has been shown in [Corollary 4.7](https://arxiv.org/abs/2101.05390).

# In[ ]:


# %run Jupyter_Notebooks/Jupyter_Notebooks_for_Final_Implementation/Benchmarks_Model_Builder_Mean_Var.ipynb
exec(open('Benchmarks_Model_Builder_Mean_Var.py').read())


# In[ ]:


print("Prediction Quality (Updated): Test")
print(Summary_pred_Qual_models_test)
Summary_pred_Qual_models_test


# In[ ]:


print("Prediction Quality (Updated): Train")
print(Summary_pred_Qual_models)
Summary_pred_Qual_models


# # 3) The natural Universal Benchmark: [Bishop's Mixture Density Network](https://publications.aston.ac.uk/id/eprint/373/1/NCRG_94_004.pdf)
# 
# This implementation is as follows:
# - For every $x$ in the trainingdata-set we fit a GMM $\hat{\nu}_x$, using the [Expectation-Maximization (EM) algorithm](https://en.wikipedia.org/wiki/Expectation%E2%80%93maximization_algorithm), with the same number of centers as the deep neural model in $\mathcal{NN}_{1_{\mathbb{R}^d},\mathcal{D}}^{\sigma:\star}$ which we are evaluating.  
# - A Mixture density network is then trained to predict the infered parameters; given any $x \in \mathbb{R}^d$.

# In[21]:


if output_dim == 1:
    # %run Mixture_Density_Network.ipynb
    exec(open('Mixture_Density_Network.py').read())


# ## Get Final Outputs
# Now we piece together all the numerical experiments and report a nice summary.

# ---
# # Final Results
# ---

# ## Prasing Quality Metric Results

# #### Finalizing Saving
# **Note:** *We do it in two steps since the grid sometimes does not want to write nicely...*

# In[ ]:


## Write Performance Metrics
### Incase caption breaks
Summary_pred_Qual_models.to_latex((results_tables_path+"/Final_Results/"+"Performance_metrics_Problem_Type_"+str(f_unknown_mode)+"Problemdimension"+str(problem_dim)+"__SUMMARY_METRICS.tex"),
                                 float_format="{:0.3g}".format)
text_file = open((results_tables_path+"/Final_Results/"+"ZZZ_CAPTION_Performance_metrics_Problem_Type_"+str(f_unknown_mode)+"Problemdimension"+str(problem_dim)+"__SUMMARY_METRICS___CAPTION.tex"), "w")
text_file.write("Quality Metrics; d:"+str(problem_dim)+", D:"+str(output_dim)+", Depth:"+str(Depth_Bayesian_DNN)+", Width:"+str(width)+", Dropout rate:"+str(Dropout_rate)+".")
text_file.close()


### Incase caption does not break
Summary_pred_Qual_models.to_latex((results_tables_path+"/Final_Results/"+"Performance_metrics_Problem_Type_"+str(f_unknown_mode)+"Problemdimension"+str(problem_dim)+"__SUMMARY_METRICS.tex"),
                                 caption=("Quality Metrics; d:"+str(problem_dim)+", D:"+str(output_dim)+", Depth:"+str(Depth_Bayesian_DNN)+", Width:"+str(width)+", Dropout rate:"+str(Dropout_rate)+"."),
                                 float_format="{:0.3g}".format)


# # For Terminal Runner(s):

# In[ ]:


# For Terminal Running
print("===================")
print("Predictive Quality:")
print("===================")
print(Summary_pred_Qual_models)
print("===================")
print(" ")
print(" ")
print(" ")
print("Kernel_Used_in_GPR: "+str(GPR_trash.kernel))
print("🙃🙃 Have a wonderful day! 🙃🙃")
Summary_pred_Qual_models


# ---
# # Fin
# ---

# ---