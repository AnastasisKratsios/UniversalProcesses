Quantization_Proportion = 1
### Simulation
## Monte-Carlo
N_Euler_Maruyama_Steps = 100
N_Monte_Carlo_Samples = 10**5
N_Monte_Carlo_Samples_Test = 1000 # How many MC-samples to draw from test-set?
T_end = 1
Direct_Sampling = False #This hyperparameter determines if we use a Euler-Maryama scheme or if we use something else.  
## Grid
N_Grid_Finess = 100
Max_Grid = 1
## CV-Search
trial_run = True
### Meta-parameters
# Test-size Ratio
test_size_ratio = .75
## SDE Simulation Hyper-Parameter(s)
### Drift
def alpha(t,x):
    return np.sin(t*math.pi) + x
### Volatility
def beta(t,x):
    return 1
# How many random polulations to visualize:
Visualization_Size = 4


# RUN
exec(open('Universal_Processes_Trainer_BACKEND.py').read())
