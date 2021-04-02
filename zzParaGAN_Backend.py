#!/usr/bin/env python
# coding: utf-8

# ---

# ---
# # Backend
# ---

# In[19]:


#------------------------------------------------------------------------------------------------#
#                                      Define Predictive Model                                   #
#------------------------------------------------------------------------------------------------#

def def_simple_deep_classifer(height, depth, learning_rate, input_dim, output_dim):
    # Initialize Simple Deep Classifier
    simple_deep_classifier = tf.keras.Sequential()
    for d_i in range(depth):
        simple_deep_classifier.add(tf.keras.layers.Dense(height, activation='relu'))

    simple_deep_classifier.add(tf.keras.layers.Dense(output_dim, activation='softmax'))    
    
    # Compile Simple Deep Classifier
    simple_deep_classifier.compile(optimizer='adam',
                  loss="mae",#loss = categorical_crossentropy,
                  metrics=['accuracy'])
    
    # Return Output
    return simple_deep_classifier

#------------------------------------------------------------------------------------------------#
#                                  Build Deep Classifier Model                                   #
#------------------------------------------------------------------------------------------------#
from tensorflow.keras import Sequential
def build_simple_deep_classifier(n_folds , n_jobs, n_iter, param_grid_in, X_train, y_train,X_test):

    # Deep Feature Network
    CV_simple_deep_classifier = tf.keras.wrappers.scikit_learn.KerasRegressor(build_fn=def_simple_deep_classifer, verbose=True)
    
    # Randomized CV
    CV_simple_deep_classifier_CVer = RandomizedSearchCV(estimator=CV_simple_deep_classifier, 
                                    n_jobs=n_jobs,
                                    cv=KFold(n_folds, random_state=2020, shuffle=True),
                                    param_distributions=param_grid_in,
                                    n_iter=n_iter,
                                    return_train_score=True,
                                    random_state=2020,
                                    verbose=10)
    
    # Fit
    CV_simple_deep_classifier_CVer.fit(X_train,y_train)

    # Make Prediction(s)
    predicted_classes_train = CV_simple_deep_classifier_CVer.predict(X_train)
    predicted_classes_test = CV_simple_deep_classifier_CVer.predict(X_test)
    
    # Counter number of parameters #
    #------------------------------#
    # Extract Best Model
    best_model = CV_simple_deep_classifier_CVer.best_estimator_
    # Count Number of Parameters
    N_params_best_classifier = np.sum([np.prod(v.get_shape().as_list()) for v in best_model.model.trainable_variables])

    
    # Return Values
    return predicted_classes_train, predicted_classes_test, N_params_best_classifier

# Update User
#-------------#
print('Deep Classifier - Ready')


# -----
# ----
# ---

# In[ ]:


#------------------------------------------------------------------------------------------------#
#                                      Define Predictive Model                                   #
#------------------------------------------------------------------------------------------------#
def get_deep_classifer(height, depth, learning_rate, input_dim, output_dim):
    #----------------------------#
    # Maximally Interacting Layer #
    #-----------------------------#
    # Initialize Inputs
    input_layer = tf.keras.Input(shape=(input_dim,))
   
    
    #------------------#
    #   Core Layers    #
    #------------------#
    core_layers = fullyConnected_Dense(height)(input_layer)
    # Activation
    core_layers = tf.nn.swish(core_layers)
    # Train additional Depth?
    if depth>1:
        # Add additional deep layer(s)
        for depth_i in range(1,depth):
            core_layers = fullyConnected_Dense(height)(core_layers)
            # Activation
            core_layers = tf.nn.relu(core_layers)
    
    #------------------#
    #  Readout Layers  #
    #------------------# 
    # Affine (Readout) Layer (Dense Fully Connected)
    readout_layer = fullyConnected_Dense(output_dim)(core_layers)  
    output_layers = tf.nn.softmax(readout_layer)  
    # Define Input/Output Relationship (Arch.)
    trainable_layers_model = tf.keras.Model(input_layer, output_layers)
    
    
    #----------------------------------#
    # Define Optimizer & Compile Archs.
    #----------------------------------#
    opt = Adam(lr=learning_rate)
    trainable_layers_model.compile(optimizer=opt, 
                                   loss="binary_crossentropy", 
                                   metrics=["accuracy", "mae"])

    return trainable_layers_model


# In[9]:


def build_simple_deep_classifier(n_folds , n_jobs, n_iter, param_grid_in, X_train, y_train,X_test):
    # Update Dictionary
    param_grid_in_internal = param_grid_in
    param_grid_in_internal['input_dim'] = [1]#[(X_train.shape[1])]
    
    # Deep Feature Network
    ffNN_CV = tf.keras.wrappers.scikit_learn.KerasRegressor(build_fn=get_deep_classifer, 
                                                            verbose=True)
    
    # Randomized CV
    ffNN_CVer = RandomizedSearchCV(estimator=ffNN_CV, 
                                    n_jobs=n_jobs,
                                    cv=KFold(n_folds, random_state=2020, shuffle=True),
                                    param_distributions=param_grid_in_internal,
                                    n_iter=n_iter,
                                    return_train_score=True,
                                    random_state=2020,
                                    verbose=10)
    
    # Fit Model #
    #-----------#
    ffNN_CVer.fit(X_train,y_train)

    # Write Predictions #
    #-------------------#
    y_hat_train = ffNN_CVer.predict(X_train)
    y_hat_test = ffNN_CVer.predict(X_test)
    
    # Counter number of parameters #
    #------------------------------#
    # Extract Best Model
    best_model = ffNN_CVer.best_estimator_
    # Count Number of Parameters
    N_params_best_ffNN = np.sum([np.prod(v.get_shape().as_list()) for v in best_model.model.trainable_variables])
    
    
    #-----------------#
    # Save Full-Model #
    #-----------------#
    print('Benchmark-Model: Saving')
#     joblib.dump(best_model, './outputs/models/Benchmarks/ffNN_trained_CV.pkl', compress = 1)
#     ffNN_CVer.best_params_['N_Trainable_Parameters'] = N_params_best_ffNN
#     pd.DataFrame.from_dict(ffNN_CVer.best_params_,orient='index').to_latex("./outputs/models/Benchmarks/Best_Parameters.tex")
    print('Benchmark-Model: Saved')
    
    # Return Values #
    #---------------#
    return y_hat_train, y_hat_test, N_params_best_ffNN

# Update User
#-------------#
print('Deep Feature Builder - Ready')


# # Data Generation

# Generates the empirical measure $\sum_{n=1}^N \delta_{X_T(\omega_n)}$ of $X_T$ conditional on $X_0=x_0\in \mathbb{R}$ *($x_0$ and $T>0$ are user-provided)*.

# In[ ]:


def Euler_Maruyama_Generator(x_0,
                             N_Euler_Maruyama_Steps = 100,
                             N_Monte_Carlo_Samples = 100,
                             T = 1): 
    
    #----------------------------#    
    # DEFINE INTERNAL PARAMETERS #
    #----------------------------#
    # Initialize Empirical Measure
    X_T_Empirical = np.zeros(N_Monte_Carlo_Samples)


    # Internal Initialization(s)
    ## Initialize current state
    n_sample = 0
    ## Initialize Incriments
    dt = T/N_Euler_Maruyama_Steps
    sqrt_dt = np.sqrt(dt)

    #-----------------------------#    
    # Generate Monte-Carlo Sample #
    #-----------------------------#
    while n_sample < N_Monte_Carlo_Samples:
        # Reset Step Counter
        t = 1
        # Initialize Current State 
        X_current = x_0
        # Perform Euler-Maruyama Simulation
        while t<N_Euler_Maruyama_Steps:
            # Update Internal Parameters
            ## Get Current Time
            t_current = t*(T/N_Euler_Maruyama_Steps)

            # Update Generated Path
            X_current = X_current + alpha(t_current,X_current)*dt + beta(t_current,X_current)*np.random.normal(0,sqrt_dt)

            # Update Counter (EM)
            t = t+1

        # Update Empirical Measure
        X_T_Empirical[n_sample] = X_current

        # Update Counter (MC)
        n_sample = n_sample + 1

    return X_T_Empirical#.reshape(1,-1)


# ---
