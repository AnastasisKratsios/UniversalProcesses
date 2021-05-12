#!/usr/bin/env python
# coding: utf-8

# # Helper Function(s)
# A little list of useful helper functions when building the architope!

# In[ ]:


# MAPE, between 0 and 100
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)

    y_true.shape = (y_true.shape[0], 1)
    y_pred.shape = (y_pred.shape[0], 1)

    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


# # Deep Learning Helper(s)

# ## Custom Layers
#  - Fully Conneted Dense: Typical Feed-Forward Layer
#  - Fully Connected Dense Invertible: Necessarily satisfies for input and output layer(s)
#  - Fully Connected Dense Destructor: Violates Assumptions for both input and ouput layer(s) (it is neither injective nor surjective)

# In[ ]:


class fullyConnected_Dense(tf.keras.layers.Layer):

    def __init__(self, units=16, input_dim=32):
        super(fullyConnected_Dense, self).__init__()
        self.units = units

    def build(self, input_shape):
        self.w = self.add_weight(name='Weights_ffNN',
                                 shape=(input_shape[-1], self.units),
                               initializer='random_normal',
                               trainable=True)
        self.b = self.add_weight(name='bias_ffNN',
                                 shape=(self.units,),
                               initializer='random_normal',
                               trainable=True)

    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b
    
class fullyConnected_Dense_Invertible(tf.keras.layers.Layer):

    def __init__(self, units=16, input_dim=32):
        super(fullyConnected_Dense_Invertible, self).__init__()
        self.units = units

    def build(self, input_shape):
        self.w = self.add_weight(name='Weights_ffNN',
                                 shape=(input_shape[-1], input_shape[-1]),
                               initializer='zeros',
                               trainable=True)
        self.b = self.add_weight(name='bias_ffNN',
                                 shape=(self.units,),
                               initializer='zeros',
                               trainable=True)

    def call(self, inputs):
        expw = tf.linalg.expm(self.w)
        return tf.matmul(inputs, expw) + self.b


# In[ ]:


#------------------------------------------------------------------------------------------------#
#                                      Define Predictive Model                                   #
#------------------------------------------------------------------------------------------------#

def def_trainable_layers_Nice_Input_Output(height, depth, learning_rate, input_dim, output_dim):
    #----------------------------#
    # Maximally Interacting Layer #
    #-----------------------------#
    # Initialize Inputs
    input_layer = tf.keras.Input(shape=(input_dim,))
    
    
    #------------------#
    # Deep Feature Map #
    #------------------#
    # For this implementation we do not use a "deep feature map!"
#     if Depth_Feature_Map >0:
#         for i_feature_depth in range(Depth_Feature_Map):
#             # First Layer
#             if i_feature_depth == 0:
#                 deep_feature_map = fullyConnected_Dense_Invertible(input_dim)(input_layer)
#                 deep_feature_map = tf.nn.leaky_relu(deep_feature_map)
#             else:
#                 deep_feature_map = fullyConnected_Dense_Invertible(input_dim)(deep_feature_map)
#                 deep_feature_map = tf.nn.leaky_relu(deep_feature_map)
#     else:
#         deep_feature_map = input_layer
        
    
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
            core_layers = tf.nn.swish(core_layers)
    
    #------------------#
    #  Readout Layers  #
    #------------------# 
    # Affine (Readout) Layer (Dense Fully Connected)
    output_layers = fullyConnected_Dense(output_dim)(core_layers)  
    # Define Input/Output Relationship (Arch.)
    trainable_layers_model = tf.keras.Model(input_layer, output_layers)
    
    
    #----------------------------------#
    # Define Optimizer & Compile Archs.
    #----------------------------------#
    opt = Adam(lr=learning_rate)
    trainable_layers_model.compile(optimizer=opt, loss="mae", metrics=["mse", "mae"])

    return trainable_layers_model

#------------------------------------------------------------------------------------------------#
#                                      Build Predictive Model                                    #
#------------------------------------------------------------------------------------------------#

def build_ffNN(n_folds , n_jobs, n_iter, param_grid_in, X_train, y_train, X_test_partial,X_test):

    # Deep Feature Network
    Nice_Model_CV = tf.keras.wrappers.scikit_learn.KerasRegressor(build_fn=def_trainable_layers_Nice_Input_Output, verbose=True)
    
    # Randomized CV
    Nice_Model_CVer = RandomizedSearchCV(estimator=Nice_Model_CV, 
                                    n_jobs=n_jobs,
                                    cv=KFold(n_folds, random_state=2020, shuffle=True),
                                    param_distributions=param_grid_in,
                                    n_iter=n_iter,
                                    return_train_score=True,
                                    random_state=2020,
                                    verbose=10)
    
    # Fit Model #
    #-----------#
    Nice_Model_CVer.fit(X_train,y_train)

    # Write Predictions #
    #-------------------#
    y_hat_train = Nice_Model_CVer.predict(X_test_partial)
    y_hat_test = Nice_Model_CVer.predict(X_test)
    
    # Counter number of parameters #
    #------------------------------#
    # Extract Best Model
    best_model = Nice_Model_CVer.best_estimator_
    # Count Number of Parameters
    N_params_best_ffNN = np.sum([np.prod(v.get_shape().as_list()) for v in best_model.model.trainable_variables])
    
    # Return Values #
    #---------------#
    return y_hat_train, y_hat_test, N_params_best_ffNN

# Update User
#-------------#
print('Deep Feature Builder - Ready')



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
                  loss='categorical_crossentropy',
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
    
    # Scaler
    scaler = MinMaxScaler()#StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.fit_transform(X_test)
    
    # Randomized CV
    CV_simple_deep_classifier_CVer = RandomizedSearchCV(estimator=CV_simple_deep_classifier, 
                                                        n_jobs=n_jobs,
                                                        cv=KFold(n_folds, random_state=2021, shuffle=True),
                                                        param_distributions=param_grid_in,
                                                        n_iter=n_iter,
                                                        return_train_score=True,
                                                        random_state=2020,
                                                        verbose=10)
    
    # Fit
    CV_simple_deep_classifier_CVer.fit(X_train_scaled,y_train)

    # Make Prediction(s)
    prediction_timer_train = time.time()
    predicted_classes_train = CV_simple_deep_classifier_CVer.predict(X_train_scaled)
    prediction_timer_train = time.time() - prediction_timer_train
    
    prediction_timer_test = time.time()
    predicted_classes_test = CV_simple_deep_classifier_CVer.predict(X_test_scaled)
    prediction_timer_test = time.time() - prediction_timer_test
    
    # Counter number of parameters #
    #------------------------------#
    # Extract Best Model
    best_model = CV_simple_deep_classifier_CVer.best_estimator_
    # Count Number of Parameters
    N_params_best_classifier = np.sum([np.prod(v.get_shape().as_list()) for v in best_model.model.trainable_variables])
    
    # Timers #
    #--------#
    timer_output = prediction_timer_test#np.array([prediction_timer_train,prediction_timer_test])

    
    # Return Values
    return predicted_classes_train, predicted_classes_test, N_params_best_classifier, timer_output

# Update User
#-------------#
print('Deep Classifier - Ready')
















#-------------------------------#
#=### Results & Summarizing ###=#
#-------------------------------#
# We empirically estimate the standard error and confidence intervals or the relevant error distributions using the method of this paper: [Bootstrap Methods for Standard Errors, Confidence Intervals, and Other Measures of Statistical Accuracy - by: B. Efron and R. Tibshirani ](https://www.jstor.org/stable/2245500?casa_token=w_8ZaRuo1qwAAAAA%3Ax5kzbYXzxGSWj-EZaC10XyOVADJyKQGXOVA9huJejP9Tt7fgMNhmPhj-C3WdgbB9AEZdqjT5q_azPmBLH6pDq61jzVFxV4XxqBuerQRBLaaOFKcyr0s&seq=1#metadata_info_tab_contents).

# In[ ]:


def bootstrap(data, n=1000, func=np.mean):
    """
    Generate `n` bootstrap samples, evaluating `func`
    at each resampling. `bootstrap` returns a function,
    which can be called to obtain confidence intervals
    of interest.
    """
    simulations = list()
    sample_size = len(data)
    xbar_init = np.mean(data)
    for c in range(n):
        itersample = np.random.choice(data, size=sample_size, replace=True)
        simulations.append(func(itersample))
    simulations.sort()
    def ci(p):
        """
        Return 2-sided symmetric confidence interval specified
        by p.
        """
        u_pval = (1+p)/2.
        l_pval = (1-u_pval)
        l_indx = int(np.floor(n*l_pval))
        u_indx = int(np.floor(n*u_pval))
        return(simulations[l_indx],xbar_init,simulations[u_indx])
    return(ci)



def get_Error_distribution_plots(test_set_data,
                                 model_test_results,
                                 NEU_model_test_results,
                                 model_name):
    # Initialization
    import seaborn as sns
    import warnings
    warnings.filterwarnings("ignore")

    # Initialize NEU-Model Name
    NEU_model_name = "NEU-"+model_name
    plt.figure(num=None, figsize=(12, 12), dpi=80, facecolor='w', edgecolor='k')

    # Initialize Errors
    Er = model_test_results - data_x_test_raw.reshape(-1,)
    NEU_Er = NEU_model_test_results - data_x_test_raw.reshape(-1,)
    
    # Internal Computations 
    xbar_init = np.mean(Er)
    NEU_xbar_init = np.mean(NEU_Er)

    # Generate Plots #
    #----------------#
    # generate 5000 resampled sample means  =>
    means = [np.mean(np.random.choice(Er,size=len(Er),replace=True)) for i in range(5000)]
    NEU_means = [np.mean(np.random.choice(NEU_Er,size=len(NEU_Er),replace=True)) for i in range(5000)]
    sns.distplot(means, color='r', kde=True, hist_kws=dict(edgecolor="r", linewidth=.675),label=model_name)
    sns.distplot(NEU_means, color='b', kde=True, hist_kws=dict(edgecolor="b", linewidth=.675),label=NEU_model_name)
    plt.xlabel("Initial Sample Mean: {}".format(xbar_init))
    plt.title("Distribution of Sample Mean")
    plt.axvline(x=xbar_init) # vertical line at xbar_init
    plt.legend(loc="upper left")
    plt.title("Model Predictions")
    # Save Plot
    plt.savefig('./outputs/plotsANDfigures/'+model_name+'.pdf', format='pdf')
    # Show Plot
    if is_visuallty_verbose == True:
        plt.show(block=False)