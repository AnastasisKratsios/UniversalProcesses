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

# # Visualization
# *From hereon out...do nothing and just let the backend sript run...the images and tables will load :).*

# #### Visualization of Training-Set Performance
# *Each "jump" is just the new $x$ point, so it is not a genuine jump but rather a flattening of a $2$-surface!*

# In[11]:


print("Visualization of Training-Set Performance")
# Initialize Plot #
#-----------------#
plt.figure(num=None, figsize=(12, 12), dpi=80, facecolor='w', edgecolor='k')

plt.plot(predictions_mean,label="prediction",color="purple")
plt.plot(true_mean,label="true",color="green")


# Format Plot #
#-------------#
plt.legend(loc="upper left",prop={'size': 10})
plt.title("Flattenned Model Predictions (Train)")

# Export #
#--------#
# SAVE Figure to .eps
plt.savefig('./outputs/plots/Train.pdf', format='pdf')


# In[1]:


print("Visualizing Training Predictions vs. Ground-Truth (Training Set)")
# Initialize Plot #
#-----------------#
plt.figure(num=None, figsize=(12, 12), dpi=80, facecolor='w', edgecolor='k')

# Generate Plots #
#----------------#
ax = plt.axes(projection='3d')
ax.plot_trisurf(X_train[:-1:,0], X_train[:-1:,1], true_mean, cmap='viridis',linewidth=0.5);
ax.plot_trisurf(X_train[:-1:,0], X_train[:-1:,1], predictions_mean, cmap='RdYlGn',linewidth=0.5);


# Format Plot #
#-------------#
plt.legend(loc="upper left",prop={'size': 10})
plt.title("Model Predictions (Train)")

# Export #
#--------#
# SAVE Figure to .eps
plt.savefig('./outputs/plots/Train3d.pdf', format='pdf')


# #### Visualization of Test-Set Performance
# *Each "jump" is just the new $x$ point, so it is not a genuine jump but rather a flattening of a $2$-surface!*

# In[13]:


print("Visualization of Test-Set Performance")
# Initialize Plot #
#-----------------#
plt.figure(num=None, figsize=(12, 12), dpi=80, facecolor='w', edgecolor='k')

plt.plot(predictions_mean_test,label="prediction",color="purple")
plt.plot(true_mean_test,label="true",color="green")


# Format Plot #
#-------------#
plt.legend(loc="upper left",prop={'size': 10})
plt.title("Flattenned Model Predictions (Train)")

# Export #
#--------#
# SAVE Figure to .eps
plt.savefig('./outputs/plots/Test.pdf', format='pdf')


# In[38]:


print("Visualizing Training Predictions vs. Ground-Truth (Testing Set)")
# sns.set()
# Initialize Plot #
#-----------------#
plt.figure(num=None, figsize=(12, 12), dpi=80, facecolor='w', edgecolor='k')

# Generate Plots #
#----------------#
ax = plt.axes(projection='3d')
ax.plot_trisurf(X_test[:-1:,0], X_test[:-1:,1], true_mean_test, cmap='viridis',linewidth=0.5);
ax.plot_trisurf(X_test[:-1:,0], X_test[:-1:,1], predictions_mean_test, cmap='RdYlGn',linewidth=0.5);


# Format Plot #
#-------------#
plt.legend(loc="upper left",prop={'size': 10})
plt.title("Model Predictions (Test)")

# Export #
#--------#
# SAVE Figure to .eps
plt.savefig('./outputs/plots/Test3d.pdf', format='pdf')


# # Error Plots

# ## Training Set

# In[36]:


print("Visualizing Predictions Erros (Training Set)")
# sns.set()
# Initialize Plot #
#-----------------#
plt.figure(num=None, figsize=(12, 12), dpi=80, facecolor='w', edgecolor='k')

# Generate Plots #
#----------------#
ax = plt.axes(projection='3d')
ax.plot_trisurf(X_train[:-1:,0], X_train[:-1:,1], true_mean-predictions_mean, cmap='RdYlBu',linewidth=0.5);


# Format Plot #
#-------------#
# plt.legend(loc="upper left",prop={'size': 10})
plt.title("Prediction Errors (Train)")

# Export #
#--------#
# SAVE Figure to .eps
plt.savefig('./outputs/plots/Train_Errors_3d.pdf', format='pdf')


# ## Test Set

# In[37]:


print("Visualizing Predictions Erros (Test Set)")

# sns.set()
# Initialize Plot #
#-----------------#
plt.figure(num=None, figsize=(12, 12), dpi=80, facecolor='w', edgecolor='k')

# Generate Plots #
#----------------#
ax = plt.axes(projection='3d')
ax.plot_trisurf(X_test[:-1:,0], X_test[:-1:,1], true_mean_test-predictions_mean_test, cmap='RdYlBu',linewidth=0.5);


# Format Plot #
#-------------#
# plt.legend(loc="upper left",prop={'size': 10})
plt.title("Prediction Errors (Test)")

# Export #
#--------#
# SAVE Figure to .eps
plt.savefig('./outputs/plots/Test_Errors_3d.pdf', format='pdf')


# ## Update User

# ### Training-Set Performance

# In[15]:


Type_A_Prediction


# ### Test-Set Performance

# In[16]:


Type_A_Prediction_test


# ---

# ---
# # Fin
# ---

# ---
