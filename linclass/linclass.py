
# coding: utf-8

# # Linear Classification
# George Ho 10/19/17

# In[1]:

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')


# In[2]:

# Ground truth
pi = 0.4
mu1 = np.ones(2)
mu2 = -np.ones(2)
sigma = np.identity(2)


# In[3]:

# Training data
train_class1 = np.random.multivariate_normal(mu1, sigma, round(500*pi))
train_class2 = np.random.multivariate_normal(mu2, sigma, round(500*(1-pi)))

train_binary_data = np.vstack([train_class1, train_class2])
train_class_label = np.hstack([np.ones(round(500*pi)), np.zeros(round(500*(1-pi)))])


# In[4]:

# Test data
test_class1 = np.random.multivariate_normal(mu1, sigma, round(100*pi))
test_class2 = np.random.multivariate_normal(mu2, sigma, round(100*(1-pi)))

test_binary_data = np.vstack([test_class1, test_class2])
test_class_labels = np.hstack([np.ones(round(100*pi)), np.zeros(round(100*(1-pi)))])


# ## Generative Model: Gaussian Class-Conditional Probabilities

# In[5]:

# ML estimation of pi, mu1, mu2

pi_est = np.count_nonzero(train_class_label)/train_class_label.size
mu1_est = train_binary_data[train_class_label == 1].mean(axis=0)
mu2_est = train_binary_data[train_class_label == 0].mean(axis=0)


# In[6]:

# There is probably a better way of doing this... but idk

# Computing S1
vectors1 = train_binary_data[train_class_label == 1] - mu1_est
foo = np.zeros([len(vectors1), 2, 2])
for i, row in enumerate(vectors1):
    foo[i] = np.outer(row, row)
S1 = foo.mean(axis=0)

# Computing S2
vectors2 = train_binary_data[train_class_label == 0] - mu2_est
bar = np.zeros([len(vectors2), 2, 2])
for i, row in enumerate(vectors2):
    bar[i] = np.outer(row, row)
S2 = bar.mean(axis=0)


# In[7]:

# ML estimation of sigma

N1 = train_binary_data[train_class_label == 1].size
N2 = train_binary_data[train_class_label == 0].size
sigma_est = (N1/(N1+N2))*S1 + (N2/(N1+N2))*S2


# In[8]:

w = np.linalg.inv(sigma_est) @ (mu1_est-mu2_est)
w_0 = - 0.5*(mu1_est.T @ np.linalg.inv(sigma_est) @ mu1_est)       + 0.5*(mu2_est.T @ np.linalg.inv(sigma_est) @ mu2_est)       + np.log(pi_est / (1-pi_est))


# In[9]:

def sigmoid(z):
    return 1/(1+np.exp(-z))

def posterior_probability(x):
    return sigmoid(w@x + w_0)


# In[10]:

class_decisions = list(map(lambda x: 1 if posterior_probability(x) > 0.5 else 0, test_binary_data))

success = class_decisions == test_class_labels
misclassification_rate = (success.size - np.count_nonzero(success)) / 100


# In[11]:

misclassification_rate


# ## Discriminative Model: Iteratively Reweighted Logistic Regression
# 
# _Without_ nonlinear basis functions... Just by graphing the training data, it looks like they're pretty linearly separable.

# In[12]:

# Initial guess
w = np.atleast_2d(np.ones([3,1]))


# In[13]:

# Logistic regression
phi = np.ones([500, 3])
phi[:, 1:] = train_binary_data
t = train_class_label.reshape(500, 1)

for _ in range(100):
    y = sigmoid((w.T*phi).sum(axis=1)).reshape([500,1])
    R = np.diag(np.multiply(y, 1-y).flatten())
    z = phi @ w - np.linalg.inv(R) @ (y - t)
    w = np.linalg.inv(phi.T @ R @ phi) @ phi.T @ R @ z


# In[14]:

def posterior_probability(x):
    return sigmoid(np.multiply(w.T, x).sum(axis=1))


# In[15]:

test_phi = np.ones([100, 3])
test_phi[:, 1:] = test_binary_data


# In[17]:

class_decisions = list(map(lambda x: 1 if posterior_probability(x) > 0.5 else 0, test_phi))

success = class_decisions == test_class_labels
misclassification_rate = (success.size - np.count_nonzero(success)) / 100


# In[18]:

misclassification_rate


# In[ ]:



