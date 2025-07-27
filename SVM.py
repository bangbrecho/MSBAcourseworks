# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 22:06:40 2025

@author: 
"""
from pyomo.environ import *
from pyomo.opt import SolverStatus, TerminationCondition
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as pt
import random

# Read data
df = pd.read_csv("Data.csv", header=None)

# Use the first 4 columns as features and the 5th column as the class
X = df.iloc[:, :4].values
y = df.iloc[:, 4].values

# Convert labels to +1 / -1
y = (y == 'Iris-setosa').astype(int) * 2 - 1

d = X.shape[1]  # Number of features
M = np.max(np.linalg.norm(X, axis=1))    # Bound for constraints
epsilon = 5e-2  # Convergence threshold
max_iter = 100  # Maximum number of iterations

# Initialize w and b randomly
np.random.seed(42)
random.seed(42)
w = np.random.uniform(-M, M, d)
b = np.random.uniform(-M, M)
k = 0

# Objective function: 1/2 * ||w||^2
def objective(z):
    w = z[:-1]
    return 0.5 * np.dot(w, w)

# Gradient of the objective function
def gradient(z):
    w = z[:-1]
    return np.append(w, 0)  # Gradient w.r.t. w is w, w.r.t. b is 0

# Compute v_k by solving the linear programming problem
def compute_vk(z):
    model = ConcreteModel()
    
    # Decision variables: v = [w_v; b_v]
    model.w_v = Var(range(d), bounds=(-M, M))
    model.b_v = Var(bounds=(-M, M))
    
    # Objective: minimize gradient(z)^T v
    def objective_rule(m):
        return sum(gradient(z)[j] * m.w_v[j] for j in range(d)) + gradient(z)[-1] * m.b_v
    
    model.obj = Objective(rule=objective_rule, sense=minimize)
    
    # Constraints: y_i (w_v^T x_i + b_v) >= 1 for all i
    def constraint_rule(m, i):
        return y[i] * (sum(m.w_v[j] * X[i, j] for j in range(d)) + m.b_v) >= 1
    
    model.constraints = Constraint(range(len(y)), rule=constraint_rule)
    
    # Solve the problem
    solver = SolverFactory('glpk')  # Use GLPK solver for linear programming
    results = solver.solve(model)
    
    # Extract the solution
    w_v = np.array([model.w_v[j]() for j in range(d)])
    b_v = model.b_v()
    
    return np.append(w_v, b_v)

# Compute step size tau_k
def compute_step_size(z, d):
    w_tau = z[:-1]
    dw_tau = d[:-1]
    tau = np.dot(-w_tau,dw_tau)/np.dot(dw_tau,dw_tau)
    tau = np.clip(tau,0,1)
    
    return tau

z = np.append(w, b)
z_history = pd.DataFrame(columns=[f'w_{j+1}' for j in range(d)] + ['b', 'obj_res'])
z_history.loc[0] = np.append(z, 0.5 * np.dot(z[:-1], z[:-1]))  # Add initial z and obj_res

# Main loop
for i in range(max_iter):
    # Compute v_k
    v_k = compute_vk(z)
    
    # Compute direction d_k = v_k - z
    d_k = v_k - z
    
    # Compute step size tau_k
    tau_k = compute_step_size(z, d_k)
    
    # Update z
    z_new = z + tau_k * d_k
    
    # Compute 0.5 * ||w||^2 for the new z
    w_new = z_new[:-1]
    obj_res = 0.5 * np.dot(w_new, w_new)
    
    # Add the new z and obj_res to the DataFrame
    z_history.loc[i + 1] = np.append(z_new, obj_res)
    
    # Check for convergence
    if np.linalg.norm(z_new - z) < epsilon:
        break
    
    z = z_new
    k += 1

# Extract optimal w and b
w_opt = z[:-1]
b_opt = z[-1]

# Predict function
def predict(X):
    return np.sign(np.dot(X, w_opt) + b_opt)

# Evaluate accuracy
y_pred = predict(X)
accuracy = np.mean(y_pred == y)
print(f"Final Accuracy: {accuracy:.4f}")

#%% objective value
objective_value = 0.5 * np.linalg.norm(w_opt)**2
print(f"Objective function value: {objective_value:.6f}")
#%% change in w and b over iterations
import matplotlib.pyplot as plt

# Plot w over iterations
plt.figure(figsize=(10, 6))
plt.plot(z_history.index, z_history['w_1'], label='w_1')
plt.plot(z_history.index, z_history['w_2'], label='w_2')
plt.plot(z_history.index, z_history['w_3'], label='w_3')
plt.plot(z_history.index, z_history['w_4'], label='w_4')
plt.xlabel('Iteration')
plt.ylabel('Value')
plt.title('Change in w over Iterations')
plt.legend()
plt.show()

# Plot b over iterations
plt.figure(figsize=(10, 6))
plt.plot(z_history.index, z_history['b'], label='b', color='red')
plt.xlabel('Iteration')
plt.ylabel('Value')
plt.title('Change in b over Iterations')
plt.legend()
plt.show()

# Plot Objective value over iterations
plt.figure(figsize=(10, 6))
plt.plot(z_history.index, z_history['obj_res'], label='objective', color='blue')
plt.xlabel('Iteration')
plt.ylabel('Value')
plt.title('Change in objective value over Iterations')
plt.legend()
plt.show()

#%% accuracy
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Reduce dimensionality to 2D
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)  # X (n_samples, 4) -> X_pca (n_samples, 2)


x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                     np.linspace(y_min, y_max, 200))

# Transform grid points back to 4D and compute predictions
grid_points_pca = np.c_[xx.ravel(), yy.ravel()]
grid_points_original = pca.inverse_transform(grid_points_pca)  
Z = np.sign(np.dot(grid_points_original, w_opt) + b_opt)
Z = Z.reshape(xx.shape)

# Plot decision boundary
plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, edgecolors='k', cmap='coolwarm')

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("SVM Decision Boundary (PCA Projection)")
plt.colorbar(label='Class')
plt.show()


# Compute SVM decision values
decision_values = np.dot(X, w_opt) + b_opt  

plt.figure(figsize=(8, 4))
plt.hist(decision_values[y == 1], bins=20, alpha=0.6, label="Iris-setosa", color='blue')
plt.hist(decision_values[y == -1], bins=20, alpha=0.6, label="Other", color='red')
plt.axvline(0, color='black', linestyle="--", label="Decision Boundary")
plt.xlabel("SVM Decision Value")
plt.ylabel("Frequency")
plt.title("Decision Value Distribution")
plt.legend()
plt.show()

