# MIT License
# 
# Copyright (c) 2025 Jeffrey Boylan
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


"""
IRMM Mass Prediction Model
Author: Jeffrey Boylan (greylikeskies)
"""

import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

PLANCK_MASS = 1.22e22  # MeV

# Log10 constants
C_pi = np.log10(np.pi)
C_e = np.log10(np.e)
C_hbar = np.log10(6.582e-22)  # MeV·s

CONSTANT_VECTOR = [C_pi, C_e, C_hbar]

# Example known particles (MeV)
particles = {
    'electron': 0.511,
    'muon': 105.66,
    'tau': 1776.86,
    'up': 2.2,
    'down': 4.7,
    'strange': 96.0,
    'charm': 1270.0,
    'bottom': 4180.0,
    'top': 173100.0,
    'proton': 938.27,
    'neutron': 939.57,
}

def compute_mu(mass_mev):
    return np.log10(mass_mev / PLANCK_MASS)

def interference_vector(mu, constants):
    return [mu - c for c in constants]

def build_dataset(particle_dict):
    X, y = [], []
    for name, mass in particle_dict.items():
        mu = compute_mu(mass)
        d = interference_vector(mu, CONSTANT_VECTOR)
        X.append(d)
        y.append(mu)
    return np.array(X), np.array(y)

def train_model(X, y):
    model = LinearRegression()
    model.fit(X, y)
    return model

def predict_mass(mu_pred):
    return 10 ** (mu_pred) * PLANCK_MASS

def plot_interference_space(X, y, labels):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for i, point in enumerate(X):
        ax.scatter(*point, label=labels[i])
        ax.text(*point, labels[i], fontsize=9)
    ax.set_xlabel('μ - log(pi)')
    ax.set_ylabel('μ - log(e)')
    ax.set_zlabel('μ - log(hbar)')
    plt.title("Particle Positions in Interference Space")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    X, y = build_dataset(particles)
    model = train_model(X, y)
    predictions = model.predict(X)

    print("\n=== Predicted vs Actual Masses ===")
    for name, actual_mu, pred_mu in zip(particles.keys(), y, predictions):
        actual_mass = predict_mass(actual_mu)
        predicted_mass = predict_mass(pred_mu)
        print(f"{name:<10} : Predicted = {predicted_mass:.2f} MeV, Actual = {actual_mass:.2f} MeV, Δ = {abs(predicted_mass - actual_mass):.6f}")

    plot_interference_space(X, y, list(particles.keys()))
