import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

# Constants
rho = 62.4  # lbm/ft^3 (density of water)
mu = 1.002e-5 * 0.020885  # lbm/ft-s (dynamic viscosity of water)
g = 32.174  # ft/s^2 (gravitational acceleration)


# Moody diagram setup
def draw_moody_diagram():
    Re = np.logspace(3, 8, 400)
    rel_roughness = [0, 0.00005, 0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01]

    plt.figure(figsize=(8, 6))
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Reynolds Number (Re)')
    plt.ylabel('Friction Factor (f)')
    plt.title('Moody Diagram')

    for rr in rel_roughness:
        f_turbulent = 1 / (1.8 * np.log10(Re / 6.9)) ** 2
        plt.plot(Re, f_turbulent, label=f'ϵ/d={rr}')

    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    return plt


# Colebrook equation solver
def colebrook(f, Re, e_d):
    return 1 / np.sqrt(f) + 2.0 * np.log10(e_d / 3.7 + 2.51 / (Re * np.sqrt(f)))


def compute_friction_factor(Re, e_d):
    if Re < 2000:
        return 64 / Re  # Laminar flow
    elif Re > 4000:
        f_guess = 0.02
        f_solution, = fsolve(colebrook, f_guess, args=(Re, e_d))
        return f_solution  # Turbulent flow
    else:
        f_lam = 64 / 2000
        f_cb, = fsolve(colebrook, 0.02, args=(4000, e_d))
        mu_f = f_lam + (f_cb - f_lam) * (Re - 2000) / 2000
        sigma_f = 0.2 * mu_f
        return np.random.normal(mu_f, sigma_f)


def compute_head_loss(d, e, Q, L):
    A = np.pi * (d / 12) ** 2 / 4  # Convert diameter to feet
    V = (Q / 448.831) / A  # Convert gpm to ft^3/s then velocity
    Re = (rho * V * (d / 12)) / mu
    e_d = e / (d * 1e6)
    f = compute_friction_factor(Re, e_d)
    h_f = (f * (L / d) * (V ** 2 / (2 * g)))
    return Re, f, h_f


# Initialize Moody diagram
plt = draw_moody_diagram()
re_list = []
f_list = []

while True:
    d = float(input("Enter pipe diameter (inches): "))
    e = float(input("Enter pipe roughness (micro-inches): "))
    Q = float(input("Enter flow rate (gallons/min): "))
    L = float(input("Enter pipe length (feet): "))

    Re, f, h_f = compute_head_loss(d, e, Q, L)
    re_list.append(Re)
    f_list.append(f)

    print(f"Reynolds Number: {Re:.2f}")
    print(f"Friction Factor: {f:.5f}")
    print(f"Head Loss per Foot: {h_f:.5f} ft/ft")

    marker = '^' if 2000 <= Re <= 4000 else 'o'
    plt.scatter(Re, f, color='red', marker=marker, s=100)
    plt.draw()
    plt.pause(0.1)

    cont = input("Do you want to enter new parameters? (yes/no): ").strip().lower()
    if cont != 'yes':
        break

plt.show()
