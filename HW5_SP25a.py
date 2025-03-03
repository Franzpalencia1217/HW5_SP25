import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

# Function to solve the Colebrook equation
def colebrook(f, Re, epsilon_d):
    return 1 / np.sqrt(f) + 2.0 * np.log10((epsilon_d / 3.7) + (2.51 / (Re * np.sqrt(f))))

# Function to find the friction factor
def solve_f(Re, epsilon_d):
    if Re < 2000:
        return 64 / Re  # Laminar flow
    elif Re > 4000:
        f_initial = 0.02  # Initial guess
        f_solution, = fsolve(colebrook, f_initial, args=(Re, epsilon_d))
        return f_solution
    else:
        return np.nan  # Transitional range

# Main function to generate the Moody Diagram
def plot_moody_diagram():
    Re_values = np.logspace(3, 8, 500)  # Reynolds number range
    relative_roughness = [0.05, 0.04, 0.03, 0.02, 0.015, 0.01, 0.008, 0.006, 0.004, 0.002, 0.001, 0.0008, 0.0006, 0.0005, 0.0004, 0.0003, 0.0002, 0.00015, 0.0001, 0.00008, 0.00006, 0.00005, 0.00001]

    plt.figure(figsize=(14, 7))
    plt.xscale("log")
    plt.yscale("log")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.xlabel("Reynolds number $Re$")
    plt.ylabel("Friction factor $f$")
    plt.title("Moody Diagram")

    # Plotting each curve for relative roughness
    for epsilon_d in relative_roughness:
        f_values = [solve_f(Re, epsilon_d) for Re in Re_values]
        plt.plot(Re_values, f_values, 'k')
        if epsilon_d >= 0.0001:
            plt.text(1e8, solve_f(1e8, epsilon_d), f"{epsilon_d}", fontsize=8, verticalalignment='center')

    # Plotting the laminar flow region
    Re_laminar = np.logspace(3, np.log10(2000), 200)
    f_laminar = 64 / Re_laminar
    plt.plot(Re_laminar, f_laminar, 'b', linewidth=2)  # Blue solid line

    # Plotting the transitional region
    Re_transitional = np.logspace(np.log10(2000), np.log10(4000), 200)
    f_transitional = np.linspace(64 / 2000, np.nan, 200)
    plt.plot(Re_transitional, f_transitional, 'r--', linewidth=2)  # Red dashed line

    plt.xlim(1e3, 1e8)
    plt.ylim(0.008, 0.1)
    plt.legend(["Laminar Flow", "Transitional Flow"], loc='upper right')
    plt.show()

if __name__ == "__main__":
    plot_moody_diagram()
