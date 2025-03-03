import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Given constants
A = 4.909e-4  # m^2
Cd = 0.6
ps = 1.4e7  # Pa
pa = 1.0e5  # Pa
V = 1.473e-4  # m^3
beta = 2.0e9  # Pa
rho = 850  # kg/m^3
K_valve = 2e-5
m = 30  # kg
gamma = 0.002  # Constant input

# Initial conditions
x0 = 0
xdot0 = 0
p1_0 = pa
p2_0 = pa
initial_conditions = [x0, xdot0, p1_0, p2_0]

# Time span
t_span = (0, 0.02)
t_eval = np.linspace(0, 0.02, 1000)

# Differential equations
def hydraulic_valve_system(t, y):
    x, xdot, p1, p2 = y

    # Equations based on given formulation
    xddot = (p1 - p2) * A / m
    p1dot = (K_valve * (ps - p1) * np.sign(gamma) * A - rho * A * xdot) / (V / beta)
    p2dot = (-K_valve * (p2 - pa) * np.sign(gamma) * A - rho * A * xdot) / (V / beta)

    return [xdot, xddot, p1dot, p2dot]

# Solving the system
solution = solve_ivp(hydraulic_valve_system, t_span, initial_conditions, t_eval=t_eval, method='RK45')

# Extract results
t = solution.t
x = solution.y[0]
p1 = solution.y[2]
p2 = solution.y[3]

# Create plots
fig, axs = plt.subplots(2, 1, figsize=(10, 10))

# Plot x vs. time
axs[0].plot(t, x, label="x (m)", color="blue")
axs[0].set_title("Velocity Response of Hydraulic Valve System")
axs[0].set_xlabel("Time [s]")
axs[0].set_ylabel("x (m)")
axs[0].grid()
axs[0].legend()

# Plot p1 and p2 vs. time
axs[1].plot(t, p1, label="p1 (Pa)", color="blue")
axs[1].plot(t, p2, label="p2 (Pa)", color="red")
axs[1].set_title("Pressure Response of Hydraulic Valve System")
axs[1].set_xlabel("Time [s]")
axs[1].set_ylabel("Pressure (Pa)")
axs[1].grid()
axs[1].legend()

# Show plots
plt.tight_layout()
plt.show()
