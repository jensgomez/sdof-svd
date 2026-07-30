# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 22:39:31 2026

@author: jeral
"""

from numpy import arange
from numpy import cos, sqrt, pi
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from numpy import gradient


# ------------- True System Inputs 
#kg, N-s/m, N/m

m_true, c_true, k_true = 1.0, 0.5, 10.0 

# Terminal output for initial inputs
def heading_forward_system():
    print("=" * 80)
    print(" ")
    print(" True Inputs for the Model ")
    print(f"   Mass      = {m_true} kg")
    print(f"   Damping   = {c_true} N*s/m")
    print(f"   Stiffness = {k_true} N/m")
    print(" ")
    print(f"   Natural Frequency = {sqrt(k_true/m_true)/(2*pi)} Hz")    
    print("=" * 80)

# Create time array
def create_time(t0=0.0, tf=10.0, n=int(1E4)):
    time = arange(start=t0, stop=tf, step=(tf-t0)/n)
    return time

# Forcing function for use in numerical solver
def cosine_forcing(t):
    return 5.0 * cos(3*t)

# Numerical solution for system of ODEs
def spring_mass_damper(t, y, m, c, k, force_func):
    """Defines the state-space ODE system with external forcing.

    Parameters:
        t          : Current time
        y          : State vector [position x, velocity v]
        m, c, k    : Mass, damping, and stiffness coefficients
        force_func : A callable function F(t) returning force at time t
    """    
    
    # Initialize initial conditions based on y = [x0, v0]
    x, v = y
    
    # Compute the forcing function
    F = force_func(t)
    
    # State derivatives for system of first order differential equations
    dxdt = v
    dvdt = (F - c * v - k * x) / m

    return [dxdt, dvdt]

# Calculate the numerical derivate
def numerical_derivative(t, ft):
    
    dft_dt = gradient(ft, t)
    
    return dft_dt
    
# Plotting function
def plot_response(
    sol, force_func=None, title="Spring-Mass-Damper System Response"):
    """Generates a single plot for position, velocity, and optional forcing input."""
    fig, ax = plt.subplots(figsize=(9, 5))

    t = sol.t
    x = sol.y[0]  # Position
    v = sol.y[1]  # Velocity

    # Plot position and velocity
    ax.plot(t, x, label="Position $x(t)$ [m]", color="tab:blue", linewidth=2)
    ax.plot(
        t,
        v,
        label="Velocity $v(t)$ [m/s]",
        color="tab:green",
        linestyle="--",
        alpha=0.7,
    )

    # Plot forcing input if provided
    if force_func is not None:
        f_values = [force_func(ti) for ti in t]
        ax.plot(
            t,
            f_values,
            label="Forcing $F(t)$ [N]",
            color="tab:red",
            linestyle=":",
            alpha=0.7,
        )

    # Formatting
    ax.set_title(title)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Amplitude")
    ax.axhline(0, color="black", linewidth=0.8, linestyle=":")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="upper right")

    plt.tight_layout()
    plt.show()
    

def main():

    # Print the system data 
    heading_forward_system()
    
    
    # --- Simulation Setup ---
    t0 = 0.0
    tf = 10.0
    t_span = (t0, tf)
    t_eval = create_time(t0, tf, n=10000)
    y0 = [0.0, 0.0]  # Start from rest at origin
    
    # Solve with harmonic forcing function passed in via `args`
    sol_harmonic = solve_ivp(
        fun = spring_mass_damper,
        t_span = t_span,
        y0 = y0,
        t_eval = t_eval,
        args=(m_true, c_true, k_true, cosine_forcing),
        method="RK45",
    )
    
    t_soln = sol_harmonic.t
    disp_soln = sol_harmonic.y[0]
    vel_soln = sol_harmonic.y[1]
    
    velo_soln_derv = numerical_derivative(t=t_soln, ft=disp_soln)
    
    print(velo_soln_derv, vel_soln)
    
    # --- Plotting Results ---
    # Plot single figure
    plot_response(
        sol_harmonic,
        force_func=cosine_forcing,
        title="Spring-Mass-Damper Response (Harmonic Force)",
    )   
        
if __name__ == "__main__":
    main()