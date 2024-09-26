import numpy as np

# Example kernel function (oscillatory behavior)
def kernel_osc(x, a, b, alpha):
    return a * (np.exp(-b*abs(x)) * ((b * np.sin(abs(alpha*x)))+np.cos(alpha*x)))

def external_input_function(x, t, input_pars):
    center, width, active_start, active_end = input_pars
    if active_start <= t <= active_end:  # Check if the input is active
        return np.exp(-((x - center) ** 2) / (2 * (width ** 2)))  # Gaussian input
    else:
        return np.zeros_like(x)  # Return zero if not active


# Simultaneous integration function to handle multiple fields
def simultaneous_integration(fields):
    num_time_steps = len(fields[0].t)

    for i in range(num_time_steps):
        for field in fields:
            field.integrate_single_step(i)
