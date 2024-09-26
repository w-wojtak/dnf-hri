import numpy as np

# Example kernel function (oscillatory behavior)
def kernel_osc(x, kernel_param1, kernel_param2):
    return np.exp(-x**2 / (2 * kernel_param1)) * np.cos(2 * np.pi * x / kernel_param2)

# Example external input function (Gaussian pulse active for some time)
def external_input_function(x, t, input_pars):
    center, width, active_start, active_end = input_pars
    if active_start <= t <= active_end:
        return np.exp(-(x - center)**2 / (2 * width**2))
    else:
        return np.zeros_like(x)

# Simultaneous integration function to handle multiple fields
def simultaneous_integration(fields):
    num_time_steps = len(fields[0].t)

    for i in range(num_time_steps):
        for field in fields:
            field.integrate_single_step(i)
