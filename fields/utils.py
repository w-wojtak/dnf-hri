import numpy as np

# Example kernel function (oscillatory behavior)
def kernel_osc(x, a, b, alpha):
    return a * (np.exp(-b*abs(x)) * ((b * np.sin(abs(alpha*x)))+np.cos(alpha*x)))

# Example external input function (Gaussian pulse active for some time)
def external_input_function(x, t, input_pars_list):
    """
    input_pars_list: A list of input parameters. Each element should be a tuple:
    (center, width, active_start, active_end)
    """
    external_input = np.zeros_like(x)

    # Loop over each input
    for input_pars in input_pars_list:
        center, width, active_start, active_end = input_pars
        if active_start <= t <= active_end:
            external_input += np.exp(-(x - center) ** 2 / (2 * width ** 2))

    return external_input


# Simultaneous integration function to handle multiple fields
def simultaneous_integration(fields):
    num_time_steps = len(fields[0].t)

    for i in range(num_time_steps):
        for field in fields:
            field.integrate_single_step(i)
