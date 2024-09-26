import os
import numpy as np
from datetime import datetime


# Example kernel function (oscillatory behavior)
def kernel_osc(x, a, b, alpha):
    return a * (np.exp(-b * abs(x)) * ((b * np.sin(abs(alpha * x))) + np.cos(alpha * x)))


def kernel_gauss(x, a_ex, s_ex, w_in):
    return a_ex * np.exp(-0.5 * x ** 2 / s_ex ** 2) - w_in


def kernel_osc(x, a, b, alpha):
    return a * (np.exp(-b * abs(x)) * ((b * np.sin(abs(alpha * x))) + np.cos(alpha * x)))


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


def save_final_state(data, name):
    # Create the 'data' directory if it doesn't exist
    os.makedirs('data', exist_ok=True)

    # Get current date and time for the file name
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = f"data/{name}_final_state_{current_time}.npy"

    # Save the final state to a .npy file
    np.save(file_name, data)
    print(f"Final state saved to {file_name}")


def load_sequence_memory(filename=None):
    if filename is None:
        files = [f for f in os.listdir('data') if f.endswith('.npy')]
        if not files:
            raise FileNotFoundError("No .npy files found in the 'data' folder.")

        latest_file = max([os.path.join('data', f) for f in files], key=os.path.getmtime)
        filename = latest_file

    data = np.load(filename)
    print(f"Loaded sequence memory from {filename}")

    # Ensure the data is at least 2D
    if data.ndim == 1:  # If it's 1D, reshape to 2D (1 row, many columns)
        data = data.reshape(1, -1)
    return data
