import os
import numpy as np
from datetime import datetime
import json


# Example kernel function (oscillatory behavior)
def kernel_osc(x, a, b, alpha):
    return a * (np.exp(-b * abs(x)) * ((b * np.sin(abs(alpha * x))) + np.cos(alpha * x)))


def kernel_gauss(x, a_ex, s_ex, w_in):
    return a_ex * np.exp(-0.5 * x ** 2 / s_ex ** 2) - w_in


def kernel_osc(x, a, b, alpha):
    return a * (np.exp(-b * abs(x)) * ((b * np.sin(abs(alpha * x))) + np.cos(alpha * x)))


def external_input_function(x, t, input_pars):
    center, amplitude, width, active_start, active_end = input_pars
    if active_start <= t <= active_end:  # Check if the input is active
        return amplitude * (np.exp(-((x - center) ** 2) / (2 * (width ** 2))))  # Gaussian input
    else:
        return np.zeros_like(x)  # Return zero if not active





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


def save_external_input_params(params, filename="external_input_params.json"):
    """Saves external input parameters to a JSON file."""
    with open(filename, 'w') as f:
        json.dump(params, f)
    print(f"Parameters saved to {filename}")


def load_external_input_params(filename="external_input_params.json"):
    """Loads external input parameters from a JSON file."""
    if not os.path.exists(filename):
        raise FileNotFoundError(f"{filename} not found.")

    with open(filename, 'r') as f:
        params = json.load(f)
    print(f"Parameters loaded from {filename}")
    return params
