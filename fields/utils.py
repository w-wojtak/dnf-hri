import os
import numpy as np
from datetime import datetime

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
    # If no filename is provided, load the latest file in the data folder
    if filename is None:
        # List all files in the 'data' directory
        files = [f for f in os.listdir('data') if f.endswith('.npy')]
        if not files:
            raise FileNotFoundError("No .npy files found in the 'data' folder.")

        # Get the latest file based on modification time
        latest_file = max([os.path.join('data', f) for f in files], key=os.path.getmtime)
        filename = latest_file

    # Load the sequence memory data from the specified file
    data = np.load(filename)
    print(f"Loaded sequence memory from {filename}")
    return data