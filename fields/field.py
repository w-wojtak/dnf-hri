# fields/field.py

import numpy as np
from fields.utils import external_input_function, kernel_osc
from fields.utils import load_sequence_memory
import matplotlib.pyplot as plt


class Field:
    def __init__(self, kernel_pars, field_pars, external_input_pars_list, tau_h=100, h_0=0, input_flag=True,
                 name="Field", field_type=None, theta=1.0):
        self.kernel_pars = kernel_pars
        self.field_pars = field_pars
        self.external_input_pars_list = external_input_pars_list
        self.name = name
        self.field_type = field_type  # New attribute for field type
        self.x_lim, self.t_lim, self.dx, self.dt = field_pars
        self.tau_h = tau_h
        self.h_0 = h_0
        self.input_flag = input_flag
        self.theta = theta

        # Spatial and temporal grids
        self.x = np.arange(-self.x_lim, self.x_lim + self.dx, self.dx)
        self.t = np.arange(0, self.t_lim + self.dt, self.dt)

        # State history and current state initialization
        self.history_u = np.zeros([len(self.t), len(self.x)])

        if field_type == "decision":
            self.u_field = load_sequence_memory().flatten() - 7  # Ensure it's 1D
            self.loaded_internal_input = load_sequence_memory().flatten()  - 7 # Store loaded data for internal input
        else:
            self.u_field = h_0 * np.ones(np.shape(self.x))  # Default initialization
            self.loaded_internal_input = np.zeros_like(self.x)  # Default for other types
        self.activity = np.zeros([len(self.t), len(self.x)])  # Initialize activity

        # Initialize activity to zeros
        self.activity = np.zeros([len(self.t), len(self.x)])  # This stays the same for both types

        self.h_u = h_0 * np.ones(np.shape(self.x))

        # Fourier transform of the kernel function
        self.w_hat = np.fft.fft(kernel_osc(self.x, *self.kernel_pars))

        # List of connected fields (internal inputs)
        self.connected_fields = []

        # Initialize histories for internal and external inputs
        self.history_external_input = np.zeros([len(self.t), len(self.x)])  # External input history
        self.history_internal_input = np.zeros([len(self.t), len(self.x)])  # Internal input history


    def integrate_single_step(self, i):
        external_input = self.get_external_input(self.t[i])
        internal_input = self.get_internal_input(i)

        # Track the internal and external inputs for this time step
        self.history_external_input[i, :] = external_input
        self.history_internal_input[i, :] = internal_input

        f = np.heaviside(self.u_field - self.theta, 1)
        f_hat = np.fft.fft(f)
        conv = self.dx * np.fft.ifftshift(np.real(np.fft.ifft(f_hat * self.w_hat)))

        # Calculate h_u based on field type
        if self.field_type == "sequence_memory":
            self.h_u += self.dt / self.tau_h * f
        elif self.field_type == "decision":
            self.h_u += self.dt / self.tau_h
        else:
            # If the type is not provided, keep h_u constant
            self.h_u = self.h_u  # No change to h_u

        self.u_field += self.dt * (-self.u_field + conv + external_input + internal_input + self.h_u)

        self.history_u[i, :] = self.u_field
        self.activity[i, :] = self.u_field  # Track activity as the current state

    def get_external_input(self, t):
        total_input = np.zeros_like(self.x)
        for input_pars in self.external_input_pars_list:  # Iterate over the list of input parameters
            total_input += external_input_function(self.x, t, input_pars)
        return total_input

    def get_internal_input(self, i):
        # Initialize internal input with the loaded data for "decision" field type
        if self.field_type == "decision":
            internal_input = self.loaded_internal_input.copy()  # Start with the loaded internal input
        else:
            internal_input = np.zeros_like(self.u_field)  # For other types, initialize to zeros

        # Add inputs from connected fields with their custom parameters
        for connected_field, weight, connection_params in self.connected_fields:
            # Extract connection-specific parameters
            threshold = connection_params.get('threshold', None)  # Get threshold if provided
            # scaling_factor = connection_params.get('scaling', 1.0)  # Default scaling factor is 1.0

            # Apply threshold logic if a threshold is provided
            if threshold is not None:
                mask = connected_field.u_field > threshold  # Create a mask where activity exceeds threshold
                internal_input += weight * connected_field.u_field * mask
            else:
                # If no threshold, just apply the connection with weight and scaling factor
                internal_input += weight * connected_field.u_field

        return internal_input

    def add_connection(self, field, weight=0.0, connection_params=None):
        """
        Adds a connection to another field with an optional weight and custom connection parameters.
        :param field: The connected field.
        :param weight: The weight of the connection.
        :param connection_params: A dictionary of connection-specific parameters like thresholds.
        """
        if connection_params is None:
            connection_params = {}  # If no parameters provided, use an empty dictionary
        self.connected_fields.append((field, weight, connection_params))

    def plot_loaded_field(self):
        """Plots the loaded u_field for the decision field."""
        if self.field_type == "decision":
            plt.plot(self.x, self.u_field)
            plt.xlabel('x')
            plt.ylabel('Activity (u_field)')
            plt.title(f'Loaded Field Data for {self.name}')
            plt.show()
        else:
            print(f"No loaded field to plot for field type {self.field_type}")
