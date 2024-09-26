# fields/field.py

import numpy as np
from fields.utils import external_input_function, kernel_osc

class Field:
    def __init__(self, kernel_pars, field_pars, external_input_pars_list, tau_h=100, h_0=0, input_flag=True, name="Field"):
        self.kernel_pars = kernel_pars
        self.field_pars = field_pars
        self.external_input_pars_list = external_input_pars_list  # Correctly store the input parameters
        self.name = name
        self.x_lim, self.t_lim, self.dx, self.dt, self.theta = field_pars
        self.tau_h = tau_h
        self.h_0 = h_0
        self.input_flag = input_flag

        # Spatial and temporal grids
        self.x = np.arange(-self.x_lim, self.x_lim + self.dx, self.dx)
        self.t = np.arange(0, self.t_lim + self.dt, self.dt)

        # State history and current state initialization
        self.history_u = np.zeros([len(self.t), len(self.x)])
        self.u_field = h_0 * np.ones(np.shape(self.x))
        self.h_u = h_0 * np.ones(np.shape(self.x))

        # Activity tracking
        self.activity = np.zeros((len(self.t), len(self.x)))  # Initialize activity array

        # Fourier transform of the kernel function
        self.w_hat = np.fft.fft(kernel_osc(self.x, *self.kernel_pars))

        # List of connected fields (internal inputs)
        self.connected_fields = []

    def integrate_single_step(self, i):
        external_input = self.get_external_input(self.t[i])
        internal_input = self.get_internal_input(i)  # Assume this function exists

        f = np.heaviside(self.u_field - self.theta, 1)
        f_hat = np.fft.fft(f)
        conv = self.dx * np.fft.ifftshift(np.real(np.fft.ifft(f_hat * self.w_hat)))

        self.h_u += self.dt / self.tau_h * f
        self.u_field += self.dt * (-self.u_field + conv + external_input + internal_input + self.h_u)

        self.history_u[i, :] = self.u_field
        self.activity[i, :] = self.u_field  # Track activity as the current state

    def get_external_input(self, t):
        total_input = np.zeros_like(self.x)
        for input_pars in self.external_input_pars_list:  # Iterate over the list of input parameters
            total_input += external_input_function(self.x, t, input_pars)
        return total_input

    def get_internal_input(self, i):
        # Here, define the logic for obtaining internal input
        # This could be influenced by connected fields or other mechanisms
        internal_input = np.zeros_like(self.x)  # Replace with actual logic
        for connected_field, weight in self.connected_fields:
            internal_input += weight * connected_field.u_field  # Example of using connected fields
        return internal_input

    def add_connection(self, field, weight=0.0):
        self.connected_fields.append((field, weight))
