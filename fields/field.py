import numpy as np
from fields.utils import kernel_osc, external_input_function  # Import from utils.py

class Field:
    def __init__(self, kernel_pars, field_pars, external_input_pars_list, tau_h=100, h_0=0, input_flag=True, name="Field"):
        self.name = name
        self.kernel_pars = kernel_pars
        self.x_lim, self.t_lim, self.dx, self.dt, self.theta = field_pars  # Unpack field_pars
        self.tau_h = tau_h
        self.h_0 = h_0
        self.input_flag = input_flag
        self.external_input_pars_list = external_input_pars_list  # List of inputs

        # Spatial and temporal grids
        self.x = np.arange(-self.x_lim, self.x_lim + self.dx, self.dx)
        self.t = np.arange(0, self.t_lim + self.dt, self.dt)

        # State history and current state initialization
        self.history_u = np.zeros([len(self.t), len(self.x)])
        self.u_field = h_0 * np.ones(np.shape(self.x))
        self.h_u = h_0 * np.ones(np.shape(self.x))

        # Fourier transform of the kernel function
        self.w_hat = np.fft.fft(kernel_osc(self.x, *self.kernel_pars))

        # List of connected fields (internal inputs)
        self.connected_fields = []

    def get_external_input(self, t):
        return external_input_function(self.x, t, self.external_input_pars_list)




    def add_connection(self, field, weight):
        self.connected_fields.append((field, weight))

    # def get_external_input(self, t):
    #     return external_input_function(self.x, t, self.external_input_pars)

    def get_internal_input(self, i):
        internal_input = np.zeros_like(self.u_field)
        for field, weight in self.connected_fields:
            internal_input += weight * field.history_u[i, :]
        return internal_input

    def integrate_single_step(self, i):
        external_input = self.get_external_input(self.t[i])
        internal_input = self.get_internal_input(i)

        f = np.heaviside(self.u_field - self.theta, 1)
        f_hat = np.fft.fft(f)
        conv = self.dx * np.fft.ifftshift(np.real(np.fft.ifft(f_hat * self.w_hat)))

        self.h_u += self.dt / self.tau_h * f
        self.u_field += self.dt * (-self.u_field + conv + external_input + internal_input + self.h_u)

        self.history_u[i, :] = self.u_field
