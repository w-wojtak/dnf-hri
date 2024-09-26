import numpy as np
import matplotlib.pyplot as plt

class Plotter:
    def __init__(self, fields):
        self.fields = fields  # Directly pass fields instead of simulator

    def plot_final_states(self):
        num_fields = len(self.fields)
        cols = 2  # Number of columns for subplots
        rows = (num_fields + 1) // cols  # Dynamically calculate rows based on number of fields

        fig, axs = plt.subplots(rows, cols, figsize=(10, 5 * rows))

        # If there's only one subplot, make axs iterable
        if num_fields == 1:
            axs = [axs]

        for i, field in enumerate(self.fields):
            ax = axs.flat[i]  # Use flat to handle both 1D and 2D axs
            x_lim, _, dx, _, _ = field.field_pars
            x = np.arange(-x_lim, x_lim + dx, dx)

            ax.plot(x, field.history_u[-1, :])
            ax.set_title(f'Final state of {field.name}')
            ax.set_xlabel('x')
            ax.set_ylabel('u(x)')
            ax.set_xlim(-x_lim, x_lim)

        plt.tight_layout()
        plt.show()
