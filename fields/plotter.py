import numpy as np
import matplotlib.pyplot as plt

class Plotter:
    def __init__(self, fields):
        self.fields = fields  # Store fields

    def plot_final_states(self):
        num_fields = len(self.fields)

        # Create subplots
        fig, axes = plt.subplots(num_fields, 1, figsize=(8, 4 * num_fields))

        # Plot each field in a separate subplot
        for i, field in enumerate(self.fields):
            x_lim = field.x_lim  # Access the x_lim from the field
            ax = axes[i]

            ax.plot(field.x, field.history_u[-1, :])  # Plot the final state
            ax.set_xlim(-x_lim, x_lim)
            ax.set_xlabel('x')
            ax.set_ylabel('u(x)')
            ax.set_title(f'Final State of {field.name}')  # Use the field name

        plt.tight_layout()
        plt.show()

