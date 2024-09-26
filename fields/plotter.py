import numpy as np
import matplotlib.pyplot as plt

class Plotter:
    def __init__(self, fields):
        self.fields = fields  # Initialize with the list of fields

    def plot_final_states(self):
        num_fields = len(self.fields)  # Use self.fields instead of self.simulator
        fig, axes = plt.subplots(num_fields, 1, figsize=(8, 4 * num_fields))

        # Ensure axes is an array even when there's only one subplot
        if num_fields == 1:
            axes = [axes]

        for i, field in enumerate(self.fields):
            axes[i].plot(field.x, field.activity[-1, :])  # Assuming 'activity' has the final state
            axes[i].set_xlim(-field.x_lim, field.x_lim)
            axes[i].set_xlabel('x')
            axes[i].set_ylabel('Activity')
            axes[i].set_title(field.name)  # Set the title to the field name

        plt.tight_layout()
        plt.show()



