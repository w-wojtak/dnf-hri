import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time


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

    def animate_activity(self, field_pars, interval=10, input_flag=False):
        num_fields = len(self.fields)
        fig, axes = plt.subplots(1, num_fields, figsize=(6 * num_fields, 4))  # Side-by-side layout

        if num_fields == 1:
            axes = [axes]

        # Prepare the plot and set axis limits individually for each field
        for i, field in enumerate(self.fields):
            max_activity = field.activity.max()  # Max value for the specific field
            min_activity = field.activity.min()  # Min value for the specific field

            axes[i].set_xlim(-field.x_lim, field.x_lim)
            axes[i].set_ylim(min_activity, max_activity)  # Set y-limits individually
            axes[i].set_xlabel('x')
            axes[i].set_ylabel('Activity')
            axes[i].set_title(f"{field.name} - Time Step: 0")

        # Line objects for each field
        lines = [axes[i].plot(field.x, field.activity[0, :])[0] for i, field in enumerate(self.fields)]

        # Animate over time steps
        for t in range(0, len(self.fields[0].t), interval):
            for i, field in enumerate(self.fields):
                # Update the y-data for each line (i.e., activity at time t)
                lines[i].set_ydata(field.activity[t, :])
                axes[i].set_title(f"{field.name} - Time Step: {t}")

            # Redraw the figure with updated data
            plt.draw()
            plt.pause(0.1)

        plt.show()

