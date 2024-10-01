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

    def animate_activity(self, field_pars, interval=10, input_flag=False, plot_inputs=False):
        num_fields = len(self.fields)
        fig, axes = plt.subplots(1, num_fields, figsize=(6 * num_fields, 4))  # Side-by-side layout

        if num_fields == 1:
            axes = [axes]

        for i, field in enumerate(self.fields):
            max_activity = field.activity.max()
            min_activity = field.activity.min()

            axes[i].set_xlim(-field.x_lim, field.x_lim)
            axes[i].set_ylim(min_activity, max_activity)
            axes[i].set_xlabel('x')
            axes[i].set_ylabel('Activity')
            axes[i].set_title(f"{field.name} - Time Step: 0")

        # Line objects for activity and optionally inputs
        lines = [axes[i].plot(field.x, field.activity[0, :], label='Activity')[0] for i, field in
                 enumerate(self.fields)]

        if plot_inputs:
            input_lines = []
            for i, field in enumerate(self.fields):
                if np.any(field.history_external_input) or np.any(field.history_internal_input):
                    ext_line, = axes[i].plot(field.x, field.history_external_input[0, :], label='External Input',
                                             linestyle='--')
                    int_line, = axes[i].plot(field.x, field.history_internal_input[0, :], label='Internal Input',
                                             linestyle=':')
                    input_lines.append((ext_line, int_line))

        for t in range(0, len(self.fields[0].t), interval):
            for i, field in enumerate(self.fields):
                lines[i].set_ydata(field.activity[t, :])
                axes[i].set_title(f"{field.name} - Time Step: {t}")

                if plot_inputs:
                    ext_line, int_line = input_lines[i]
                    ext_line.set_ydata(field.history_external_input[t, :])
                    int_line.set_ydata(field.history_internal_input[t, :])

            plt.draw()
            plt.pause(0.3)

        plt.show()


