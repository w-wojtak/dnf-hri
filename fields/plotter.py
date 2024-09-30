import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation



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
        """
        Animate the time evolution of activity u(x,t) and optional input with time step display.
        """
        x_lim, _, dx, dt, _ = field_pars  # Include dt to calculate the time in seconds
        x = np.arange(-x_lim, x_lim + dx, dx)

        # Compute min and max activity values across all time steps
        activity_min = np.min(self.fields[0].history_u)
        activity_max = np.max(self.fields[0].history_u)

        # If input_flag is true, adjust based on external input limits as well
        if input_flag and hasattr(self.fields[0], 'external_input'):
            input_min = np.min(self.fields[0].external_input)
            input_max = np.max(self.fields[0].external_input)
            y_min = min(activity_min, input_min)
            y_max = max(activity_max, input_max)
        else:
            y_min = activity_min
            y_max = activity_max

        fig, ax = plt.subplots()
        ax.set_xlim(-x_lim, x_lim)
        ax.set_ylim(y_min - 0.1 * abs(y_min), y_max + 0.1 * abs(y_max))  # Set y limits with a small margin
        ax.set_xlabel('x')
        ax.set_ylabel('Activity')

        line1, = ax.plot([], [], lw=2, label='Activity')
        if input_flag:
            line2, = ax.plot([], [], lw=2, label='Input')

        ax.legend()

        # Add a text element to display the current time step
        time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)

        def init():
            line1.set_data([], [])
            if input_flag:
                line2.set_data([], [])
            time_text.set_text('')
            return (line1, line2, time_text) if input_flag else (line1, time_text)

        def update(frame):
            time_step = frame * interval
            activity = self.fields[0].history_u[time_step, :]
            line1.set_data(x, activity)

            if input_flag:
                inputs = self.fields[0].external_input[time_step, :]
                line2.set_data(x, inputs)

            # Update the time_text with the current time step (converted to time)
            time_text.set_text(f"Time: {time_step * dt:.2f} s")
            return (line1, line2, time_text) if input_flag else (line1, time_text)

        ani = animation.FuncAnimation(fig, update, frames=len(self.fields[0].history_u) // interval,
                                      init_func=init, blit=True, repeat=False)

        plt.show()
