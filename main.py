# main.py

from fields.field import Field
from fields.plotter import Plotter
from fields.utils import simultaneous_integration, save_final_state

if __name__ == "__main__":
    # Define kernel parameters, field parameters, and external input parameters
    kernel_pars = (1, 0.5, 0.9)
    field_pars = (80, 100, 0.1, 0.1, .85)  # x_lim, t_lim, dx, dt, theta
    # Define external input parameters for sequence_memory
    external_input_pars1 = [
        (0.0, 1.5, 10, 15),  # Input 1 parameters: center, width, active_start, active_end
        (30.0, 1.0, 20, 25)  # Input 2 parameters
    ]

    # Define external input parameters for field2
    external_input_pars2 = [
        (50.0, 1.5, 10, 15)  # Input for field2
    ]

    # Create fields
    sequence_memory = Field(kernel_pars, field_pars, external_input_pars1, name="Sequence Memory")
    field2 = Field(kernel_pars, field_pars, external_input_pars2, name="Field2")

    # Add connection (example)
    sequence_memory.add_connection(field2, weight=0.5)  # Connect field2 to sequence_memory

    # Mode: Learning
    mode = "learning"  # Change this to "recall" for the other mode
    if mode == "learning":
        simultaneous_integration([sequence_memory])

        # Save the final state of sequence_memory
        save_final_state(sequence_memory.history_u[-1, :], sequence_memory.name)  # Save the last time step

    elif mode == "recall":
        simultaneous_integration([sequence_memory, field2])

    # Plot final states of the fields
    plotter = Plotter([sequence_memory, field2])
    plotter.plot_final_states()
