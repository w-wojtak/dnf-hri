# main.py

from fields.field import Field
from fields.plotter import Plotter
from fields.utils import simultaneous_integration, save_final_state, load_sequence_memory

if __name__ == "__main__":
    # Define kernel parameters, field parameters, and external input parameters
    kernel_sm = (1, 0.5, 0.9)
    kernel_action = (1, 1, 0.5)
    field_pars = (80, 100, 0.1, 0.1, .85)  # x_lim, t_lim, dx, dt, theta
    # Define external input parameters for sequence_memory
    external_input_pars1 = [
        (0.0, 1.5, 10, 15),  # Input 1 parameters: center, width, active_start, active_end
        (30.0, 1.0, 40, 45)  # Input 2 parameters
    ]

    # Create fields
    sequence_memory = Field(kernel_sm, field_pars, external_input_pars1, field_type="sequence_memory")
    action_onset = Field(kernel_action, field_pars, [(0, 1, 0, 0)], field_type="decision")

    # Add connection (example)
    sequence_memory.add_connection(action_onset, weight=0.0)  # Connect field2 to sequence_memory

    # Mode: Learning
    mode = "recall"  # "learning" / "recall" mode choice

    if mode == "learning":
        simultaneous_integration([sequence_memory])

        # Save the final state of sequence_memory
        save_final_state(sequence_memory.history_u[-1, :], sequence_memory.name)  # Save the last time step

    elif mode == "recall":
        simultaneous_integration([action_onset])

    # Plot final states of the fields
    plotter = Plotter([sequence_memory, action_onset])
    plotter.plot_final_states()
