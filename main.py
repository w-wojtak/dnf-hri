# main.py

from fields.field import Field
from fields.plotter import Plotter
from fields.utils import simultaneous_integration, load_sequence_memory
from fields.utils import save_final_state, save_external_input_params, load_external_input_params


if __name__ == "__main__":
    # Define kernel parameters, field parameters, and external input parameters
    kernel_sm = (1, 0.7, 0.9)
    kernel_action = (1.5, 0.9, 0.0)
    kernel_wm = (1.5, 0.5, 0.75)
    field_pars = (80, 100, 0.1, 0.1)  # x_lim, t_lim, dx, dt
    # Define external input parameters for sequence_memory
    external_input_pars1 = [
        (0.0, 3.0, 1.5, 10, 15),  # Input 1 parameters: center, amplitude, width, active_start, active_end
        (30.0, 3.0, 1.5, 30, 35),  # Input 2 parameters
        (-40.0, 3.0, 1.5, 50, 55)
    ]

    # Add connection (example)
    # sequence_memory.add_connection(action_onset, weight=0.0)  # Connect field2 to sequence_memory

    # Mode: Learning
    mode = "recall"  # "learning" / "recall" mode choice

    if mode == "learning":
        # Create fields
        sequence_memory = Field(kernel_sm, field_pars, external_input_pars1, tau_h=20,
                                name="Sequence Memory",
                                field_type="sequence_memory",
                                theta = 1.5
                                )

        simultaneous_integration([sequence_memory])

        # Plot final states of the fields
        plotter = Plotter([sequence_memory])
        plotter.plot_final_states()

        # Plot activity evolution over time
        # plotter.animate_activity(field_pars, interval=10,
        #                          input_flag=False)  # Set input_flag=True if you want to plot inputs

        # Save the final state of sequence_memory
        save_final_state(sequence_memory.history_u[-1, :], sequence_memory.name)  # Save the last time step

        # Save the parameters of external inputs
        save_external_input_params(external_input_pars1)

        # Extract input centers
        input_centers = [param[0] for param in
                         external_input_pars1]  # Assuming centers are the first element in the tuple

        # Plot the evolution of fields' activities at the specified input centers
        plotter.plot_activity_at_input_centers(input_centers, interval=10)

    elif mode == "recall":
        # Create fields
        action_onset = Field(kernel_action, field_pars, [(0, 0, 1, 0, 0)], tau_h=20, h_0=0,
                             name="Action Onset", field_type="decision", theta = 1)
        working_memory = Field(kernel_wm, field_pars, [(0, 0, 1, 0, 0)],
                               h_0=-1.0, name="Working Memory", theta = 0.5)

        # action_onset.plot_loaded_field()
        # Add connection: working_memory receives input from action_onset
        working_memory.add_connection(action_onset, weight= 1.0, connection_params={'threshold': 1})
        action_onset.add_connection(working_memory, weight= -5.0, connection_params={'threshold': 0.5})

        simultaneous_integration([action_onset, working_memory])

        # Plot final states of the fields
        plotter = Plotter([action_onset, working_memory])
        plotter.plot_final_states()

        # Plot activity evolution over time
        # plotter.animate_activity(field_pars, interval=10, plot_inputs=True)

        # Load external input parameters
        external_input_params = load_external_input_params()

        # Extract input centers
        input_centers = [param[0] for param in
                         external_input_params]  # Assuming centers are the first element in the tuple

        # Plot the evolution of fields' activities at the specified input centers
        plotter.plot_activity_at_input_centers(input_centers, interval=10)




