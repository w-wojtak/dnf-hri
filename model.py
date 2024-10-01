# model.py

from fields.field import Field
from fields.utils import (
    simultaneous_integration,
    save_final_state,
    save_external_input_params,
    load_external_input_params,
)
from fields.plotter import Plotter

# Constants
KERNEL_SM = (1, 0.7, 0.9)
KERNEL_ACTION = (1.5, 0.9, 0.0)
KERNEL_WM = (1.5, 0.5, 0.75)
FIELD_PARS = (80, 100, 0.1, 0.1)  # x_lim, t_lim, dx, dt

# Define external input parameters for sequence_memory
EXTERNAL_INPUT_PARS1 = [
    (0.0, 3.0, 1.5, 10, 15),  # Input 1 parameters: center, amplitude, width, active_start, active_end
    (30.0, 3.0, 1.5, 30, 35),  # Input 2 parameters
    (-40.0, 3.0, 1.5, 50, 55),
]

def create_sequence_memory():
    """Create and return the Sequence Memory field."""
    return Field(
        KERNEL_SM,
        FIELD_PARS,
        EXTERNAL_INPUT_PARS1,
        tau_h=20,
        name="Sequence Memory",
        field_type="sequence_memory",
        theta=1.5,
    )

def create_recall_fields():
    """Create and return the Action Onset and Working Memory fields."""
    action_onset = Field(
        KERNEL_ACTION,
        FIELD_PARS,
        None,  # No external inputs
        tau_h=20,
        h_0=0,
        name="Action Onset",
        field_type="decision",
        theta=1,
    )

    working_memory = Field(
        KERNEL_WM,
        FIELD_PARS,
        None,  # No external inputs
        h_0=-1.0,
        name="Working Memory",
        theta=0.5,
    )

    return action_onset, working_memory

def run_learning_mode():
    """Execute the learning mode."""
    sequence_memory = create_sequence_memory()
    simultaneous_integration([sequence_memory])

    # Plot final states of the fields
    plotter = Plotter([sequence_memory])
    plotter.plot_final_states()

    # Save the final state of sequence_memory
    save_final_state(sequence_memory.history_u[-1, :], sequence_memory.name)

    # Save the parameters of external inputs
    save_external_input_params(EXTERNAL_INPUT_PARS1)

    # Extract input centers and plot the evolution of fields' activities
    input_centers = [param[0] for param in EXTERNAL_INPUT_PARS1]
    plotter.plot_activity_at_input_centers(input_centers, interval=10)

    # Animate activity
    plotter.animate_activity(FIELD_PARS, interval=10)

def run_recall_mode():
    """Execute the recall mode."""
    action_onset, working_memory = create_recall_fields()

    # Add connections
    working_memory.add_connection(action_onset, weight=1.0, connection_params={'threshold': 1})
    action_onset.add_connection(working_memory, weight=-5.0, connection_params={'threshold': 0.5})

    simultaneous_integration([action_onset, working_memory])

    # Plot final states of the fields
    plotter = Plotter([action_onset, working_memory])
    plotter.plot_final_states()

    # Load external input parameters and extract input centers
    external_input_params = load_external_input_params()
    input_centers = [param[0] for param in external_input_params]

    # Plot the evolution of fields' activities at the specified input centers
    plotter.plot_activity_at_input_centers(input_centers, interval=10)

    # Animate activity
    plotter.animate_activity(FIELD_PARS, interval=10)