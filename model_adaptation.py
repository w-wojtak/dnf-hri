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
EXTERNAL_INPUT_PARS_SM = [
    (0.0, 3.0, 1.5, 10, 15),  # Input 1 parameters: center, amplitude, width, active_start, active_end
    (30.0, 3.0, 1.5, 30, 35),  # Input 2 parameters
    (-40.0, 3.0, 1.5, 50, 55),
]

EXTERNAL_INPUT_PARS_H = [
    (0.0, 3.0, 1.5, 20, 25),  # center, amplitude, width, active_start, active_end
    (30.0, 3.0, 1.5, 38, 43),
    (-40.0, 3.0, 1.5, 70, 75),
]


def create_sequence_memory():
    """Create and return the Sequence Memory field."""
    return Field(
        KERNEL_SM,
        FIELD_PARS,
        EXTERNAL_INPUT_PARS_SM,
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

    human_feedback = Field(
        KERNEL_ACTION,
        FIELD_PARS,
        EXTERNAL_INPUT_PARS_H,
        tau_h=20,
        h_0=0,
        name="Human feedback",
        theta=1,
    )

    return action_onset, working_memory, human_feedback


def run_learning_mode(plot_options, input_centers):
    """Execute the learning mode."""
    sequence_memory = create_sequence_memory()
    simultaneous_integration([sequence_memory], input_centers)
    # simultaneous_integration(fields, input_centers)

    # Initialize plotter
    plotter = Plotter([sequence_memory])

    # Plot final states if specified
    if plot_options.get("plot_final_states", False):
        plotter.plot_final_states()

    # Save the final state of sequence_memory
    save_final_state(sequence_memory.history_u[-1, :], sequence_memory.name)

    # Save the parameters of external inputs
    save_external_input_params(EXTERNAL_INPUT_PARS_SM)

    # Extract input centers and plot the evolution of fields' activities
    input_centers = [param[0] for param in EXTERNAL_INPUT_PARS_SM]

    # Plot activity at input centers if specified
    if plot_options.get("plot_activity_at_input_centers", False):
        plotter.plot_activity_at_input_centers(input_centers, interval=10)

    # Animate activity if specified
    if plot_options.get("animate_activity", False):
        plotter.animate_activity(FIELD_PARS, interval=10)


def run_recall_mode(plot_options, input_centers):
    """Execute the recall mode."""
    action_onset, working_memory, human_feedback = create_recall_fields()

    # Add connections
    working_memory.add_connection(action_onset, weight=1.0, connection_params={'threshold': 1})
    action_onset.add_connection(working_memory, weight=-5.0, connection_params={'threshold': 0.5})

    # simultaneous_integration([action_onset, working_memory, human_feedback])
    fields = [action_onset, working_memory, human_feedback]  # List of Field instances
    simultaneous_integration(fields, input_centers)

    # Initialize plotter
    plotter = Plotter([action_onset, working_memory, human_feedback])

    # Plot final states if specified
    if plot_options.get("plot_final_states", False):
        plotter.plot_final_states()

    # Load external input parameters and extract input centers
    external_input_params = load_external_input_params()
    input_centers = [param[0] for param in external_input_params]

    # Plot activity at input centers if specified
    if plot_options.get("plot_activity_at_input_centers", False):
        plotter.plot_activity_at_input_centers(input_centers, interval=10)

    # Animate activity if specified
    if plot_options.get("animate_activity", False):
        plotter.animate_activity(FIELD_PARS, interval=10)
