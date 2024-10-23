from model import run_learning_mode, run_recall_mode
from fields.utils import load_external_input_params

def main(mode, plot_options):
    """Run the model in the specified mode with given plot options."""

    # Load external input parameters and extract input centers
    if mode == "recall":
        external_input_params = load_external_input_params()
        input_centers = [param[0] for param in external_input_params]

    if mode == "learning":
        run_learning_mode(plot_options)
    elif mode == "recall":
        run_recall_mode(plot_options, input_centers)


if __name__ == "__main__":
    # Mode: Learning or Recall
    mode = "recall"  # Change to "learning" or "recall"

    # Define plot options
    plot_options = {
        "plot_final_states": True,
        "plot_activity_at_input_centers": True,
        "animate_activity": True,
    }

    main(mode, plot_options)
