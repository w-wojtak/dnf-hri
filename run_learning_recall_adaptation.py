from model_adaptation import run_learning_mode, run_recall_mode


def main(mode, plot_options):
    """Run the model in the specified mode with given plot options."""
    if mode == "learning":
        run_learning_mode(plot_options)
    elif mode == "recall":
        run_recall_mode(plot_options)


if __name__ == "__main__":
    # Mode: Learning or Recall
    mode = "learning"  # Change to "learning" or "recall" as needed

    # Define plot options
    plot_options = {
        "plot_final_states": True,
        "plot_activity_at_input_centers": True,
        "animate_activity": False,
    }

    main(mode, plot_options)
