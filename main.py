from model import run_learning_mode, run_recall_mode

if __name__ == "__main__":
    # Mode: Learning or Recall
    mode = "recall"  # Change to "learning" or "recall" as needed

    if mode == "learning":
        run_learning_mode()
    elif mode == "recall":
        run_recall_mode()
