def simultaneous_integration(fields, input_centers):
    """
    Integrates multiple fields over time and monitors action_onset at input_centers.
    :param fields: List of Field objects
    :param input_centers: Positions (x values) to monitor in the action_onset field
    """
    num_time_steps = len(fields[0].t)

    # Find the fields
    action_onset_field = next((field for field in fields if field.name == "Action Onset"), None)
    robot_feedback_field = next((field for field in fields if field.name == "Robot Feedback"), None)

    for i in range(num_time_steps):
        # Step 1: Integrate all fields
        for field in fields:
            if field.name != "Robot Feedback":
                field.integrate_single_step(i)  # Regular integration for other fields

        # Step 2: After each integration step, monitor the action_onset field
        threshold_crossings = None
        if action_onset_field and input_centers is not None:
            threshold_crossings = action_onset_field.monitor_action_onset(input_centers, i)

        # Step 3: Pass threshold_crossings to Human feedback field and integrate it
        if robot_feedback_field:
            robot_feedback_field.integrate_single_step(i, threshold_crossings)
