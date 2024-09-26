from fields.field import Field
from fields.plotter import Plotter
from fields.utils import simultaneous_integration

if __name__ == "__main__":
    # Define kernel parameters, field parameters, and external input parameters
    kernel_pars = (1, 0.5, 0.9)
    field_pars = (80, 100, 0.1, 0.1, .85)  # x_lim, t_lim, dx, dt, theta
    external_input_pars = (0.0, 1.5, 10, 15)  # center, width, active_start, active_end

    # Create two fields with external inputs
    field1 = Field(kernel_pars, field_pars, external_input_pars, name="Field1")
    field2 = Field(kernel_pars, field_pars, external_input_pars, name="Field2")

    # Connect field1 to field2 with a weight of 0.5
    field2.add_connection(field1, weight=0.0)

    # Run simultaneous integration for both fields
    simultaneous_integration([field1, field2])

    # Plot final states of both fields
    plotter = Plotter([field1, field2])
    plotter.plot_final_states()
