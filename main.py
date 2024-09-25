from fields.field import Field
from fields.simulator import Simulator
from fields.plotter import Plotter
from fields.field_network import FieldNetwork

if __name__ == "__main__":
    # Set up simulation parameters
    simulation_length = 100  # Define the length of the simulation
    use_plotting = True  # Toggle Plotter on or off

    # Create fields, network, etc.
    field1 = Field(...)
    field2 = Field(...)

    # Create and run simulator
    simulator = Simulator([field1, field2], simulation_length)
    simulator.run()

    # Optionally plot results
    if use_plotting:
        plotter = Plotter(simulator)
        plotter.plot()
