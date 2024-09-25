class Simulator:
    def __init__(self, fields, simulation_length):
        self.fields = fields
        self.simulation_length = simulation_length

    def run(self):
        # Code to run the simulation for the given length
        for field in self.fields:
            field.integrate()
