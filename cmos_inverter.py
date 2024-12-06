import matplotlib.pyplot as plt
import numpy as np

import PySpice.Logging.Logging as Logging
logger = Logging.setup_logging()

from PySpice.Doc.ExampleTools import find_libraries
from PySpice.Probe.Plot import plot
from PySpice.Spice.Library import SpiceLibrary
from PySpice.Spice.Netlist import Circuit, SubCircuitFactory
from PySpice.Unit import u_V, u_A, u_Ohm, u_F, u_s, u_Hz, u_nF, u_mV, u_kOhm, u_mA, u_uF, u_uA, u_kHz, u_mHz, u_nH, u_mOhm, u_pF, u_nm

# Define the channel lenght as minimum size of of the technology (45nm)
MINIMAL_TECNOLOGY_SIZE = 45@u_nm
EPSILON = 10
VDD = 5
MAX_CHANNEL_WIDTH = 500


class CMOSInverter(SubCircuitFactory):
    NAME = 'CMOSInverter'
    NODES = ('avdd', 'vin', 'vout', 'agnd')
    def __init__(
            self, 
            nmos_channel_width  = MINIMAL_TECNOLOGY_SIZE,
            nmos_channel_lenght = MINIMAL_TECNOLOGY_SIZE,
            pmos_channel_width  = MINIMAL_TECNOLOGY_SIZE,
            pmos_channel_lenght = MINIMAL_TECNOLOGY_SIZE,
        ):
        super().__init__()
        # M <name> <drain node> <gate node> <source node> <bulk/substrate node>
        self.MOSFET(1, 'vout', 'vin', 'agnd', 'agnd', model='nmos', w=nmos_channel_width, l=nmos_channel_lenght)
        self.MOSFET(2, 'avdd', 'vin', 'vout', 'avdd', model='pmos', w=pmos_channel_width, l=pmos_channel_lenght)


def cmos_inverter_testbench(vdd=3, channel_width=MINIMAL_TECNOLOGY_SIZE):
    circuit = Circuit('CMOS Inverter')
    circuit.include("p045/p045_cmos_models_tt.inc")

    # Intantiate Voltage sources
    circuit.V('dd', 'vdd', circuit.gnd, u_V(vdd))
    circuit.V('in', 'vin', circuit.gnd, u_V(vdd/2))

    # Instantiate the CMOS Inverter
    circuit.subcircuit(CMOSInverter(pmos_channel_width=u_nm(channel_width)))
    circuit.X(1, 'CMOSInverter', 'vdd', 'vin', 'vout', circuit.gnd)

    # Define the output load
    circuit.C('load', 'vout', circuit.gnd, 1@u_pF)
    return circuit


def parametric_sweep(vdd=VDD, best_channel_width=None):
    channel_width_values = np.linspace(MINIMAL_TECNOLOGY_SIZE.value, MAX_CHANNEL_WIDTH)
    op_vout = np.zeros(len(channel_width_values))
    for index, channel_width in enumerate(channel_width_values):
        circuit = cmos_inverter_testbench(vdd=vdd, channel_width=channel_width)
        simulator = circuit.simulator(temperature=27, nominal_temperature=27)
        op = simulator.operating_point()
        
        op_vout[index] = op.vout



    plt.plot(channel_width_values, op_vout)
    plt.axvline(best_channel_width, color='r', linestyle='--')

    l_ylim, u_ylim = plt.ylim()
    y_pos = l_ylim + ((u_ylim - l_ylim)/2)

    plt.text(best_channel_width + 10, y_pos, f'Best Channel Width: {best_channel_width}')
    plt.xlabel('Channel Width [nm]')
    plt.ylabel('Vout [V]')
    plt.title('Vout vs Channel Width')
    plt.show()


def particle_swarm_optimization():
    """
    This function will optimize the channel width of the CMOS inverter using a particle swarm optimization algorithm
    """
     # Define the search space
    search_space = [(MINIMAL_TECNOLOGY_SIZE.value, MAX_CHANNEL_WIDTH)]
    search_space = np.array(search_space)

    best_fitness_per_epoch = []

    # Define the number of particles
    n_particles = 10

    # Define the number of iterations
    n_iterations = 100

    # Define the inertia weight
    w = 0.5

    # Define the cognitive weight
    c1 = 2

    # Define the social weight
    c2 = 1

    # Initialize the particles, each assumed value is a integer
    particles = np.random.randint(search_space[:, 0], search_space[:, 1], (n_particles, search_space.shape[0]))

    # Initialize the best position of the particles
    best_positions = particles.copy()

    # Initialize the best global position
    best_global_position = particles[np.argmin(fitness(particle) for particle in particles)]
    best_global_fitness : np.float64 = np.inf

    # Initialize the velocity of the particles
    velocities = np.zeros((n_particles, search_space.shape[0]))

    same_result_count = 0
    for iteration in range(n_iterations):
        print(f'Iteration: {iteration}/{n_iterations}')
        print(f'Best Global Fitness: {best_global_fitness}')
        print(f'Best Global Position: {best_global_position}')
        print('')

        if same_result_count > 20:
            break
        
        best_iteration_fitness : np.float64 = best_global_fitness
        best_iteration_position = best_global_position
        for i, particle in enumerate(particles):
            # Update the velocity
            velocities[i] = w*velocities[i] + c1*np.random.rand()*(best_positions[i] - particle) + c2*np.random.rand()*(best_global_position - particle)

            # Update the position
            new_position = particle + velocities[i]
            particles[i] = np.clip(new_position, search_space[:, 0], search_space[:, 1])

            # Evaluate the fitness of the particle
            fitness_value = np.float64(fitness(particle))
            
            # Update the best position of the particle
            if fitness_value < fitness(best_positions[i]):
                best_positions[i] = particle

            # Update the best global position
            if fitness_value < best_iteration_fitness:
                best_iteration_fitness = fitness_value
                best_iteration_position = particle
        
        best_fitness_per_epoch.append(best_iteration_fitness)
        if  best_iteration_fitness < best_global_fitness:
            best_global_fitness = best_iteration_fitness
            best_global_position = best_iteration_position
            same_result_count = 0
            
        else:
            same_result_count += 1    

    return best_global_position, best_fitness_per_epoch


def fitness(particle, vdd: float = VDD):
    """
    This function will evaluate the fitness of a particle
    """
    circuit = cmos_inverter_testbench(vdd=vdd, channel_width=particle[0])
    simulator = circuit.simulator(temperature=27, nominal_temperature=27)
    op = simulator.operating_point()
    vout = op.vout[0].value

    target = np.float64(vdd/2.0)

    return np.abs(vout - target)


if __name__ == '__main__':
    pso_result, best_fitness_per_epoch = particle_swarm_optimization()
    print(pso_result)

    # Plot the best fitness per epoch
    plt.plot(best_fitness_per_epoch)
    plt.xlabel('Epoch')
    plt.ylabel('Best Fitness')
    plt.title('Best Fitness per Epoch')
    plt.show()

    parametric_sweep(best_channel_width=pso_result[0])