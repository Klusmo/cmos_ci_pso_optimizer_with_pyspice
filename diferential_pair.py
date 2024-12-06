import matplotlib.pyplot as plt
import numpy as np
from enum import Enum

import PySpice.Logging.Logging as Logging
logger = Logging.setup_logging()

from PySpice.Doc.ExampleTools import find_libraries
from PySpice.Probe.Plot import plot
from PySpice.Spice.Library import SpiceLibrary
from PySpice.Spice.Netlist import Circuit, SubCircuitFactory
from PySpice.Unit import u_A, u_F, u_Hz, u_kHz, u_mV, u_mV, u_mA, u_mV, u_nA, u_Ohm, u_pF, u_s, u_uA, u_um, u_us, u_V, u_nm, u_ms, u_GHz

# Define the channel lenght as minimum size of of the technology (45nm)
MINIMAL_TECNOLOGY_SIZE = (45*3)@u_nm
MAX_CHANNEL_WIDTH = 500
EPSILON = 10
VDD = 5
FITNESS_VALUES = []

class TransistorType(Enum):
    NMOS = 'nmos'
    PMOS = 'pmos'


class DifferentialPair(SubCircuitFactory):
    NAME = 'DifferentialPair'
    NODES = ('vdd', 'agnd', 'vin_plus', 'vin_minus', 'vout')

    def __init__(
            self,
            reference_current: float = 1@u_mA,
            pmos_active_load_channel_width  = MINIMAL_TECNOLOGY_SIZE,
            pmos_active_load_channel_lenght = MINIMAL_TECNOLOGY_SIZE,
            nmos_channel_width  = MINIMAL_TECNOLOGY_SIZE,
            nmos_channel_lenght = MINIMAL_TECNOLOGY_SIZE,
        ):
        super().__init__()
        
        # M <name> <drain node> <gate node> <source node> <bulk/substrate node>
        self.MOSFET(1, 'vdd', 'vcg', 'vcg', 'vdd', model='pmos', w=pmos_active_load_channel_width, l=pmos_active_load_channel_lenght)
        self.MOSFET(2, 'vdd', 'vcg', 'vout', 'vdd', model='pmos', w=pmos_active_load_channel_width, l=pmos_active_load_channel_lenght)

        self.MOSFET(3, 'vout', 'vin_minus', 'vcs', 'agnd', model='nmos', w=nmos_channel_width, l=nmos_channel_lenght)
        self.MOSFET(4, 'vcg', 'vin_plus', 'vcs', 'agnd', model='nmos', w=nmos_channel_width, l=nmos_channel_lenght)

        self.I('1', 'vcs', 'agnd', reference_current)

        self.I1.plus.add_current_probe(self)
        self.M1.drain.add_current_probe(self)
        self.M2.drain.add_current_probe(self)
        self.M3.drain.add_current_probe(self)
        self.M4.drain.add_current_probe(self)


def differential_pair_testbench(
        vdd=5,
        vcm=2.5,
        reference_current=1@u_mA,
        pmos_active_load_channel_width=MINIMAL_TECNOLOGY_SIZE,
        pmos_active_load_channel_lenght=MINIMAL_TECNOLOGY_SIZE,
        nmos_channel_width=MINIMAL_TECNOLOGY_SIZE,
        nmos_channel_lenght=MINIMAL_TECNOLOGY_SIZE,
    ):
    circuit = Circuit('Differential Pair')
    circuit.include("p045/p045_cmos_models_tt.inc")

    # Intantiate Voltage sources
    circuit.V('dd', 'vdd', circuit.gnd, u_V(vdd))
    circuit.V('cm', 'vcm', circuit.gnd, u_V(vcm))
    # Instantiate Voltage sources
    circuit.SinusoidalVoltageSource('in_plus', 'vin_plus', "vcm", amplitude=0.5@u_mV, frequency=1@u_kHz, ac_magnitude=1)
    circuit.SinusoidalVoltageSource('in_minus', "vcm", 'vin_minus', amplitude=0.5@u_mV, frequency=1@u_kHz)
    
    # Instantiate the Differential Pair
    circuit.subcircuit(
        DifferentialPair(
            reference_current=u_uA(reference_current),
            pmos_active_load_channel_width=u_nm(pmos_active_load_channel_width),
            pmos_active_load_channel_lenght=u_nm(pmos_active_load_channel_lenght),
            nmos_channel_width=u_nm(nmos_channel_width),
            nmos_channel_lenght=u_nm(nmos_channel_lenght)
        )
    )
    # NODES = ('vdd', 'agnd', 'vin_plus', 'vin_minus', 'vout')
    circuit.X(1, 'DifferentialPair', 'vdd', circuit.gnd, 'vin_plus', 'vin_minus', 'vout')

    # load
    circuit.C('load', 'vout', circuit.gnd, 30@u_pF)

    return circuit


def projectedDifferentialPair(
    vdd=3,
    vcm=1.5,
    reference_current=30@u_uA,
    pmos_active_load_channel_width=171@u_nm,
    pmos_active_load_channel_lenght=200@u_nm,
    nmos_channel_width=155@u_nm,
    nmos_channel_lenght=200@u_nm,
    compare_with_ltspice=False
):
    """
    Runs the differential pair testbench and displays the results
    the default values are the manually projected values of the differential pair, on LTSpice
    """
    default_values = {
        vdd: 3,
        vcm: 1.5,
        reference_current: 30,
        pmos_active_load_channel_width: 171,
        pmos_active_load_channel_lenght: 200,
        nmos_channel_width: 155,
        nmos_channel_lenght: 200
    }

    circuit = differential_pair_testbench(**default_values)
    display_circuit(circuit)

    print(f"Builiding Simulator")
    simulator = circuit.simulator(temperature=27, nominal_temperature=27, save_currents=True)
    op = simulator.operating_point()
    transient = simulator.transient(step_time=10@u_us, end_time=10@u_ms)
    ac_analysis = simulator.ac(start_frequency=1@u_Hz, stop_frequency=1@u_GHz, number_of_points=50,  variation='dec')

    display_operating_point(op)

    vin_diff = transient['vin_plus'] - transient['vin_minus']
    vout_delta = np.abs(np.max(transient['vout']) - np.min(transient['vout']))
    vin_diff_delta = np.abs(np.max(vin_diff) - np.min(vin_diff))
    gain = vout_delta/vin_diff_delta

    plot_transient_analisis(transient)
    display_transient_results(transient)

    gain = np.abs(ac_analysis['vout'].as_ndarray())
    phase = np.angle(ac_analysis['vout'].as_ndarray(), deg=True)
    # cut-off frequency, where the gain is equal to -3 dB (1/sqrt(2)) of the maximum gain
    cut_off_frequency = ac_analysis.frequency[np.argmax(gain < (np.max(gain) / np.sqrt(2)))]
    # unit gain frequency, where the gain is equal to 0 dB
    unit_gain_frequency = ac_analysis.frequency[np.argmin(np.abs(gain - 1))]

    display_ac_analysis_frequencies(ac_analysis)
    plot_bode_analysis(ac_analysis)


def compare_diferential_pais(
    vdd=3,
    vcm=1.5,
    reference_current=30,
    pmos_active_load_channel_width=171,
    pmos_active_load_channel_lenght=200,
    nmos_channel_width=155,
    nmos_channel_lenght=200,
    compare_with_ltspice=False
):
    """
    Runs the differential pair testbench and displays the results
    the default values are the manually projected values of the differential pair, on LTSpice
    """
    default_values = {
        "vdd": 3,
        "vcm": 1.5,
        "reference_current": 30,
        "pmos_active_load_channel_width": 171,
        "pmos_active_load_channel_lenght": 200,
        "nmos_channel_width": 155,
        "nmos_channel_lenght": 200
    }

    default_circuit = differential_pair_testbench(**default_values)
    # display_circuit(default_circuit)

    circuit = differential_pair_testbench(
        vdd=vdd,
        vcm=vcm,
        reference_current=reference_current,
        pmos_active_load_channel_width=pmos_active_load_channel_width,
        pmos_active_load_channel_lenght=pmos_active_load_channel_lenght,
        nmos_channel_width=nmos_channel_width,
        nmos_channel_lenght=nmos_channel_lenght
    )

    print(f"          Comparing Differential Pairs")
    print(f"|{' '*40}| Projected | Optimized |")
    print(f"|{'-'*40}|-----------|-----------|")
    for i, key in enumerate(default_values):
        if i in [0, 1]:
            unit = " V"
        elif i in [2]:
            unit = "uA"
        else:
            unit = "nm"
        print(f"| {key:39}| {default_values[key]:6} {unit} | {eval(key):6} {unit} |")
    print()
    print("\n ------------------------------------------ \n") 

    # print(f"          Building Simulators")
    default_simulator = default_circuit.simulator(temperature=27, nominal_temperature=27, save_currents=True)
    default_op, default_transient, default_ac_analysis = simulator_analysis(default_simulator)

    simulator = circuit.simulator(temperature=27, nominal_temperature=27, save_currents=True)
    op, transient, ac_analysis = simulator_analysis(simulator)
    
    # display_operating_point(default_op)

    # plot_transient_analisis(default_transient)
    # display_transient_results(default_transient)

    # display_ac_analysis_frequencies(default_ac_analysis)
    # plot_bode_analysis(default_ac_analysis)
    # display_operating_point(op)

    # plot_transient_analisis(transient)
    display_transient_results(default_transient, transient)

    display_ac_analysis_frequencies(default_ac_analysis, ac_analysis)
    plot_2_bode_analysis(default_ac_analysis, ac_analysis)


def simulator_analysis(default_simulator):
    op = default_simulator.operating_point()
    transient = default_simulator.transient(step_time=10@u_us, end_time=10@u_ms)
    ac_analysis = default_simulator.ac(start_frequency=1@u_Hz, stop_frequency=1@u_GHz, number_of_points=50,  variation='dec')
    return op,transient,ac_analysis


def plot_bode_analysis(ac_analysis):
    gain = np.abs(ac_analysis['vout'].as_ndarray())
    phase = np.angle(ac_analysis['vout'].as_ndarray(), deg=True)
    # cut-off frequency, where the gain is equal to -3 dB (1/sqrt(2)) of the maximum gain
    cut_off_frequency = ac_analysis.frequency[np.argmax(gain < (np.max(gain) / np.sqrt(2)))]
    # unit gain frequency, where the gain is equal to 0 dB
    unit_gain_frequency = ac_analysis.frequency[np.argmin(np.abs(gain - 1))]

    plt.figure(figsize=(10, 5))
    plt.title('Bode Plot')
    plt.semilogx(ac_analysis.frequency, 20*np.log10(gain), color='b')
    plt.semilogx(ac_analysis.frequency, phase, color='b', linestyle='--')
    plt.axvline(float(cut_off_frequency), color='r', linestyle='--')
    plt.axvline(float(unit_gain_frequency), color='r', linestyle='--')
    plt.text(float(cut_off_frequency)*1.5, 42, f'F-3dB: {cut_off_frequency}')
    plt.text(float(unit_gain_frequency)*1.5, 2, f'F-0dB: {unit_gain_frequency}')
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Gain [dB]')
    plt.grid()
    plt.show()

def plot_2_bode_analysis(default_ac_analysis, ac_analysis):
    gain = np.abs(ac_analysis['vout'].as_ndarray())
    phase = np.angle(ac_analysis['vout'].as_ndarray(), deg=True)
    # cut-off frequency, where the gain is equal to -3 dB (1/sqrt(2)) of the maximum gain
    cut_off_frequency = ac_analysis.frequency[np.argmax(gain < (np.max(gain) / np.sqrt(2)))]
    # unit gain frequency, where the gain is equal to 0 dB
    unit_gain_frequency = ac_analysis.frequency[np.argmin(np.abs(gain - 1))]

    default_gain = np.abs(default_ac_analysis['vout'].as_ndarray())
    default_phase = np.angle(default_ac_analysis['vout'].as_ndarray(), deg=True)
    default_cut_off_frequency = default_ac_analysis.frequency[np.argmax(default_gain < (np.max(default_gain) / np.sqrt(2)))]
    default_unit_gain_frequency = default_ac_analysis.frequency[np.argmin(np.abs(default_gain - 1))]

    plt.figure(figsize=(16, 8))
    plt.subplot(2, 1, 1)
    plt.title('Comparação diagrama de Bode - Manual x Otimizado')
    plt.semilogx(default_ac_analysis.frequency, 20*np.log10(gain), color='b')
    plt.semilogx(default_ac_analysis.frequency, default_phase, color='b', linestyle='--')
    plt.axvline(float(default_cut_off_frequency), color='r', linestyle='--')
    plt.axvline(float(default_unit_gain_frequency), color='r', linestyle='--')
    plt.text(float(default_cut_off_frequency)*1.5, 40, f'f_H: {float(default_cut_off_frequency):.2f}', fontsize=8)
    plt.text(float(default_unit_gain_frequency)*1.5, 0, f'f_u: {float(default_unit_gain_frequency):.2f}', fontsize=8)
    plt.legend(['Manual'])
    
    plt.subplot(2, 1, 2)
    plt.semilogx(ac_analysis.frequency, 20*np.log10(gain), color='b')
    plt.semilogx(ac_analysis.frequency, phase, color='b', linestyle='--')
    plt.axvline(float(cut_off_frequency), color='r', linestyle='--')
    plt.axvline(float(unit_gain_frequency), color='r', linestyle='--')
    plt.text(float(cut_off_frequency)*1.5, 40, f'f_H: {float(cut_off_frequency):.2f}', fontsize=8)
    plt.text(float(unit_gain_frequency)*1.5, 0, f'f_u: {float(unit_gain_frequency):.2f}', fontsize=8)
    plt.xlabel('Frequência [Hz]')
    plt.ylabel('Ganho [dB]')
    plt.legend(['Otimizado'])
    plt.show()


def display_ac_analysis_frequencies(projected_ac_analysis, ac_analysis=None):
    if ac_analysis is None:
        projected_gain = np.abs(projected_ac_analysis['vout'].as_ndarray())
        # cut-off frequency, where the gain is equal to -3 dB (1/sqrt(2)) of the maximum gain
        projected_cut_off_frequency = projected_ac_analysis.frequency[np.argmax(projected_gain < (np.max(projected_gain) / np.sqrt(2)))]
        # unit gain frequency, where the gain is equal to 0 dB
        projected_unit_gain_frequency = projected_ac_analysis.frequency[np.argmin(np.abs(projected_gain - 1))]

        print("    AC Analysis")
        print("    Calculating the cut-off frequency and unit gain frequency")
        print(f"    Cut-off Frequency: {projected_cut_off_frequency}")
        print(f"    Unit Gain Frequency: {projected_unit_gain_frequency}")
        print("\n ------------------------------------------ \n")
    else:
        gain = np.abs(ac_analysis['vout'].as_ndarray())
        # cut-off frequency, where the gain is equal to -3 dB (1/sqrt(2)) of the maximum gain
        cut_off_frequency = ac_analysis.frequency[np.argmax(gain < (np.max(gain) / np.sqrt(2)))]
        # unit gain frequency, where the gain is equal to 0 dB
        unit_gain_frequency = ac_analysis.frequency[np.argmin(np.abs(gain - 1))]

        projected_gain = np.abs(projected_ac_analysis['vout'].as_ndarray())
        # cut-off frequency, where the gain is equal to -3 dB (1/sqrt(2)) of the maximum gain
        projected_cut_off_frequency = projected_ac_analysis.frequency[np.argmax(projected_gain < (np.max(projected_gain) / np.sqrt(2)))]
        # unit gain frequency, where the gain is equal to 0 dB
        projected_unit_gain_frequency = projected_ac_analysis.frequency[np.argmin(np.abs(projected_gain - 1))]

        print("    AC Analysis")
        print("    Comparing the cut-off frequency and unit gain frequency")
        print()
        print(f"    Projected Cut-off Frequency: {projected_cut_off_frequency}")
        print(f"    Projected Unit Gain Frequency: {projected_unit_gain_frequency}")
        print()
        print(f"    Optimized Cut-off Frequency: {cut_off_frequency}")
        print(f"    Optimized Unit Gain Frequency: {unit_gain_frequency}")
        print("\n ------------------------------------------ \n")


def display_transient_results(projected_transient, transient=None):
    if transient is None:
        vout_delta = np.abs(np.max(projected_transient['vout']) - np.min(projected_transient['vout']))
        vin_diff = projected_transient['vin_plus'] - projected_transient['vin_minus']
        vin_diff_delta = np.abs(np.max(vin_diff) - np.min(vin_diff))
        gain = vout_delta/vin_diff_delta

        print(f"   Projected Gain: {gain}")
        print("\n ------------------------------------------ \n")
    else:
        vin_diff = transient['vin_plus'] - transient['vin_minus']
        vout_delta = np.abs(np.max(transient['vout']) - np.min(transient['vout']))
        vin_diff_delta = np.abs(np.max(vin_diff) - np.min(vin_diff))
        gain = vout_delta/vin_diff_delta

        projected_vin_diff = projected_transient['vin_plus'] - projected_transient['vin_minus']
        projected_vout_delta = np.abs(np.max(projected_transient['vout']) - np.min(projected_transient['vout']))
        projected_vin_diff_delta = np.abs(np.max(projected_vin_diff) - np.min(projected_vin_diff))
        projected_gain = projected_vout_delta/projected_vin_diff_delta

        # print("Transient analysis")
        # print(f"Vout Delta: {vout_delta}")
        # print(f"Vin Diff Delta: {vin_diff_delta}")
        print(f"   Projected Gain: {projected_gain}")
        print(f"   Optimized Gain: {gain}")
        print("\n ------------------------------------------ \n")


def display_operating_point(default_op, op=None):
    if op is None:
        print("\n ------------------------------------------ \n")
        print("         Operating Point")
        for node in default_op.nodes:
            print(f"Node: {node} - {default_op[node][0]}")
        print()
        for branch in default_op.branches.values():
            print(f"Branch: {branch} - {branch[0]}")

        print("\n ------------------------------------------ \n")
    else:
        print("\n ------------------------------------------ \n")
        print("         Comparing Operating Point")
        # build the comparison table
        print("|        Nodes        | Default OP | Current OP |")
        print("|---------------------|------------|------------|")
        for node in default_op.nodes:
            print(f"| {node:20}| {default_op[node][0].value:>10.2f} | {op[node][0].value:>10.2f} |")

        print()
        print("|       Branches      | Default OP | Current OP |")
        print("|---------------------|------------|------------|")
        for branch in default_op.branches:
            print(f"| {branch:20}| {default_op.branches[branch][0].value:>10.2e} | {op.branches[branch][0].value:>10.2e} |")
        print("\n ------------------------------------------ \n")


def display_circuit(circuit):
    print("\n ------------------------------------------ \n")
    print(f"Differential Pair Circuit: ")
    print(circuit)
    print("\n ------------------------------------------ \n")


def plot_transient_analisis(default_analysis, analysis=None):
    if analysis is None:
        input_signal = default_analysis['vin_plus'] - default_analysis['vin_minus']
        plt.figure(figsize=(10, 5))
        plt.subplot(2, 1, 1)
        plt.title('Vin Plus')
        plt.plot(default_analysis.time, input_signal)
        plt.xlabel('Time [s]')
        plt.ylabel('Vin Plus [V]')
        plt.grid()
        plt.subplot(2, 1, 2)
        plt.title('Vout')
        plt.plot(default_analysis.time, default_analysis['vout'])
        plt.xlabel('Time [s]')
        plt.ylabel('Vout [V]')
        plt.grid()
        plt.show()
    else:
        dft_input_signal = default_analysis['vin_plus'] - default_analysis['vin_minus']
        input_signal = analysis['vin_plus'] - analysis['vin_minus']
        plt.figure(figsize=(10, 5))
        plt.subplot(2, 1, 1)
        plt.title('Vin Plus')
        plt.plot(default_analysis.time, dft_input_signal, label='Default', color='b')
        plt.plot(default_analysis.time, input_signal, label='Current', color='r', linestyle='--')
        plt.xlabel('Time [s]')
        plt.ylabel('Vin Plus [V]')
        plt.grid()
        plt.subplot(2, 1, 2)
        plt.title('Vout')
        plt.plot(default_analysis.time, default_analysis['vout'], label='Default', color='b')
        plt.plot(default_analysis.time, analysis['vout'], label='Current', color='r', linestyle='--')
        plt.xlabel('Time [s]')
        plt.ylabel('Vout [V]')
        plt.grid()
        plt.show()


def particle_swarm_optimization(parameters: dict):
    """
    This function will optimize the channel width of the CMOS inverter using a particle swarm optimization algorithm
    """
    FITNESS_VALUES.clear()
    # Define the search space
    params_ranges = [ v for _, v in parameters.items()]
    search_space = np.array(params_ranges)
    
    best_fitness_per_epoch = []

    # Define the number of particles
    n_particles = 6

    # Define the number of iterations
    n_iterations = 100

    # Define the inertia weight
    w = 0.5

    # Define the cognitive weight
    c1 = 2

    # Define the social weight
    c2 = 1.5

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
        # print(f'Iteration: {iteration}/{n_iterations}')
        # print(f'Best Global Fitness: {best_global_fitness}')
        # print(f'Best Global Position: {best_global_position}')
        # print('')

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


def fitness(particle, display=False):
    """
    This function will evaluate the fitness of a particle
    """
    circuit = differential_pair_testbench(
        vdd=particle[0],
        vcm=particle[0]/2.0,
        reference_current=particle[1]@u_uA,
        pmos_active_load_channel_width=particle[2]@u_nm,
        pmos_active_load_channel_lenght=200@u_nm,
        nmos_channel_width=particle[3]@u_nm,
        nmos_channel_lenght=200@u_nm
    )
    area = particle[2] * 200 + particle[3] * 200

    simulator = circuit.simulator(temperature=27, nominal_temperature=27, save_currents=True)
    try:
        op = simulator.operating_point()
    except:
        return np.inf

    vout = op.vout[0].value
    target = np.float64(particle[0]/2.0)
    polarization_fitness = np.abs(vout - target)

    try :
        transient = simulator.transient(step_time=10@u_us, end_time=5@u_ms)
    except:
        return np.inf
    
    vin_diff = transient['vin_plus'] - transient['vin_minus']
    vout_delta = np.abs(np.max(transient['vout']) - np.min(transient['vout']))
    vin_diff_delta = np.abs(np.max(vin_diff) - np.min(vin_diff))
    gain = float(vout_delta/vin_diff_delta)

    # need to maimize the gain
    gain_fitness = 1/gain

    try:
        ac_analysis = simulator.ac(start_frequency=1@u_Hz, stop_frequency=1@u_GHz, number_of_points=10,  variation='dec')
    except:
        return np.inf
    
    ac_gain = np.abs(ac_analysis['vout'].as_ndarray())
    # cut-off frequency, where the gain is equal to -3 dB (1/sqrt(2)) of the maximum gain
    cut_off_frequency = ac_analysis.frequency[np.argmax(ac_gain < (np.max(ac_gain) / np.sqrt(2)))]
    # unit gain frequency, where the gain is equal to 0 dB
    unit_gain_frequency = ac_analysis.frequency[np.argmin(np.abs(ac_gain - 1))]
    
    # need to maximize the cut-off frequency an the unit gain frequency
    cut_off_fitness = 1/cut_off_frequency
    
    unit_gain_fitness = 1/unit_gain_frequency

    weights = [
        1000, 
        200, 
        10000, 
        1000000,
        1/20000
    ]

    if display:
        print("\n ------------------------------------------ \n")
        print(f"VDD: {particle[0]}")
        print(f"Reference Current: {particle[1]}")
        print(f"PMOS Active Load Channel Width: {particle[2]}")
        print(f"NMOS Channel Width: {particle[3]}")
        print()

        print(f"Vout: {vout}")
        print(f"Gain: {gain}")
        print(f"Cut-off Frequency: {cut_off_frequency}")
        print(f"Unit Gain Frequency: {unit_gain_frequency}")
        print(f"Area: {area}")
        print()

        print(f"Polarization Fitness: {polarization_fitness * weights[0]}")
        print(f"Gain Fitness: {gain_fitness * weights[1]}")
        print(f"Cut-off Fitness: {cut_off_fitness * weights[2]}")
        print(f"Unit Gain Fitness: {unit_gain_fitness * weights[3]}")
        print(f"Area: {area * weights[4]}")       
        print("\n ------------------------------------------ \n")


    return (
            polarization_fitness * weights[0]
          + gain_fitness * weights[1]
          + cut_off_fitness * weights[2]
          + unit_gain_fitness * weights[3]
          + area * weights[4]
    )


def plot_epochs(best_fitness_per_epoch):
    plt.figure(figsize=(10, 5))
    plt.plot(best_fitness_per_epoch)
    plt.xlabel('Iteration')
    plt.ylabel('Fitness')
    plt.grid()
    plt.show()


def display_best_results(best_global_position, best_fitness_per_epoch):
    print("\n ------------------------------------------ \n")
    print(f"Best Global Position: ")
    print(f"VDD: {best_global_position[0]}")
    print(f"Reference Current: {best_global_position[1]}")
    print(f"PMOS Active Load Channel Width: {best_global_position[2]}")
    print(f"NMOS Channel Width: {best_global_position[3]}")
    print(f"Best Global Fitness: {best_fitness_per_epoch[-1]}")
    print("\n ------------------------------------------ \n")

def epoch_convergence(parameters):
    pso_results = []
    for i in range(30):
        print(f"Run {i}")

        best_global_position, best_fitness_per_epoch = particle_swarm_optimization(parameters)
        print(f"Best Global Position: {best_global_position}")
        print(f"Best Fitness: {best_fitness_per_epoch[-1]}")
        print()

        pso_results.append((best_global_position, best_fitness_per_epoch))

    # over 30 runs, get the average of the best global position and the mean epoch of when ths PSO converged
    best_global_position_mean = np.mean([result[0] for result in pso_results], axis=0)
    best_fitness_per_epoch_mean = np.mean(
        [np.argmin(result[1]) for result in pso_results], axis=0
    )

    print(f"Beast Global Position Mean: {best_global_position_mean}")
    print(f"Best Fitness Epoch Mean: {best_fitness_per_epoch_mean}")
    # Beast Global Position Mean: [  2.06666667  18.26666667 176.7        135.        ]
    # Best Fitness Epoch Mean: 17.366666666666667

def optimize_differential_pair():
    MINIMAL_TECNOLOGY_SIZE = (45*3)@u_nm
    MAX_CHANNEL_WIDTH = 200

    parameters = {
        'vdd': (1, 5),
        'reference_current': (30, 31),
        'pmos_active_load_channel_width': (MINIMAL_TECNOLOGY_SIZE.value, MAX_CHANNEL_WIDTH),
        'nmos_channel_width': (MINIMAL_TECNOLOGY_SIZE.value, MAX_CHANNEL_WIDTH),
    }
    best_global_position, best_fitness_per_epoch = particle_swarm_optimization(parameters)

    display_best_results(best_global_position, best_fitness_per_epoch)
    fitness(best_global_position, display=True)
    plot_epochs(best_fitness_per_epoch)


if __name__ == '__main__':
    # projectedDifferentialPair()
    # optimize_differential_pair()


    """
    OPTIMIZATION RESULTS

    ------------------------------------------

    Best Global Position:
    VDD: 3
    Reference Current: 30
    PMOS Active Load Channel Width: 172
    NMOS Channel Width: 168
    Best Global Fitness: 10.799201623816963

    ------------------------------------------


    ------------------------------------------

    VDD: 3
    Reference Current: 30
    PMOS Active Load Channel Width: 172
    NMOS Channel Width: 168

    Vout: 1.5015847518043657
    Gain: 60.66858609463531
    Cut-off Frequency: 7943.2823472428345 Hz
    Unit Gain Frequency: 794328.2347242847 Hz
    Area: 68000

    Polarization Fitness: 1.5847518043656805
    Gain Fitness: 3.296598995862955
    Cut-off Fitness: 1.2589254117941642
    Unit Gain Fitness: 1.2589254117941622
    Area: 3.4000000000000004

    ------------------------------------------


    """
    compare_diferential_pais(
        vdd=3,
        vcm=1.5,
        reference_current=30,
        pmos_active_load_channel_width=172,
        pmos_active_load_channel_lenght=200,
        nmos_channel_width=168,
        nmos_channel_lenght=200,
        compare_with_ltspice=False
    )
