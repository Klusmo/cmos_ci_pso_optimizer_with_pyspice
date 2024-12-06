import numpy as np
import matplotlib.pyplot as plt

def display_circuit(circuit):
    print("\n ------------------------------------------ \n")
    print(f"Differential Pair Circuit: ")
    print(circuit)
    print("\n ------------------------------------------ \n")


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
