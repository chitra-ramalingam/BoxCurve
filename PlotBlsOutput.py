import numpy as np
import matplotlib.pyplot as plt


class BlsOutput:
    def __init__(self):
        pass

    def plot_bls_output(self, clean, results, bls, best_period, best_t0, best_duration):

        # --- Helpers to ensure plain floats/arrays (Quantities -> floats) ---
        bp  = float(np.array(best_period))    # days
        bt0 = float(np.array(best_t0))        # BKJD days
        bd  = float(np.array(best_duration))  # days

        per_vals = getattr(results.period, "value", results.period)
        pow_vals = getattr(results.power, "value", results.power)

        # 1) Cleaned light curve + BLS box model in time domain
        t = clean.time.value
        y = clean.flux.value

        plt.figure()
        plt.plot(t, y, ".", markersize=2, alpha=0.6, label="Cleaned flux")
        # Astropy BLS model at the best solution
        model = bls.model(t, bp, bd, bt0)
        plt.plot(t, model, linewidth=2, label="BLS model")
        plt.xlabel("Time (BKJD, days)")
        plt.ylabel("Normalized flux")
        plt.title("Cleaned Light Curve with BLS Model")
        plt.legend()
        plt.show()

        # 2) BLS periodogram (power vs period) with peak marker
        plt.figure()
        plt.plot(per_vals, pow_vals)
        plt.xscale("log")
        plt.axvline(bp, linestyle="--", alpha=0.7)
        plt.xlabel("Period (days)")
        plt.ylabel("BLS power (SNR objective)")
        plt.title("BLS Periodogram")
        plt.show()

        # 3) Phase-folded light curve at detected ephemeris (hours from mid-transit)
        folded = clean.fold(period=bp, epoch_time=bt0)
        phase_hours = folded.phase.value * bp * 24.0

        plt.figure()
        plt.scatter(phase_hours, folded.flux.value, s=3, alpha=0.5, label="Folded")
        half_dur_h = (bd * 24.0) / 2.0
        plt.axvspan(-half_dur_h, half_dur_h, alpha=0.2, label="Transit window")

        # Optional: binned overlay for clarity (Lightkurve FoldedLightCurve.bin)
        try:
            folded_bin = folded.bin(time_bin_size=bd/6.0)  # bin ~6 samples across the transit
            phase_bin_h = folded_bin.phase.value * bp * 24.0
            plt.plot(phase_bin_h, folded_bin.flux.value, linewidth=2, label="Binned")
        except Exception:
            pass

        plt.xlabel("Phase (hours from mid-transit)")
        plt.ylabel("Normalized flux")
        plt.title(f"Folded LC  P={bp:.6f} d, T0={bt0:.6f}, Dur={bd*24:.2f} h")
        plt.legend()
        plt.show()
