import lightkurve as lk
import numpy as np
from BlsCreator import BlsCreator
from PlotBlsOutput import BlsOutput

# Fetch PDCSAP light curve for KIC 11446443 (TrES-2A)
# lc = lk.search_lightcurve("KIC 11446443", mission="Kepler", author="Kepler", exptime=1800)\
#        .download_all().stitch().remove_nans()

# # Take the first 20 rows (as you asked)
# sample = lc[:20].to_table()[['time', 'flux', 'flux_err', 'quality']]

# # Quick metrics (very rough)
# period = 2.4706132  # days (from catalogs)
# folded = lc.fold(period=period, epoch_time=lc.time[0])
# baseline = np.nanmedian(folded.flux.value)
# dip_min = np.nanmin(folded.flux.value)
# depth_pct = (baseline - dip_min) / baseline * 100
# # crude duration: width around minimum below (baseline - 0.5 * depth)
# threshold = baseline - 0.5*(baseline - dip_min)
# in_transit = folded.flux.value < threshold
# duration_hours = (np.sum(in_transit)/len(folded.flux.value)) * period * 24
# snr = (baseline - dip_min) / np.nanmedian(lc.flux_err.value)

# print(sample)           # first 20 rows
# print(depth_pct, duration_hours, period, snr)


def main():

    creator = BlsCreator()
    clean_fold =creator.run_bls()
    plotter = BlsOutput()
    plotter.plot_bls_output(clean_fold[0], clean_fold[1], clean_fold[2], clean_fold[3], clean_fold[4], clean_fold[5])
    print("Welcome to Exo-Planets")

if __name__ == "__main__":
    main()