import numpy as np
import lightkurve as lk
from astropy.timeseries import BoxLeastSquares
from types import SimpleNamespace

# 0) (Optional) Trim to a single Kepler quarter to reduce span (or keep your lc if you like)
# If you've already got lc, you can just subset a 120-day window:
# lc is your stitched, NaN-removed light curve
lc = lk.search_lightcurve("KIC 11446443", mission="Kepler", author="Kepler", exptime=1800, quarter=2)\
        .download().remove_nans()

# 1) Preprocess + gently bin (~1 hour) to halve points without smearing a ~2–3 h transit
clean = (lc.remove_outliers(sigma=5)
           .normalize()
           .flatten(window_length=301)
           .bin(time_bin_size=1.0/24.0))   # 1 hour in days

# Arrays as float32 to cut RAM
t  = clean.time.value.astype("float32")
y  = clean.flux.value.astype("float32")
dy = (clean.flux_err.value.astype("float32")
      if getattr(clean, "flux_err", None) is not None else None)

bls = BoxLeastSquares(t, y, dy)

# 2) Memory-safe BLS: coarse → fine, chunked in frequency and looping durations
def bls_autopower_chunked(bls, pmin, pmax, durations_days, nfreq=20000, chunks=8):
    # Frequency grid (even in frequency is best for BLS)
    fmin, fmax = 1.0/pmax, 1.0/pmin
    freq = np.linspace(fmin, fmax, nfreq, dtype="float64")
    periods = 1.0 / freq

    # Collect periodogram as "max over durations" per period (autopower-like)
    per_out, pow_out = [], []
    t0_out, dur_out, depth_out = [], [], []

    edges = np.linspace(0, nfreq, chunks+1, dtype=int)
    for a, b in zip(edges[:-1], edges[1:]):
        p_chunk = periods[a:b]
        max_pow = np.full(p_chunk.shape, -np.inf, dtype="float64")
        best_t0 = np.zeros_like(p_chunk, dtype="float64")
        best_dep = np.zeros_like(p_chunk, dtype="float64")
        best_dur = np.zeros_like(p_chunk, dtype="float64")

        for d in durations_days:
            # Ensure we don’t violate "max duration < min period" rule
            if d >= p_chunk.min():
                continue
            try:
                res = bls.power(p_chunk, d, objective="snr")
            except TypeError:
                res = bls.power(p_chunk, d)  # older astropy fallback

            pw = np.array(res["power"])
            upd = pw > max_pow
            if np.any(upd):
                max_pow[upd] = pw[upd]
                best_t0[upd] = np.array(res["transit_time"])[upd]
                best_dep[upd] = np.array(res["depth"])[upd]
                best_dur[upd] = d

        per_out.append(p_chunk)
        pow_out.append(max_pow)
        t0_out.append(best_t0)
        dur_out.append(best_dur)
        depth_out.append(best_dep)

    period = np.concatenate(per_out)
    power  = np.concatenate(pow_out)
    t0     = np.concatenate(t0_out)
    dur    = np.concatenate(dur_out)
    depth  = np.concatenate(depth_out)

    # Best index
    i = int(np.nanargmax(power))
    return (SimpleNamespace(period=period, power=power),
            dict(best_period=float(period[i]),
                 best_t0=float(t0[i]),
                 best_duration=float(dur[i]),
                 best_depth=float(depth[i]),
                 best_power=float(power[i])))

# ---- Stage 1: coarse (keeps RAM tiny)
dur_coarse_h = np.array([1.0, 2.0, 3.0], dtype="float32")   # hours
dur_coarse   = dur_coarse_h / 24.0
res1, best1 = bls_autopower_chunked(bls, pmin=0.5, pmax=10.0,
                                    durations_days=dur_coarse,
                                    nfreq=15000, chunks=6)

p0 = best1["best_period"]

# ---- Stage 2: fine around the coarse peak
win = 0.05 * p0  # ±5%
pmin = max(0.5, p0 - win)
pmax = p0 + win

dur_fine_h = np.linspace(0.75, 3.5, 9, dtype="float32")  # 0.75–3.5 h
dur_fine   = dur_fine_h / 24.0
# Make sure longest duration < min period
dur_fine = dur_fine[dur_fine < pmin]

results, best = bls_autopower_chunked(bls, pmin=pmin, pmax=pmax,
                                      durations_days=dur_fine,
                                      nfreq=20000, chunks=8)

best_period   = best["best_period"]
best_t0       = best["best_t0"]
best_duration = best["best_duration"]
best_depth    = best["best_depth"]
best_snr_like = best["best_power"]  # SNR objective

print(f"Best P={best_period:.6f} d, T0={best_t0:.6f}, Dur={best_duration*24:.2f} h, "
      f"Depth={best_depth*100:.3f} %, SNR~{best_snr_like:.2f}")
