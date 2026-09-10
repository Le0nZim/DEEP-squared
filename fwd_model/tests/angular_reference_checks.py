"""Independent numerical checks of the scattering equations (NumPy only).

This does NOT execute or validate MATLAB source. Run test_scattering.m in
MATLAB for implementation verification. These calculations reproduce the
original angular defects and compare the corrected laws with analytical
solid-angle, Snell, and HG-moment identities. No tissue PSF is simulated.

Usage: python fwd_model/tests/angular_reference_checks.py --output result.json
"""
import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    seed, n = 20260910, 1_000_000
    rng = np.random.default_rng(seed)
    uniform = rng.random(n)
    phi = 2 * np.pi * rng.random(n)
    original_z = np.cos(np.pi * uniform)
    isotropic_z = 2 * uniform - 1

    def second_moments(z):
        r = np.sqrt(1 - z * z)
        return [float(np.mean((r * np.cos(phi)) ** 2)),
                float(np.mean((r * np.sin(phi)) ** 2)),
                float(np.mean(z * z))]

    old_moments = second_moments(original_z)
    new_moments = second_moments(isotropic_z)
    assert max(abs(np.array(new_moments) - 1 / 3)) < 0.003
    assert old_moments[2] > 0.49

    na, medium_index = 1.0, 1.33
    ratio = na / medium_index
    expected = (1 - np.sqrt(1 - ratio * ratio)) / 2
    collected = (isotropic_z > 0) & (np.sqrt(1-isotropic_z**2) <= ratio)
    fraction = float(np.mean(collected))
    assert abs(fraction - expected) < 6 * np.sqrt(expected*(1-expected)/n)

    # Isolate the reversed-NA mask, then combine it with the original launch.
    old_mask_on_isotropic = (isotropic_z > 0) & (isotropic_z <= ratio)
    old_mask_on_original = (original_z > 0) & (original_z <= ratio)

    # Physical acceptance at a plane: n_t sin(alpha_t) = n_m sin(alpha_m).
    incident_deg = np.array([0.0, 30.0, 50.0])
    incidence = np.deg2rad(incident_deg)
    transmitted_sin = 1.5 * np.sin(incidence)
    transmitted = transmitted_sin < 1
    transmitted_deg = [float(np.rad2deg(np.arcsin(s))) if s < 1 else None
                       for s in transmitted_sin]
    with np.errstate(divide="ignore", invalid="ignore"):
        original_sin2 = 1.5**2 * np.cos(incidence)**2
        original_uz2 = np.sin(incidence)**2 * original_sin2/(1-original_sin2)
    original_transmits = original_uz2 >= 0
    # The original treats the exactly axial case inconsistently, but the
    # 30- and 50-degree classifications are unambiguously reversed.
    assert not bool(original_transmits[1]) and bool(transmitted[1])
    assert bool(original_transmits[2]) and not bool(transmitted[2])

    g = 0.9
    sample = rng.random(n)
    t = (1-g*g)/(1-g+2*g*sample)
    hg_cos = (1+g*g-t*t)/(2*g)
    hg_p1 = float(np.mean(hg_cos))
    hg_p2 = float(np.mean((3*hg_cos**2-1)/2))
    assert abs(hg_p1-g) < 0.003
    assert abs(hg_p2-g*g) < 0.003

    # Check the small-g stable rearrangement against extended precision.
    q = np.linspace(-1, 1, 10001, dtype=np.longdouble)
    algebra_error = 0.0
    for small_g in [np.longdouble("0.0001"), np.longdouble("-0.0001")]:
        reference = (1+small_g**2 - ((1-small_g**2)/(1+small_g*q))**2)/(2*small_g)
        stable = (2*q + small_g*(q*q+3) + 2*small_g**2*q
                  + small_g**3*(q*q-1))/(2*(1+small_g*q)**2)
        algebra_error = max(algebra_error, float(np.max(np.abs(reference-stable))))
    assert algebra_error < 1e-10

    theta = np.deg2rad(np.array([0,10,40,45,48,50,60,80,90]))
    angle_table = [{"angle_deg": int(round(np.rad2deg(a))),
                    "original_accepts_limit": bool(np.cos(a) <= ratio),
                    "correct_accepts": bool(np.sin(a) <= ratio)} for a in theta]

    result = {
        "scope": "Numerical equation checks only; not a MATLAB or GPU execution, or a tissue PSF experiment.",
        "seed": seed, "samples": n, "status": "PASS",
        "launch_second_moments_xyz": {"original": old_moments,
                                      "corrected": new_moments,
                                      "isotropic_target": [1/3]*3},
        "objective": {"NA": na, "medium_index": medium_index,
                      "correct_max_angle_deg": float(np.rad2deg(np.arcsin(ratio))),
                      "original_min_angle_deg": float(np.rad2deg(np.arccos(ratio))),
                      "angle_table": angle_table},
        "ballistic_fraction_of_all_launched_photons": {
            "corrected_numeric": fraction, "correct_analytic": float(expected),
            "original_NA_on_isotropic_source_numeric": float(np.mean(old_mask_on_isotropic)),
            "original_NA_on_isotropic_source_analytic": ratio/2,
            "original_launch_and_NA_numeric": float(np.mean(old_mask_on_original)),
            "original_launch_and_NA_analytic": float(np.arcsin(ratio)/np.pi)},
        "snell_nt_1p5_nm_1": {"incident_deg": incident_deg.tolist(),
                             "transmitted_deg": transmitted_deg,
                             "correct_transmits": transmitted.tolist(),
                             "original_transmits_oblique_rays": original_transmits[1:].tolist()},
        "unmodified_HG_g_0p9": {"mean_cos": hg_p1, "target_mean_cos": g,
                               "mean_P2": hg_p2, "target_mean_P2": g*g},
        "small_g_rearrangement_max_error": algebra_error,
    }
    report = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.write_text(report, encoding="utf-8")
    print(report, end="")


if __name__ == "__main__":
    main()
