"""Scenario-based Monte Carlo model for an LKs nearest-mean contest.

The simulation is deliberately reproducible (fixed RNG seeds).  It samples
latent strategy counts, a shared-signal environment, and the aggregate
distribution of all other bids.  The component weights are illustrative stress
parameters, not estimates of LKs audience demographics.  Conditional
nearest-price occupancy is Rao-Blackwellized:
instead of materialising billions of individual bids, the exact discrete-CDF
probability of a closer bid and a Poisson approximation for equal-price ties
are integrated out.  This is materially more precise than sampling one winner
indicator per replication.

All prices are in CNY and all actions lie on a 0.01-CNY grid.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Scenario:
    # The 0.35/0.20 defaults carry the user-supplied tennis-interest/AI priors
    # into an auxiliary decision route.  They are proxies for latent bidding
    # strategies, not measured audience demographics or verified identities.
    upper_strategy_share: float = 0.35
    shared_signal_share: float = 0.20
    shared_upper_ratio: float = 1.30
    rho_shared: float = 0.10
    sigma_shared: float = 6.0
    exact_focus_share: float = 0.35
    mu_lower: float = 148.0
    sd_lower: float = 48.0
    mu_upper: float = 196.0
    sd_upper: float = 58.0
    shared_centers: tuple[float, ...] = (155.0, 160.0, 165.0, 170.0, 175.0)
    shared_weights: tuple[float, ...] = (0.10, 0.20, 0.40, 0.20, 0.10)
    shared_upper_shift: float = 10.0
    anchor_scale: float = 1.0   # deviations from the public 150-CNY anchor
    h1_lower: float = 0.12      # integer-yuan heaping
    h5_lower: float = 0.08      # 5-yuan heaping
    h1_upper: float = 0.10
    h5_upper: float = 0.10


DEFAULT_CANDIDATES = [
    160.00, 162.00, 163.00, 164.00, 164.50, 164.70, 164.74,
    164.75, 164.80, 164.81, 164.82, 164.83, 164.84, 164.85, 164.86,
    164.90, 165.00, 166.00, 168.00, 170.00,
]

# Final public count shown for LOT 18 after the activity deadline.
REFERENCE_N = 12_368

# The cent grid is localized in two stages.  The broad economic domain spans
# 5%-20% of the CNY 1,500 reference price.  A 0.13 phase shift keeps the
# coarse pass away from whole-yuan and half-yuan focal prices; it is a region
# locator, not the final action grid.  The final pass evaluates every cent.
VALUE_REFERENCE = 1_500.0
COARSE_LOWER_RATE = 0.05
COARSE_UPPER_RATE = 0.20
COARSE_PHASE = 0.13
COARSE_STEP = 0.25
COARSE_PLATEAU_FRACTION = 0.90
FINE_HALF_WIDTH = 1.00
FINE_STEP = 0.01
BOUNDARY_RELATIVE_UTILITY_MAX = 0.50


def norm_cdf(x: np.ndarray) -> np.ndarray:
    """Fast vector normal CDF, max absolute error about 7.5e-8."""
    x = np.asarray(x, dtype=np.float64)
    z = np.abs(x)
    t = 1.0 / (1.0 + 0.2316419 * z)
    poly = t * (0.319381530 + t * (-0.356563782 + t * (
        1.781477937 + t * (-1.821255978 + t * 1.330274429))))
    tail = np.exp(-0.5 * z * z) * 0.3989422804014327 * poly
    out = np.where(x >= 0.0, 1.0 - tail, tail)
    return np.clip(out, 0.0, 1.0)


def component_probs(m: Scenario) -> np.ndarray:
    """Lower/upper strategy crossed with independent/shared signal."""
    d = m.upper_strategy_share * min(
        0.95, m.shared_signal_share * m.shared_upper_ratio
    )
    c = m.shared_signal_share - d
    b = m.upper_strategy_share - d
    a = 1.0 - b - c - d
    if min(a, b, c, d) < 0:
        raise ValueError("Infeasible latent-strategy proportions")
    return np.array([a, b, c, d], dtype=np.float64)


def transformed_parameters(m: Scenario):
    k = m.anchor_scale
    anchor = 150.0
    mu_lower = anchor + k * (m.mu_lower - anchor)
    mu_upper = anchor + k * (m.mu_upper - anchor)
    centers = anchor + k * (np.asarray(m.shared_centers) - anchor)
    upper_shift = k * m.shared_upper_shift
    return mu_lower, mu_upper, centers, upper_shift


def scenario_expected_mean(m: Scenario) -> float:
    """Analytical population mean implied by one selected scenario."""
    mu_lower, mu_upper, centers, shift = transformed_parameters(m)
    probs = component_probs(m)
    component_means = np.array([
        mu_lower,
        mu_upper,
        np.dot(m.shared_weights, centers),
        np.dot(m.shared_weights, centers) + shift,
    ])
    return float(probs @ component_means)


def predictive_variance_decomposition(n_total: int, m: Scenario) -> dict:
    """Approximate mean variance from common and idiosyncratic uncertainty.

    This matches the aggregate generator before cent/integer heaping.  The
    common Dirichlet environment does not average away with a large N, while
    conditional individual dispersion does at the usual 1/N rate.
    """
    n_other = n_total - 1
    probs = component_probs(m)
    mu_lower, mu_upper, centers, shift = transformed_parameters(m)
    base_w = np.asarray(m.shared_weights, dtype=np.float64)
    shared_mean = float(base_w @ centers)
    center_variance = float(base_w @ (centers - shared_mean) ** 2)
    alpha0 = max(1e-6, 1.0 / m.rho_shared - 1.0)
    shared_mean_variance = center_variance / (alpha0 + 1.0)
    shared_share = float(probs[2] + probs[3])
    common_variance = shared_share ** 2 * shared_mean_variance

    # E[Var(shared bid | Dirichlet weights)].
    shared_conditional_variance = (
        center_variance - shared_mean_variance
        + (1.0 - m.exact_focus_share) * m.sigma_shared ** 2
    )
    expected_component_variances = np.array([
        m.sd_lower ** 2,
        m.sd_upper ** 2,
        shared_conditional_variance,
        shared_conditional_variance,
    ])
    expected_squared_means = np.array([
        mu_lower ** 2,
        mu_upper ** 2,
        shared_mean ** 2 + shared_mean_variance,
        (shared_mean + shift) ** 2 + shared_mean_variance,
    ])
    population_mean = scenario_expected_mean(m)
    expected_population_mean_squared = population_mean ** 2 + common_variance
    conditional_bid_variance = float(
        probs @ (expected_component_variances + expected_squared_means)
        - expected_population_mean_squared
    )
    idiosyncratic_mean_variance = conditional_bid_variance / n_other
    total_variance = common_variance + idiosyncratic_mean_variance
    return {
        "common_signal_variance": common_variance,
        "common_signal_sd": math.sqrt(common_variance),
        "idiosyncratic_mean_variance": idiosyncratic_mean_variance,
        "idiosyncratic_mean_sd": math.sqrt(idiosyncratic_mean_variance),
        "total_variance": total_variance,
        "total_sd": math.sqrt(total_variance),
        "common_variance_share": common_variance / total_variance,
        "note": "aggregate approximation before cent/integer heaping",
    }


def inclusive_grid(lower: float, upper: float, step: float) -> list[float]:
    """Build an inclusive decimal grid without cumulative arange drift."""
    count = int(round((upper - lower) / step))
    return np.round(lower + step * np.arange(count + 1), 2).tolist()


def coarse_localization_grid(value_ref: float = VALUE_REFERENCE) -> list[float]:
    """Phase-shifted broad grid used only to locate the objective region."""
    lower = value_ref * COARSE_LOWER_RATE + COARSE_PHASE
    upper = value_ref * COARSE_UPPER_RATE + COARSE_PHASE
    return inclusive_grid(lower, upper, COARSE_STEP)


def snapped_half_yuan_center(price: float) -> float:
    """Snap a coarse location to the nearest CNY 0.50 for an auditable window."""
    return round(round(price * 2.0) / 2.0, 2)


def coarse_peak_plateau(rows, fraction: float = COARSE_PLATEAU_FRACTION):
    """Return the contiguous near-optimal coarse region containing its peak."""
    utility = np.asarray(
        [row["unconditional_utility_v1500"] for row in rows], dtype=np.float64
    )
    best_index = int(np.argmax(utility))
    best_utility = float(utility[best_index])
    threshold = fraction * best_utility
    left_index = best_index
    right_index = best_index
    while left_index > 0 and utility[left_index - 1] >= threshold:
        left_index -= 1
    while right_index + 1 < len(rows) and utility[right_index + 1] >= threshold:
        right_index += 1
    lower = float(rows[left_index]["bid"])
    upper = float(rows[right_index]["bid"])
    return {
        "fraction": fraction,
        "lower": lower,
        "upper": upper,
        "midpoint": round((lower + upper) / 2.0, 3),
        "points": right_index - left_index + 1,
    }


def local_fine_grid(center: float, half_width: float) -> list[float]:
    return inclusive_grid(center - half_width, center + half_width, FINE_STEP)


def fine_boundary_audit(rows, threshold: float = BOUNDARY_RELATIVE_UTILITY_MAX):
    """Check that a local optimum is interior and both edges have decayed."""
    best = max(rows, key=lambda row: row["unconditional_utility_v1500"])
    left = rows[0]
    right = rows[-1]
    interior = left["bid"] < best["bid"] < right["bid"]
    passed = (
        interior
        and left["relative_utility"] < threshold
        and right["relative_utility"] < threshold
    )
    return {
        "left_bid": left["bid"],
        "left_relative_utility": left["relative_utility"],
        "right_bid": right["bid"],
        "right_relative_utility": right["relative_utility"],
        "interior_best": interior,
        "threshold": threshold,
        "passed": passed,
    }


def independent_cdf(k_cent: np.ndarray, mu: float, sd: float,
                    h1: float, h5: float) -> np.ndarray:
    """CDF of a normal strategy draw followed by cent/integer/5-yuan heaping."""
    k = np.asarray(k_cent, dtype=np.int64)
    p_cent = norm_cdf((((k + 0.5) / 100.0) - mu) / sd)
    max_integer_yuan = np.floor_divide(k, 100).astype(np.float64)
    p_int = norm_cdf((max_integer_yuan + 0.5 - mu) / sd)
    max_five_yuan = 5.0 * np.floor_divide(k, 500).astype(np.float64)
    p_five = norm_cdf((max_five_yuan + 2.5 - mu) / sd)
    return (1.0 - h1 - h5) * p_cent + h1 * p_int + h5 * p_five


def shared_signal_cdf(k_cent: np.ndarray, weights: np.ndarray,
                      centers: np.ndarray, sigma: float,
                      q_exact: float) -> np.ndarray:
    """Conditional CDF of a rounded shared-signal mixture."""
    k = np.asarray(k_cent, dtype=np.int64)
    threshold = (k.astype(np.float64) + 0.5) / 100.0
    residual = norm_cdf((threshold[:, None] - centers[None, :]) / sigma)
    center_cents = np.rint(centers * 100.0).astype(np.int64)
    exact = center_cents[None, :] <= k[:, None]
    return np.sum(weights * ((1.0 - q_exact) * residual + q_exact * exact), axis=1)


def cdf_components(k_cent: np.ndarray, weights: np.ndarray, m: Scenario):
    mu_lower, mu_upper, centers, shift = transformed_parameters(m)
    a = independent_cdf(
        k_cent, mu_lower, m.sd_lower, m.h1_lower, m.h5_lower
    )
    b = independent_cdf(
        k_cent, mu_upper, m.sd_upper, m.h1_upper, m.h5_upper
    )
    c = shared_signal_cdf(
        k_cent, weights, centers, m.sigma_shared, m.exact_focus_share
    )
    d = shared_signal_cdf(
        k_cent, weights, centers + shift, m.sigma_shared,
        m.exact_focus_share
    )
    return a, b, c, d


def draw_worlds(n_total: int, reps: int, m: Scenario, seed: int):
    """Draw latent-strategy counts, shared-signal weights, and bid means."""
    rng = np.random.default_rng(seed)
    n_other = n_total - 1
    probs = component_probs(m)
    counts = rng.multinomial(n_other, probs, size=reps).astype(np.int32)

    alpha0 = max(1e-6, 1.0 / m.rho_shared - 1.0)
    base_w = np.asarray(m.shared_weights, dtype=np.float64)
    weights = rng.dirichlet(alpha0 * base_w, size=reps)

    mu_lower, mu_upper, centers, shift = transformed_parameters(m)
    shared_mu = weights @ centers
    shared_var_between = (
        np.sum(weights * centers[None, :] ** 2, axis=1) - shared_mu ** 2
    )
    shared_var = (
        shared_var_between
        + (1.0 - m.exact_focus_share) * m.sigma_shared ** 2
    )

    means = np.column_stack((
        np.full(reps, mu_lower), np.full(reps, mu_upper),
        shared_mu, shared_mu + shift
    ))
    variances = np.column_stack((
        np.full(reps, m.sd_lower ** 2), np.full(reps, m.sd_upper ** 2),
        shared_var, shared_var
    ))
    expected_sum = np.sum(counts * means, axis=1)
    conditional_var_sum = np.sum(counts * variances, axis=1)
    bid_sum = expected_sum + rng.normal(size=reps) * np.sqrt(conditional_var_sum)
    mean_other = bid_sum / n_other
    return {"counts": counts, "weights": weights, "mean_other": mean_other}


def tie_factor_poisson(lam: np.ndarray) -> np.ndarray:
    out = np.ones_like(lam, dtype=np.float64)
    mask = lam > 1e-10
    out[mask] = -np.expm1(-lam[mask]) / lam[mask]
    return out


def score_candidates(worlds, n_total: int, m: Scenario, candidates,
                     value_ref: float = VALUE_REFERENCE):
    """Rao-Blackwellized nearest-mean win probabilities for fixed bids."""
    n_other = n_total - 1
    counts = worlds["counts"].astype(np.float64)
    weights = worlds["weights"]
    mean_other = worlds["mean_other"]
    reps = len(mean_other)
    rows = []
    for price in candidates:
        b_cent = int(round(price * 100.0))
        final_mean = (n_other * mean_other + price) / n_total
        mc = final_mean * 100.0
        dist = np.abs(b_cent - mc)
        lower = mc - dist
        upper = mc + dist

        # Strictly interior cent quotes beat the user's boundary quote.
        k_min = np.floor(lower + 1e-9).astype(np.int64) + 1
        k_max = np.ceil(upper - 1e-9).astype(np.int64) - 1
        valid = k_min <= k_max
        hi = cdf_components(k_max, weights, m)
        lo = cdf_components(k_min - 1, weights, m)
        q_in = [np.where(valid, np.clip(hi[j] - lo[j], 0.0, 1.0), 0.0)
                for j in range(4)]

        b_arr = np.full(reps, b_cent, dtype=np.int64)
        cb = cdf_components(b_arr, weights, m)
        cbm = cdf_components(b_arr - 1, weights, m)
        q_eq = [np.clip(cb[j] - cbm[j], 0.0, 1.0) for j in range(4)]

        log_no_closer = np.zeros(reps)
        lam_eq_cond = np.zeros(reps)
        for j in range(4):
            qi = np.minimum(q_in[j], 1.0 - 1e-15)
            log_no_closer += counts[:, j] * np.log1p(-qi)
            lam_eq_cond += counts[:, j] * q_eq[j] / np.maximum(1e-15, 1.0 - qi)
        p_no_closer = np.exp(np.maximum(log_no_closer, -745.0))
        win_weight = p_no_closer * tie_factor_poisson(lam_eq_cond)

        p_win = float(np.mean(win_weight))
        p_win_se = float(np.std(win_weight, ddof=1) / math.sqrt(reps))
        if p_win > 0:
            mean_if_win = float(np.sum(win_weight * final_mean) / np.sum(win_weight))
            cond_surplus = value_ref - mean_if_win
        else:
            mean_if_win = float("nan")
            cond_surplus = float("nan")
        utility_draw = win_weight * (value_ref - final_mean)
        uncond_utility = float(np.mean(utility_draw))
        utility_se = float(np.std(utility_draw, ddof=1) / math.sqrt(reps))
        rows.append({
            "bid": round(price, 2),
            "p_win": p_win,
            "p_win_pct": 100.0 * p_win,
            "p_win_pct_mc_se": 100.0 * p_win_se,
            "mean_if_win": mean_if_win,
            "conditional_surplus_v1500": cond_surplus,
            "unconditional_utility_v1500": uncond_utility,
            "utility_v1500_mc_se": utility_se,
        })
    best = max(r["unconditional_utility_v1500"] for r in rows)
    for r in rows:
        r["relative_utility"] = r["unconditional_utility_v1500"] / best if best else 0.0
    return rows


def run_once(n_total: int, reps: int, m: Scenario, candidates, seed: int):
    worlds = draw_worlds(n_total, reps, m, seed)
    rows = score_candidates(worlds, n_total, m, candidates)
    mean = worlds["mean_other"]
    return {
        "N": n_total,
        "reps": reps,
        "scenario": asdict(m),
        "component_probs": component_probs(m).tolist(),
        "mean_mean": float(np.mean(mean)),
        "mean_sd": float(np.std(mean, ddof=1)),
        "mean_q025": float(np.quantile(mean, 0.025)),
        "mean_q975": float(np.quantile(mean, 0.975)),
        "rows": rows,
    }


def best_row(result):
    return max(result["rows"], key=lambda x: x["unconditional_utility_v1500"])


def smoothed_choice(rows, window: int = 15):
    """Choose a stable cent bid by smoothing adjacent Monte Carlo utilities."""
    if window < 3 or window % 2 == 0 or window > len(rows):
        raise ValueError("Smoothing window must be an odd integer within the grid")
    bids = np.asarray([row["bid"] for row in rows], dtype=np.float64)
    utility = np.asarray(
        [row["unconditional_utility_v1500"] for row in rows],
        dtype=np.float64,
    )
    smooth = np.convolve(utility, np.ones(window) / window, mode="valid")
    half = window // 2
    grid_indices = np.arange(half, len(rows) - half)
    best_valid = int(np.argmax(smooth))
    best_index = int(grid_indices[best_valid])
    best_smooth = float(smooth[best_valid])

    def near_interval(fraction: float) -> list[float]:
        selected = bids[grid_indices[smooth >= fraction * best_smooth]]
        return [round(float(selected.min()), 2), round(float(selected.max()), 2)]

    return {
        "window_cents": window,
        "bid": round(float(bids[best_index]), 2),
        "smoothed_utility_v1500": best_smooth,
        "near_99pct": near_interval(0.99),
        "near_97_5pct": near_interval(0.975),
    }


def operational_choice(m: Scenario, rows, window: int = 15):
    """Prefer the analytical centre when it lies on the simulation plateau."""
    smooth = smoothed_choice(rows, window)
    analytical_bid = round(scenario_expected_mean(m), 2)
    lower, upper = smooth["near_99pct"]
    is_whole_yuan = math.isclose(analytical_bid, round(analytical_bid))
    if lower <= analytical_bid <= upper and not is_whole_yuan:
        selected = analytical_bid
        rule = "non-integer analytical centre within the 99% smoothed-utility plateau"
    else:
        eligible = [
            row for row in rows
            if lower <= row["bid"] <= upper
            and not math.isclose(row["bid"], round(row["bid"]))
        ]
        selected = max(
            eligible, key=lambda row: row["unconditional_utility_v1500"]
        )["bid"]
        rule = "best non-integer candidate within the 99% smoothed-utility plateau"
    return {"bid": selected, "rule": rule, "smoothed_grid": smooth}


def write_csv(path: Path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def direct_person_level_validation(n_total: int, reps: int, m: Scenario,
                                   candidates, seed: int, batch: int = 50):
    """Literal person-level generator used to validate the aggregate engine."""
    rng = np.random.default_rng(seed)
    probs = component_probs(m)
    cut = np.cumsum(probs)[:-1]
    mu_lower, mu_upper, centers, shift = transformed_parameters(m)
    alpha0 = max(1e-6, 1.0 / m.rho_shared - 1.0)
    base_w = np.asarray(m.shared_weights, dtype=np.float64)
    sums = {float(x): 0.0 for x in candidates}
    mean_sum = 0.0
    mean_sq = 0.0
    done = 0
    while done < reps:
        r = min(batch, reps - done)
        n = n_total - 1
        component = np.sum(
            rng.random((r, n))[:, :, None] > cut[None, None, :], axis=2
        )
        z = rng.normal(size=(r, n))
        quote = np.where(
            component == 0,
            mu_lower + m.sd_lower * z,
            np.where(component == 1, mu_upper + m.sd_upper * z, 0.0),
        )

        w = rng.dirichlet(alpha0 * base_w, size=r)
        u = rng.random((r, n))
        cum = np.cumsum(w, axis=1)
        idx = np.sum(u[:, :, None] > cum[:, None, :-1], axis=2)
        shared_base = centers[idx] + np.where(component == 3, shift, 0.0)
        exact = rng.random((r, n)) < m.exact_focus_share
        shared_quote = shared_base + np.where(
            exact, 0.0, m.sigma_shared * rng.normal(size=(r, n))
        )
        quote = np.where(component >= 2, shared_quote, quote)

        heap_u = rng.random((r, n))
        a_int = (component == 0) & (heap_u < m.h1_lower)
        a_five = ((component == 0) & (heap_u >= m.h1_lower)
                  & (heap_u < m.h1_lower + m.h5_lower))
        b_int = (component == 1) & (heap_u < m.h1_upper)
        b_five = ((component == 1) & (heap_u >= m.h1_upper)
                  & (heap_u < m.h1_upper + m.h5_upper))
        quote = np.where(a_int | b_int, np.rint(quote), quote)
        quote = np.where(a_five | b_five, 5.0 * np.rint(quote / 5.0), quote)
        quote = np.maximum(1.0, quote)
        cents = np.rint(quote * 100.0).astype(np.int32)
        row_sum = np.sum(cents, axis=1, dtype=np.int64)
        mean_other = row_sum / (100.0 * n)
        mean_sum += float(np.sum(mean_other))
        mean_sq += float(np.sum(mean_other ** 2))

        for price in candidates:
            bc = int(round(100.0 * price))
            final_cent = (row_sum + bc) / n_total
            diff = np.abs(cents - final_cent[:, None])
            min_diff = np.min(diff, axis=1)
            user_diff = np.abs(bc - final_cent)
            strictly = user_diff < min_diff - 1e-12
            equal = np.isclose(user_diff, min_diff, atol=1e-12, rtol=0.0)
            ties = np.sum(np.isclose(diff, user_diff[:, None], atol=1e-12,
                                     rtol=0.0), axis=1)
            share = strictly.astype(np.float64)
            share[equal] = 1.0 / (ties[equal] + 1.0)
            sums[float(price)] += float(np.sum(share))
        done += r
        print(f"direct validation {done}/{reps}", flush=True)
    mean = mean_sum / reps
    sd = math.sqrt(max(0.0, (mean_sq - reps * mean * mean) / (reps - 1)))
    return [{"N": n_total, "reps": reps, "bid": b,
             "p_win_pct": 100.0 * sums[b] / reps,
             "mean": mean, "mean_sd": sd} for b in sums]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true",
                        help="Use smaller replication counts for a fast check")
    parser.add_argument("--section", choices=["all", "main", "n", "sensitivity", "direct"],
                        default="all")
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help=("Directory for generated CSV/JSON files. Defaults to "
              "<repository>/data/results."),
    )
    parser.add_argument(
        "--participants", type=int, default=REFERENCE_N,
        help=("Number of bids for the target item. Defaults to the final "
              "public LOT 18 count."),
    )
    parser.add_argument(
        "--upper-strategy-share", type=float,
        help=("Override the illustrative upper-price strategy share. This is "
              "a scenario input, not an estimated audience demographic."),
    )
    parser.add_argument(
        "--shared-signal-share", type=float,
        help=("Override the illustrative shared-signal strategy share. This "
              "must not be interpreted as a measured AI-use rate."),
    )
    parser.add_argument(
        "--anchor-scale", type=float,
        help="Override the illustrative response to the public price anchor.",
    )
    args = parser.parse_args()
    if args.output_dir is None:
        out = Path(__file__).resolve().parents[1] / "data" / "results"
    else:
        out = args.output_dir.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    if args.participants < 2:
        parser.error("--participants must be at least 2")
    base = Scenario()
    if args.upper_strategy_share is not None:
        base = replace(base, upper_strategy_share=args.upper_strategy_share)
    if args.shared_signal_share is not None:
        base = replace(base, shared_signal_share=args.shared_signal_share)
    if args.anchor_scale is not None:
        base = replace(base, anchor_scale=args.anchor_scale)
    component_probs(base)  # validate the selected scenario before simulation

    r_coarse = 10_000 if args.quick else 30_000
    r_fine = 50_000 if args.quick else 180_000
    r_requested = 80_000 if args.quick else 600_000
    coarse_result = None
    main_result = None
    search_audit = None
    if args.section in ("all", "main"):
        coarse = coarse_localization_grid()
        print(
            f"broad coarse grid: {coarse[0]:.2f}-{coarse[-1]:.2f}, "
            f"step={COARSE_STEP:.2f}, R={r_coarse}", flush=True,
        )
        coarse_result = run_once(
            args.participants, r_coarse, base, coarse, 20260831
        )
        write_csv(out / "global_coarse_grid.csv", coarse_result["rows"])
        coarse_best = best_row(coarse_result)
        coarse_plateau = coarse_peak_plateau(coarse_result["rows"])
        fine_center = snapped_half_yuan_center(coarse_plateau["midpoint"])

        # A predeclared edge guardrail prevents a convenient but truncated
        # local window from being accepted.  Expand symmetrically when either
        # edge retains at least half of the interior peak utility.
        max_expansions = 3
        for expansions in range(max_expansions + 1):
            fine_half_width = FINE_HALF_WIDTH + expansions
            fine = local_fine_grid(fine_center, fine_half_width)
            print(
                f"fine grid: {fine[0]:.2f}-{fine[-1]:.2f}, "
                f"step={FINE_STEP:.2f}, R={r_fine}", flush=True,
            )
            main_result = run_once(
                args.participants, r_fine, base, fine, 20260901
            )
            boundary = fine_boundary_audit(main_result["rows"])
            if boundary["passed"]:
                break
        else:
            raise RuntimeError(
                "Fine-grid boundary audit failed after three expansions; "
                "review the declared broad search domain and scenario."
            )

        search_audit = {
            "declared_economic_domain": {
                "reference_value": VALUE_REFERENCE,
                "lower_rate": COARSE_LOWER_RATE,
                "upper_rate": COARSE_UPPER_RATE,
                "rationale": (
                    "broad localization domain around the public one-tenth "
                    "anchor; not an assumed feasible action set"
                ),
            },
            "coarse_grid": {
                "lower": coarse[0],
                "upper": coarse[-1],
                "step": COARSE_STEP,
                "phase_offset": COARSE_PHASE,
                "reps": r_coarse,
                "best_bid": coarse_best["bid"],
                "best_relative_utility": coarse_best["relative_utility"],
                "near_optimal_plateau": coarse_plateau,
                "left_boundary_relative_utility": (
                    coarse_result["rows"][0]["relative_utility"]
                ),
                "right_boundary_relative_utility": (
                    coarse_result["rows"][-1]["relative_utility"]
                ),
            },
            "fine_grid": {
                "center_rule": (
                    "nearest CNY 0.50 to the midpoint of the contiguous "
                    "90%-of-peak coarse plateau"
                ),
                "center": fine_center,
                "initial_half_width": FINE_HALF_WIDTH,
                "actual_half_width": fine_half_width,
                "lower": fine[0],
                "upper": fine[-1],
                "step": FINE_STEP,
                "reps": r_fine,
                "expansions": expansions,
                "boundary_audit": boundary,
            },
        }
        write_csv(out / "illustrative_fine_grid.csv", main_result["rows"])
        print(f"illustrative requested quotes: R={r_requested}", flush=True)
        requested_result = run_once(args.participants, r_requested, base,
                                    DEFAULT_CANDIDATES, 20260902)
        write_csv(out / "illustrative_requested_quotes.csv",
                  requested_result["rows"])

    # N scenarios, each with >=50,000 replications in the full run.
    n_reps = 15_000 if args.quick else 80_000
    n_candidates = np.round(np.arange(164.50, 164.951, 0.01), 2).tolist()
    n_rows = []
    if args.section in ("all", "n"):
        for idx, n in enumerate([10_000, REFERENCE_N, 14_000, 20_000, 25_000, 30_000, 50_000]):
            print(f"N scenario: N={n}, R={n_reps}", flush=True)
            rr = run_once(n, n_reps, base, n_candidates, 1100 + idx)
            br = best_row(rr)
            n_rows.append({
                "N": n, "reps": n_reps, "mean": rr["mean_mean"],
                "mean_sd": rr["mean_sd"], "best_bid": br["bid"],
                "p_win_pct": br["p_win_pct"],
                "utility_v1500": br["unconditional_utility_v1500"],
            })
        write_csv(out / "participants_sensitivity.csv", n_rows)

    # Explicit 10k/25k/50k convergence audit at N=25k.
    conv_rows = []
    if args.section in ("all", "n"):
        for idx, reps in enumerate([10_000, 25_000, 50_000]):
            print(f"convergence: R={reps}", flush=True)
            rr = run_once(REFERENCE_N, reps, base, DEFAULT_CANDIDATES, 2100 + idx)
            br = best_row(rr)
            conv_rows.append({
                "reps": reps, "mean": rr["mean_mean"], "mean_sd": rr["mean_sd"],
                "best_requested_bid": br["bid"], "p_win_pct": br["p_win_pct"],
            })
        write_csv(out / "replication_convergence.csv", conv_rows)

    # One-way sensitivity. Search a +/-1.25 range at 0.10, then refine cents.
    sens_reps = 8_000 if args.quick else 30_000
    sens_rows = []

    def one_scenario(group, value, model, seed):
        expected_mean = scenario_expected_mean(model)
        coarse = np.round(np.arange(max(80.0, expected_mean - 1.25),
                                    min(280.0, expected_mean + 1.251), 0.10), 2)
        worlds = draw_worlds(25_000, sens_reps, model, seed)
        rows0 = score_candidates(worlds, 25_000, model, coarse)
        b0 = max(rows0, key=lambda x: x["unconditional_utility_v1500"])["bid"]
        fine_local = np.round(np.arange(b0 - 0.10, b0 + 0.101, 0.01), 2)
        rows = score_candidates(worlds, 25_000, model, fine_local)
        br = max(rows, key=lambda x: x["unconditional_utility_v1500"])
        sens_rows.append({
            "parameter": group, "value": value,
            "population_mean": float(np.mean(worlds["mean_other"])),
            "mean_sd": float(np.std(worlds["mean_other"], ddof=1)),
            "optimal_bid": br["bid"], "p_win_pct": br["p_win_pct"],
            "relative_note": "one-way change; all other parameters illustrative",
        })

    if args.section in ("all", "sensitivity"):
        seed = 3000
        scenarios = []
        scenarios += [("upper_strategy_share", x,
                       replace(base, upper_strategy_share=x))
                      for x in [0.25, 0.30, 0.35, 0.40, 0.45]]
        scenarios += [("shared_signal_share", x,
                       replace(base, shared_signal_share=x))
                      for x in [0.10, 0.15, 0.20, 0.25, 0.30]]
        scenarios += [("rho_shared", label, replace(base, rho_shared=x))
                      for label, x in [("low", 0.02), ("medium", 0.10), ("high", 0.30)]]
        scenarios += [("anchor_response", label, replace(base, anchor_scale=x))
                      for label, x in [("low", 1.30), ("medium", 1.00), ("high", 0.65)]]
        scenarios += [("upper_independent_mean", x,
                       replace(base, mu_upper=x))
                      for x in [180.0, 188.0, 196.0, 204.0, 212.0]]
        scenarios += [("lower_independent_mean", x,
                       replace(base, mu_lower=x))
                      for x in [136.0, 142.0, 148.0, 154.0, 160.0]]
        scenarios += [("sigma_shared", x,
                       replace(base, sigma_shared=x))
                      for x in [2.0, 6.0, 12.0]]
        for group, value, model in scenarios:
            print(f"sensitivity: {group}={value}", flush=True)
            one_scenario(group, value, model, seed)
            seed += 1
        write_csv(out / "one_way_sensitivity.csv", sens_rows)

    if args.section == "direct":
        direct_rows = direct_person_level_validation(
            10_000, 10_000, base, [164.70, 164.74, 164.83, 165.00], 20260903)
        write_csv(out / "direct_person_level_validation.csv", direct_rows)
        print(direct_rows, flush=True)

    if main_result is None:
        illustrative_summary = None
    else:
        illustrative_summary = {
            "N": args.participants,
            "fine_grid_reps": r_fine,
            "requested_quote_reps": r_requested,
            "mean": main_result["mean_mean"],
            "mean_sd": main_result["mean_sd"],
            "mean_95_interval": [main_result["mean_q025"], main_result["mean_q975"]],
            "analytical_mean": scenario_expected_mean(base),
            "predictive_variance_decomposition": (
                predictive_variance_decomposition(args.participants, base)
            ),
            "search_audit": search_audit,
            "best_fine": best_row(main_result),
            "smoothed_choice": smoothed_choice(main_result["rows"]),
            "operational_choice": operational_choice(base, main_result["rows"]),
            "component_probs": main_result["component_probs"],
        }
    summary_path = out / "simulation_summary.json"
    existing_summary = {}
    if summary_path.exists():
        try:
            existing_summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            existing_summary = {}
    summary = {
        "illustrative_scenario": (
            illustrative_summary
            if illustrative_summary is not None
            else existing_summary.get("illustrative_scenario")
        ),
        "participants": (
            n_rows
            if args.section in ("all", "n")
            else existing_summary.get("participants", [])
        ),
        "convergence": (
            conv_rows
            if args.section in ("all", "n")
            else existing_summary.get("convergence", [])
        ),
        "sensitivity": (
            sens_rows
            if args.section in ("all", "sensitivity")
            else existing_summary.get("sensitivity", [])
        ),
        "method_note": (
            "Monte Carlo samples latent strategy counts, shared-signal Dirichlet "
            "weights, and aggregate bid sums. Component shares are scenario "
            "inputs, not measured audience demographics. Conditional closer-bid "
            "probabilities use the discrete bid CDF; tie multiplicity uses a "
            "Poisson approximation. A phase-shifted broad grid localizes the "
            "contiguous coarse near-optimal plateau before the cent grid is "
            "constructed and boundary-audited."
        ),
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(summary["illustrative_scenario"], ensure_ascii=False,
                     indent=2), flush=True)
    print("N scenarios:")
    for row in n_rows:
        print(row)
    print("Convergence:")
    for row in conv_rows:
        print(row)


if __name__ == "__main__":
    main()
