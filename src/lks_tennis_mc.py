"""Monte Carlo model for the LKs nearest-mean tennis-racket contest.

The simulation is deliberately reproducible (fixed RNG seeds).  It samples
type counts, a shared AI-answer environment, and the aggregate distribution of
all other bids.  Conditional nearest-price occupancy is Rao-Blackwellized:
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
class Model:
    fan_share: float = 0.35
    ai_share: float = 0.20
    ai_fan_ratio: float = 1.30  # P(AI|fan) / P(AI)
    rho_ai: float = 0.10        # ICC of latent AI answer-focus draws
    sigma_ai: float = 6.0       # idiosyncratic noise around an AI focus
    exact_focus_share: float = 0.35
    mu_a: float = 148.0         # non-fan, non-AI quote centre
    sd_a: float = 48.0
    mu_b: float = 196.0         # fan, non-AI quote centre
    sd_b: float = 58.0
    ai_centers: tuple[float, ...] = (155.0, 160.0, 165.0, 170.0, 175.0)
    ai_weights: tuple[float, ...] = (0.10, 0.20, 0.40, 0.20, 0.10)
    ai_fan_shift: float = 10.0
    anchor_scale: float = 1.0   # deviations from the public 150-CNY anchor
    h1_a: float = 0.12          # integer-yuan heaping
    h5_a: float = 0.08          # 5-yuan heaping
    h1_b: float = 0.10
    h5_b: float = 0.10


DEFAULT_CANDIDATES = [
    160.00, 162.00, 163.00, 164.00, 164.50, 164.70, 164.75,
    164.80, 164.81, 164.82, 164.83, 164.84, 164.85, 164.86,
    164.90, 165.00, 166.00, 168.00, 170.00,
]


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


def joint_type_probs(m: Model) -> np.ndarray:
    """A, B, C, D = nonfan/nonAI, fan/nonAI, nonfan/AI, fan/AI."""
    d = m.fan_share * min(0.95, m.ai_share * m.ai_fan_ratio)
    c = m.ai_share - d
    b = m.fan_share - d
    a = 1.0 - b - c - d
    if min(a, b, c, d) < 0:
        raise ValueError("Infeasible fan/AI joint proportions")
    return np.array([a, b, c, d], dtype=np.float64)


def transformed_parameters(m: Model):
    k = m.anchor_scale
    anchor = 150.0
    mu_a = anchor + k * (m.mu_a - anchor)
    mu_b = anchor + k * (m.mu_b - anchor)
    centers = anchor + k * (np.asarray(m.ai_centers) - anchor)
    fan_shift = k * m.ai_fan_shift
    return mu_a, mu_b, centers, fan_shift


def nonai_cdf(k_cent: np.ndarray, mu: float, sd: float,
              h1: float, h5: float) -> np.ndarray:
    """CDF of a normal strategy draw followed by cent/integer/5-yuan heaping."""
    k = np.asarray(k_cent, dtype=np.int64)
    p_cent = norm_cdf((((k + 0.5) / 100.0) - mu) / sd)
    max_integer_yuan = np.floor_divide(k, 100).astype(np.float64)
    p_int = norm_cdf((max_integer_yuan + 0.5 - mu) / sd)
    max_five_yuan = 5.0 * np.floor_divide(k, 500).astype(np.float64)
    p_five = norm_cdf((max_five_yuan + 2.5 - mu) / sd)
    return (1.0 - h1 - h5) * p_cent + h1 * p_int + h5 * p_five


def ai_cdf(k_cent: np.ndarray, weights: np.ndarray, centers: np.ndarray,
           sigma: float, q_exact: float) -> np.ndarray:
    """Conditional CDF of the rounded AI-focus mixture."""
    k = np.asarray(k_cent, dtype=np.int64)
    threshold = (k.astype(np.float64) + 0.5) / 100.0
    residual = norm_cdf((threshold[:, None] - centers[None, :]) / sigma)
    center_cents = np.rint(centers * 100.0).astype(np.int64)
    exact = center_cents[None, :] <= k[:, None]
    return np.sum(weights * ((1.0 - q_exact) * residual + q_exact * exact), axis=1)


def cdf_types(k_cent: np.ndarray, weights: np.ndarray, m: Model):
    mu_a, mu_b, centers, shift = transformed_parameters(m)
    a = nonai_cdf(k_cent, mu_a, m.sd_a, m.h1_a, m.h5_a)
    b = nonai_cdf(k_cent, mu_b, m.sd_b, m.h1_b, m.h5_b)
    c = ai_cdf(k_cent, weights, centers, m.sigma_ai, m.exact_focus_share)
    d = ai_cdf(k_cent, weights, centers + shift, m.sigma_ai,
               m.exact_focus_share)
    return a, b, c, d


def draw_worlds(n_total: int, reps: int, m: Model, seed: int):
    """Draw type counts, shared AI cluster weights, and the other-bid mean."""
    rng = np.random.default_rng(seed)
    n_other = n_total - 1
    probs = joint_type_probs(m)
    counts = rng.multinomial(n_other, probs, size=reps).astype(np.int32)

    alpha0 = max(1e-6, 1.0 / m.rho_ai - 1.0)
    base_w = np.asarray(m.ai_weights, dtype=np.float64)
    weights = rng.dirichlet(alpha0 * base_w, size=reps)

    mu_a, mu_b, centers, shift = transformed_parameters(m)
    ai_mu = weights @ centers
    ai_var_between = np.sum(weights * centers[None, :] ** 2, axis=1) - ai_mu ** 2
    ai_var = ai_var_between + (1.0 - m.exact_focus_share) * m.sigma_ai ** 2

    means = np.column_stack((
        np.full(reps, mu_a), np.full(reps, mu_b), ai_mu, ai_mu + shift
    ))
    variances = np.column_stack((
        np.full(reps, m.sd_a ** 2), np.full(reps, m.sd_b ** 2),
        ai_var, ai_var
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


def score_candidates(worlds, n_total: int, m: Model, candidates,
                     value_ref: float = 1500.0):
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
        hi = cdf_types(k_max, weights, m)
        lo = cdf_types(k_min - 1, weights, m)
        q_in = [np.where(valid, np.clip(hi[j] - lo[j], 0.0, 1.0), 0.0)
                for j in range(4)]

        b_arr = np.full(reps, b_cent, dtype=np.int64)
        cb = cdf_types(b_arr, weights, m)
        cbm = cdf_types(b_arr - 1, weights, m)
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
        if p_win > 0:
            mean_if_win = float(np.sum(win_weight * final_mean) / np.sum(win_weight))
            cond_surplus = value_ref - mean_if_win
        else:
            mean_if_win = float("nan")
            cond_surplus = float("nan")
        uncond_utility = float(np.mean(win_weight * (value_ref - final_mean)))
        rows.append({
            "bid": round(price, 2),
            "p_win": p_win,
            "p_win_pct": 100.0 * p_win,
            "mean_if_win": mean_if_win,
            "conditional_surplus_v1500": cond_surplus,
            "unconditional_utility_v1500": uncond_utility,
        })
    best = max(r["unconditional_utility_v1500"] for r in rows)
    for r in rows:
        r["relative_utility"] = r["unconditional_utility_v1500"] / best if best else 0.0
    return rows


def run_once(n_total: int, reps: int, m: Model, candidates, seed: int):
    worlds = draw_worlds(n_total, reps, m, seed)
    rows = score_candidates(worlds, n_total, m, candidates)
    mean = worlds["mean_other"]
    return {
        "N": n_total,
        "reps": reps,
        "model": asdict(m),
        "type_probs": joint_type_probs(m).tolist(),
        "mean_mean": float(np.mean(mean)),
        "mean_sd": float(np.std(mean, ddof=1)),
        "mean_q025": float(np.quantile(mean, 0.025)),
        "mean_q975": float(np.quantile(mean, 0.975)),
        "rows": rows,
    }


def best_row(result):
    return max(result["rows"], key=lambda x: x["unconditional_utility_v1500"])


def write_csv(path: Path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def direct_person_level_validation(n_total: int, reps: int, m: Model,
                                   candidates, seed: int, batch: int = 50):
    """Literal person-level generator used to validate the aggregate engine."""
    rng = np.random.default_rng(seed)
    probs = joint_type_probs(m)
    cut = np.cumsum(probs)[:-1]
    mu_a, mu_b, centers, shift = transformed_parameters(m)
    alpha0 = max(1e-6, 1.0 / m.rho_ai - 1.0)
    base_w = np.asarray(m.ai_weights, dtype=np.float64)
    sums = {float(x): 0.0 for x in candidates}
    mean_sum = 0.0
    mean_sq = 0.0
    done = 0
    while done < reps:
        r = min(batch, reps - done)
        n = n_total - 1
        typ = np.sum(rng.random((r, n))[:, :, None] > cut[None, None, :], axis=2)
        z = rng.normal(size=(r, n))
        quote = np.where(typ == 0, mu_a + m.sd_a * z,
                         np.where(typ == 1, mu_b + m.sd_b * z, 0.0))

        w = rng.dirichlet(alpha0 * base_w, size=r)
        u = rng.random((r, n))
        cum = np.cumsum(w, axis=1)
        idx = np.sum(u[:, :, None] > cum[:, None, :-1], axis=2)
        ai_base = centers[idx] + np.where(typ == 3, shift, 0.0)
        exact = rng.random((r, n)) < m.exact_focus_share
        ai_quote = ai_base + np.where(exact, 0.0, m.sigma_ai * rng.normal(size=(r, n)))
        quote = np.where(typ >= 2, ai_quote, quote)

        heap_u = rng.random((r, n))
        a_int = (typ == 0) & (heap_u < m.h1_a)
        a_five = (typ == 0) & (heap_u >= m.h1_a) & (heap_u < m.h1_a + m.h5_a)
        b_int = (typ == 1) & (heap_u < m.h1_b)
        b_five = (typ == 1) & (heap_u >= m.h1_b) & (heap_u < m.h1_b + m.h5_b)
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
    args = parser.parse_args()
    if args.output_dir is None:
        out = Path(__file__).resolve().parents[1] / "data" / "results"
    else:
        out = args.output_dir.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    base = Model()

    # Fine search around the previously proposed 164.83, plus the requested list.
    fine = np.round(np.arange(163.50, 165.501, 0.01), 2).tolist()
    all_candidates = sorted(set(DEFAULT_CANDIDATES + fine))
    r_fine = 50_000 if args.quick else 180_000
    r_requested = 80_000 if args.quick else 600_000
    main_result = None
    if args.section in ("all", "main"):
        print(f"baseline fine grid: R={r_fine}", flush=True)
        main_result = run_once(25_000, r_fine, base, fine, 20260901)
        write_csv(out / "baseline_fine_grid.csv", main_result["rows"])
        print(f"baseline requested quotes: R={r_requested}", flush=True)
        requested_result = run_once(25_000, r_requested, base,
                                    DEFAULT_CANDIDATES, 20260902)
        write_csv(out / "baseline_requested_quotes.csv", requested_result["rows"])

    # N scenarios, each with >=50,000 replications in the full run.
    n_reps = 15_000 if args.quick else 80_000
    n_candidates = np.round(np.arange(164.50, 164.951, 0.01), 2).tolist()
    n_rows = []
    if args.section in ("all", "n"):
        for idx, n in enumerate([10_000, 12_500, 14_000, 20_000, 25_000, 30_000, 50_000]):
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
            rr = run_once(25_000, reps, base, DEFAULT_CANDIDATES, 2100 + idx)
            br = best_row(rr)
            conv_rows.append({
                "reps": reps, "mean": rr["mean_mean"], "mean_sd": rr["mean_sd"],
                "best_requested_bid": br["bid"], "p_win_pct": br["p_win_pct"],
            })
        write_csv(out / "replication_convergence.csv", conv_rows)

    # One-way sensitivity.  Search a wider 0.05 grid, then refine +/-0.10 at cents.
    sens_reps = 8_000 if args.quick else 30_000
    sens_rows = []

    def one_scenario(group, value, model, seed):
        mu_a, mu_b, centers, shift = transformed_parameters(model)
        p = joint_type_probs(model)
        expected_mean = p @ np.array([mu_a, mu_b,
                                      np.dot(model.ai_weights, centers),
                                      np.dot(model.ai_weights, centers) + shift])
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
            "relative_note": "one-way change; all other parameters baseline",
        })

    if args.section in ("all", "sensitivity"):
        seed = 3000
        scenarios = []
        scenarios += [("fan_share", x, replace(base, fan_share=x))
                      for x in [0.25, 0.30, 0.35, 0.40, 0.45]]
        scenarios += [("ai_share", x, replace(base, ai_share=x))
                      for x in [0.10, 0.15, 0.20, 0.25, 0.30]]
        scenarios += [("rho_ai", label, replace(base, rho_ai=x))
                      for label, x in [("low", 0.02), ("medium", 0.10), ("high", 0.30)]]
        scenarios += [("one_tenth_obedience", label, replace(base, anchor_scale=x))
                      for label, x in [("low", 1.30), ("medium", 1.00), ("high", 0.65)]]
        scenarios += [("fan_nonAI_mean", x, replace(base, mu_b=x))
                      for x in [180.0, 188.0, 196.0, 204.0, 212.0]]
        scenarios += [("nonfan_nonAI_mean", x, replace(base, mu_a=x))
                      for x in [136.0, 142.0, 148.0, 154.0, 160.0]]
        scenarios += [("sigma_ai", x, replace(base, sigma_ai=x))
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
        baseline_summary = None
    else:
        baseline_summary = {
            "N": 25_000,
            "fine_grid_reps": r_fine,
            "requested_quote_reps": r_requested,
            "mean": main_result["mean_mean"],
            "mean_sd": main_result["mean_sd"],
            "mean_95_interval": [main_result["mean_q025"], main_result["mean_q975"]],
            "best_fine": best_row(main_result),
            "type_probs": main_result["type_probs"],
        }
    summary = {
        "baseline": {
            **(baseline_summary or {})
        } if baseline_summary else None,
        "participants": n_rows,
        "convergence": conv_rows,
        "sensitivity": sens_rows,
        "method_note": (
            "Monte Carlo samples type counts, shared Dirichlet AI focus weights, "
            "and aggregate bid sums. Conditional closer-bid probabilities use the "
            "discrete bid CDF; tie multiplicity uses a Poisson approximation."
        ),
    }
    (out / "simulation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(summary["baseline"], ensure_ascii=False, indent=2), flush=True)
    print("N scenarios:")
    for row in n_rows:
        print(row)
    print("Convergence:")
    for row in conv_rows:
        print(row)


if __name__ == "__main__":
    main()
