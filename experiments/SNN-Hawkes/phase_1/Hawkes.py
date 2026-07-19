"""
Multivariate Hawkes process with exponential kernels.

Kernel convention (branching-ratio parametrisation):
    phi_ij(t) = A_ij * beta * exp(-beta * t),   t > 0
    integral_0^inf phi_ij = A_ij           (A_ij = expected # of type-i
                                            offspring triggered by one type-j event)

Conditional intensity for type i:
    lambda_i(t) = mu_i + sum_j A_ij * beta * sum_{t_k^j < t} exp(-beta (t - t_k^j))

Stability: the process is stationary iff spectralRadius(A) < 1.
"""
import numpy as np


def spectral_radius(A):
    return np.max(np.abs(np.linalg.eigvals(A)))


def simulate_hawkes(mu, A, beta, T, seed=0):
    """Ogata thinning simulation of a multivariate exp-kernel Hawkes process.

    Returns
    -------
    times : (N,) float array, sorted event times
    types : (N,) int array, event type in [0, D)
    """
    rng = np.random.default_rng(seed)
    mu = np.asarray(mu, float)
    A = np.asarray(A, float)
    D = len(mu)

    times, types = [], []
    t = 0.0
    # g[j] = sum_{t_k^j < t} exp(-beta (t - t_k^j)); decayed as we advance t
    g = np.zeros(D)
    last_t = 0.0

    while t < T:
        # current intensities using history strictly before t
        g_now = g * np.exp(-beta * (t - last_t))
        lam = mu + A @ (beta * g_now)          # (D,)
        lam_sum = lam.sum()
        # upper bound: intensity only jumps up at events, decays between them,
        # so lam_sum at current t is a valid local upper bound
        w = rng.exponential(1.0 / lam_sum)
        t = t + w
        if t >= T:
            break
        # recompute intensity at candidate time t
        g_cand = g * np.exp(-beta * (t - last_t))
        lam_cand = mu + A @ (beta * g_cand)
        lam_cand_sum = lam_cand.sum()
        # accept with prob lam_cand_sum / lam_sum (thinning)
        if rng.uniform() <= lam_cand_sum / lam_sum:
            # choose type
            c = rng.choice(D, p=lam_cand / lam_cand_sum)
            times.append(t)
            types.append(c)
            # update trace to reflect events up to and including t
            g = g_cand.copy()
            g[c] += 1.0
            last_t = t
        else:
            # rejected: still advance the trace bookkeeping to t
            g = g_cand
            last_t = t

    return np.asarray(times), np.asarray(types, dtype=int)


def true_intensity_on_grid(times, types, mu, A, beta, grid):
    """Left-continuous intensity lambda_i(t) on `grid`, using events strictly
    before each grid point. Returns (len(grid), D)."""
    mu = np.asarray(mu, float)
    A = np.asarray(A, float)
    D = len(mu)
    out = np.zeros((len(grid), D))
    g = np.zeros(D)
    last_t = grid[0]
    ev_ptr = 0
    N = len(times)
    for gi, tg in enumerate(grid):
        # decay trace to tg
        g = g * np.exp(-beta * (tg - last_t))
        last_t = tg
        # add events in [prev grid, tg) -> events strictly before tg
        while ev_ptr < N and times[ev_ptr] < tg:
            g[types[ev_ptr]] += np.exp(-beta * (tg - times[ev_ptr]))
            ev_ptr += 1
        out[gi] = mu + A @ (beta * g)
    return out


def _nll_and_compensator(params_flat, times, types, beta, T, D):
    """Continuous-time negative log-likelihood of exp-kernel Hawkes.
    params_flat = [mu (D), A (D*D)] , all >= 0 enforced by caller bounds."""
    mu = params_flat[:D]
    A = params_flat[D:].reshape(D, D)
    N = len(times)
    # log-lambda term via O(N) recursion
    g = np.zeros(D)           # decaying per-type kernel sum (exp only, no beta)
    last_t = 0.0
    loglik = 0.0
    for k in range(N):
        tk = times[k]
        g = g * np.exp(-beta * (tk - last_t))
        last_t = tk
        lam = mu + A @ (beta * g)      # intensity at tk^- (strictly before)
        ck = types[k]
        loglik += np.log(max(lam[ck], 1e-12))
        g[ck] += 1.0                   # include current event for future
    # compensator: integral_0^T lambda_i ds  summed over i
    # mu_i T + sum_j A_ij * sum_{t_l^j} (1 - exp(-beta (T - t_l^j)))
    comp = mu.sum() * T
    decay_term = 1.0 - np.exp(-beta * (T - times))   # (N,)
    # per-type sums of decay_term
    S = np.zeros(D)
    np.add.at(S, types, decay_term)
    comp += (A @ S).sum()              # sum_i sum_j A_ij S_j
    loglik -= comp
    return -loglik


def fit_hawkes_mle(times, types, beta, T, D, x0=None, verbose=False):
    """Fit mu, A by maximum likelihood (beta assumed known). Returns mu, A."""
    from scipy.optimize import minimize
    if x0 is None:
        x0 = np.concatenate([np.full(D, 0.1), np.full(D * D, 0.05)])
    bounds = [(1e-6, None)] * D + [(0.0, None)] * (D * D)
    res = minimize(_nll_and_compensator, x0, args=(times, types, beta, T, D),
                   method="L-BFGS-B", bounds=bounds,
                   options={"maxiter": 500, "ftol": 1e-10})
    if verbose:
        print("  MLE success:", res.success, "nll=%.2f" % res.fun)
    mu = res.x[:D]
    A = res.x[D:].reshape(D, D)
    return mu, A, res.fun


def binned_poisson_nll(intensity_grid, counts, dt):
    """Common evaluation metric: binned Poisson NLL.
    intensity_grid: (T,D) predicted lambda at left edge of each bin
    counts:         (T,D) integer event counts per bin
    Returns total NLL (dropping the constant log(n!) which is ~0 for n<=1)."""
    rate = np.clip(intensity_grid * dt, 1e-12, None)
    nll = rate.sum() - (counts * np.log(rate)).sum()
    return nll