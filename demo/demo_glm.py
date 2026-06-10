"""Demo: CMP GLM adaptive filtering for simulated neuronal spike data.

This script demonstrates the full pipeline:
1. Generate synthetic spike data from a CMP distribution with a
   drifting tuning curve (figure1_singleNu_dirShift analogue).
2. Fit a CMP GLM using adaptive filtering (ppasmoo_compoisson_fisher_na).
3. Optimise the process-noise covariance Q with scipy.optimize.
4. Refine the fit with Newton-Raphson (newtonGH).
5. Plot mean firing rate and Fano factor.

Usage
-----
    python demo/demo_glm.py

Dependencies
------------
    pip install com_poisson matplotlib
"""

from __future__ import annotations

import warnings

import numpy as np
import scipy.optimize as opt
from scipy.special import gammaln

from com_poisson.basis import getCubicBSplineBasis
from com_poisson.distribution import CMPmoment, com_rnd
from com_poisson.optimization import (
    gradHessTheta_na,
    helper_na,
    newtonGH,
)
from com_poisson.smoother import ppasmoo_compoisson_fisher_na

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def main() -> None:
    """Run the CMP GLM demo."""
    rng = np.random.default_rng(6)

    # ------------------------------------------------------------------ #
    # 1. Simulation setup                                                 #
    # ------------------------------------------------------------------ #
    nknots = 10
    x0 = np.linspace(0, 1, 100)
    basX = getCubicBSplineBasis(x0, nknots, isCirc=False)  # (100, nknots+1)

    T = 100           # number of time steps (trials)
    kStep = T         # same as T here (dt=1)
    angle = 180 * (x0 - x0.min()) / np.ptp(x0)  # orientation in degrees

    # Dispersion varies linearly: nu = exp(gamma)
    gam = np.tile(np.linspace(-1, 1, kStep), (len(x0), 1))  # (100, 100)
    nu = np.exp(gam)
    nuSing = nu[0, :]   # single nu per trial

    # Build the true log-lambda coefficients
    n_basis = basX.shape[1]
    beta = np.zeros((n_basis, kStep))

    basMean = 2.0
    logLamBas = nuSing * np.log(basMean + (nuSing - 1) / (2 * nuSing))

    targetMean = np.linspace(5, 10, kStep) - basMean
    logLam = nuSing * np.log(targetMean + (nuSing - 1) / (2 * nuSing))
    weightSum = logLam / np.max(basX[:, 1:])

    weightKnots = 3
    weightBas = getCubicBSplineBasis(np.linspace(0, 1, kStep), weightKnots, isCirc=False)
    weight_cols = weightBas[:, 1:]
    weight = weight_cols / weight_cols.sum(axis=1, keepdims=True)

    beta[0, :] = logLamBas
    beta[5, :] = weight[:, 0] * weightSum
    beta[6, :] = weight[:, 1] * weightSum
    beta[7, :] = weight[:, 2] * weightSum

    # True CMP means
    lam_true = np.exp(basX @ beta)  # (100, 100)

    print("Computing true CMP moments…")
    CMP_mean = np.zeros_like(lam_true)
    CMP_var = np.zeros_like(lam_true)
    for m in range(lam_true.shape[0]):
        for n in range(lam_true.shape[1]):
            CMP_mean[m, n], CMP_var[m, n], *_ = CMPmoment(lam_true[m, n], nu[m, n])

    # ------------------------------------------------------------------ #
    # 2. Simulate spike counts                                            #
    # ------------------------------------------------------------------ #
    print("Simulating spike counts…")
    spk = np.zeros_like(lam_true)
    for m in range(lam_true.shape[0]):
        for n in range(lam_true.shape[1]):
            # Replace rng usage to use global numpy rng via seed
            spk[m, n] = com_rnd(lam_true[m, n], nu[m, n], 1)[0]

    # ------------------------------------------------------------------ #
    # 3. Prepare model matrices                                           #
    # ------------------------------------------------------------------ #
    Tall = len(x0) * kStep
    basX_trans = np.tile(basX, (kStep, 1))  # (Tall, n_basis)
    G_nu = np.ones((Tall, 1))               # single nu coefficient
    spk_vec = spk.ravel(order="F")          # (Tall,)

    # Initial parameters via simple Poisson GLM (log-link)
    from scipy.optimize import minimize

    def neg_poisson_llhd(beta_flat):
        lam = np.exp(basX_trans[:len(x0), :] @ beta_flat[:n_basis])
        lam = np.clip(lam, 1e-10, None)
        return -np.sum(spk_vec[:len(x0)] * np.log(lam) - lam)

    beta0_init = np.zeros(n_basis + 1)
    res = minimize(neg_poisson_llhd, beta0_init[:n_basis], method="L-BFGS-B")
    theta0 = np.concatenate([res.x, [0.0]])  # append log(nu)=0 => nu=1

    print(f"theta0 shape: {theta0.shape}")

    # ------------------------------------------------------------------ #
    # 4. Initial filter run                                               #
    # ------------------------------------------------------------------ #
    p = len(theta0)
    Q_init = 1e-4 * np.eye(p)
    W0 = np.eye(p)
    F = np.eye(p)

    print("Running initial CMP smoother…")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        theta_fit_tmp, W_fit_tmp, *_ = ppasmoo_compoisson_fisher_na(
            theta0, spk_vec[np.newaxis, :], basX_trans, G_nu, W0, F, Q_init
        )

    theta01 = theta_fit_tmp[:, 0]
    W01 = W_fit_tmp[:, :, 0]

    # ------------------------------------------------------------------ #
    # 5. Optimise process-noise covariance Q                             #
    # ------------------------------------------------------------------ #
    print("Optimising Q…")
    n_lam = basX_trans.shape[1]
    n_nu = G_nu.shape[1]
    n_Q = min(2, n_lam) + min(2, n_nu)

    Q_lb = 1e-8
    Q_ub = 1e-3
    Q0_vec = 1e-4 * np.ones(n_Q)
    bounds = [(Q_lb, Q_ub)] * n_Q

    def obj_Q(q):
        return helper_na(q, theta01, spk_vec[np.newaxis, :], basX_trans, G_nu, W01, F)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = opt.minimize(
            obj_Q,
            Q0_vec,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 100, "ftol": 1e-9},
        )
    Qopt = result.x
    print(f"Optimal Q params: {Qopt}")

    # Build Q matrix
    Q_lam = np.concatenate([[Qopt[0]], Qopt[1] * np.ones(n_lam - 1)])
    Q_nu = np.array([Qopt[2]])
    Qoptmatrix = np.diag(np.concatenate([Q_lam, Q_nu]))

    # ------------------------------------------------------------------ #
    # 6. Newton-Raphson refinement                                        #
    # ------------------------------------------------------------------ #
    print("Running Newton-Raphson refinement…")

    def gradHess_fn(vecTheta):
        return gradHessTheta_na(
            vecTheta, basX_trans, G_nu,
            theta01, W01, F, Qoptmatrix, spk_vec
        )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            theta_newton_vec, _, _ = newtonGH(
                gradHess_fn, theta_fit_tmp.ravel(order="F"), TolX=1e-6, MaxIter=100
            )
            theta_fit = theta_newton_vec.reshape(p, Tall, order="F")
        except (RuntimeError, ValueError) as exc:
            print(f"Newton-Raphson warning: {exc}. Using smoother estimate.")
            theta_fit = theta_fit_tmp

    # ------------------------------------------------------------------ #
    # 7. Extract fitted rates                                             #
    # ------------------------------------------------------------------ #
    print("Extracting fitted CMP moments…")
    lam_fit = np.zeros(Tall)
    nu_fit = np.zeros(Tall)
    CMP_mean_fit = np.zeros(Tall)
    CMP_var_fit = np.zeros(Tall)

    for m in range(Tall):
        lam_fit[m] = np.exp(basX_trans[m, :] @ theta_fit[:n_basis, m])
        nu_fit[m] = np.exp(float(G_nu[m, :] @ theta_fit[n_basis:, m]))
        CMP_mean_fit[m], CMP_var_fit[m], *_ = CMPmoment(lam_fit[m], nu_fit[m])

    CMP_ff_fit = CMP_var_fit / np.maximum(CMP_mean_fit, 1e-10)
    CMP_mean_fit_mat = CMP_mean_fit.reshape(len(x0), kStep, order="F")
    CMP_ff_fit_mat = CMP_ff_fit.reshape(len(x0), kStep, order="F")

    # ------------------------------------------------------------------ #
    # 8. Plot results                                                     #
    # ------------------------------------------------------------------ #
    if HAS_MATPLOTLIB:
        t1 = int(np.round(kStep / 5))
        t2 = int(np.round(kStep * 4 / 5))

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle("CMP GLM Adaptive Filtering Demo", fontsize=14)

        # True mean firing rate
        im0 = axes[0, 0].imshow(
            CMP_mean, aspect="auto", origin="lower",
            extent=[1, T, angle[0], angle[-1]], cmap="gray_r"
        )
        axes[0, 0].set_title("True Mean Firing Rate")
        axes[0, 0].set_xlabel("Trial")
        axes[0, 0].set_ylabel("Orientation (°)")
        plt.colorbar(im0, ax=axes[0, 0])

        # True Fano factor
        im1 = axes[0, 1].imshow(
            CMP_var / np.maximum(CMP_mean, 1e-10), aspect="auto",
            origin="lower", extent=[1, T, angle[0], angle[-1]], cmap="gray_r"
        )
        axes[0, 1].set_title("True Fano Factor")
        axes[0, 1].set_xlabel("Trial")
        axes[0, 1].set_ylabel("Orientation (°)")
        plt.colorbar(im1, ax=axes[0, 1])

        # Fitted mean firing rate
        im2 = axes[1, 0].imshow(
            CMP_mean_fit_mat, aspect="auto", origin="lower",
            extent=[1, T, angle[0], angle[-1]], cmap="gray_r"
        )
        axes[1, 0].set_title("Fitted Mean Firing Rate")
        axes[1, 0].set_xlabel("Trial")
        axes[1, 0].set_ylabel("Orientation (°)")
        plt.colorbar(im2, ax=axes[1, 0])

        # Tuning curves at two time points
        idt1 = int(np.argmax(CMP_mean[:, t1]))
        idt2 = int(np.argmax(CMP_mean[:, t2]))
        axes[1, 1].plot(angle, CMP_mean[:, t1], "b-", lw=2, label=f"True t={t1}")
        axes[1, 1].plot(angle, CMP_mean[:, t2], "r-", lw=2, label=f"True t={t2}")
        axes[1, 1].plot(angle, CMP_mean_fit_mat[:, t1], "b--", lw=2, label=f"Fit t={t1}")
        axes[1, 1].plot(angle, CMP_mean_fit_mat[:, t2], "r--", lw=2, label=f"Fit t={t2}")
        axes[1, 1].set_title("Tuning Curves")
        axes[1, 1].set_xlabel("Orientation (°)")
        axes[1, 1].set_ylabel("Mean Firing Rate")
        axes[1, 1].legend(fontsize=8)

        plt.tight_layout()
        plt.savefig("demo_glm_results.png", dpi=150, bbox_inches="tight")
        print("Saved demo_glm_results.png")
        plt.show()
    else:
        print("matplotlib not available – skipping plots.")
        print(f"Max fitted mean firing rate: {CMP_mean_fit_mat.max():.3f}")
        print(f"Max true mean firing rate:   {CMP_mean.max():.3f}")

    print("Done.")


if __name__ == "__main__":
    main()
