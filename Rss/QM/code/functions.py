import sympy as sp
import string
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm
from scipy.special import airy, jv, jvp
import subprocess
import os
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from scipy.optimize import brentq

plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "pgf.texsystem": "pdflatex",
        "pgf.rcfonts": False,
        # "font.size": 20,
        # 
        "font.size": 10,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "lines.linewidth": 0.8,
    }
)

# -------------
# Misc function
# -------------

def done():
  subprocess.run(["notify-send", "-u", "low","-t", "1000", "Done", "Cell finished"])
  subprocess.run(["espeak","-a", "30","done"])

# -----------
# Hamiltonian
# -----------

def HxLong(N, C, a, nu, B,alp):
    H = np.zeros((N, N), dtype=float)

    def J(i, j):
        if i == j:
            return 0.0

        r = float(abs(i - j)) * float(a)
        return C / r**nu

    for i in range(N):
        for j in range(N):
            if i != j:
                H[i, j] = -2 * J(i, j)
            else:
                H[i, j] = 2 * sum(J(i, k) for k in range(N) if k != i) - 2 * B

    return H

def HxxDHLong(N, C, a, nu, B, alph):
    H = np.zeros((N, N), dtype=float)

    def J(i, j):
        if i == j:
            return 0.0

        r = float(abs(i - j)) * float(a)
        J0 = C / r**nu

        if i in (1, N-2) or j in (1, N-2):
            return alph * J0
        return J0
    
    for i in range(N):
        for j in range(N):
            if i != j:
                H[i, j] = -2 * J(i, j)
            else:
                H[i, j] = - 2 * B
    return H

def HxxDHLong1(N, C, a, nu, B, alph):
    H = np.zeros((N, N), dtype=float)

    def J(i, j):
        if i == j:
            return 0.0

        r = float(abs(i - j)) * float(a)
        J0 = C / r**nu

        if i in (0, N-1) or j in (0, N-1):
            return alph * J0
        return J0
    
    for i in range(N):
        for j in range(N):
            if i != j:
                H[i, j] = -2 * J(i, j)
            else:
                H[i, j] = - 2 * B
    return H


def HxxSHLongSender1(N, C, a, nu, B, alph):
    H = np.zeros((N, N), dtype=float)
    def J(i, j):
        if i == j:
            return 0.0
        r = float(abs(i - j)) * float(a)
        J0 = C / r**nu
        if (i, j) in ((0, 1), (1, 0)):
            return alph * J0
        return J0
    for i in range(N):
        for j in range(N):
            if i != j:
                H[i, j] = -2 * J(i, j)
            else:
                H[i, j] = - 2 * B
    return H


def HxxSHLongSender2(N, C, a, nu, B, alph):
    H = np.zeros((N, N), dtype=float)
    def J(i, j):
        if i == j:
            return 0.0
        r = float(abs(i - j)) * float(a)
        J0 = C / r**nu
        if i in (1,1) or j in (1,1):
            return alph * J0
        return J0
    for i in range(N):
        for j in range(N):
            if i != j:
                H[i, j] = -2 * J(i, j)
            else:
                H[i, j] = - 2 * B
    return H

def HxxSHLongReceiver1(N, C, a, nu, B, alph):
    H = np.zeros((N, N), dtype=float)
    def J(i, j):
        if i == j:
            return 0.0
        r = float(abs(i - j)) * float(a)
        J0 = C / r**nu
        if (i, j) in ((N-1, N-2), (N-2, N-1)):
            return alph * J0
        return J0
    for i in range(N):
        for j in range(N):
            if i != j:
                H[i, j] = -2 * J(i, j)
            else:
                H[i, j] = - 2 * B
    return H


def HxxSHLongReceiver2(N, C, a, nu, B, alph):
    H = np.zeros((N, N), dtype=float)
    def J(i, j):
        if i == j:
            return 0.0
        r = float(abs(i - j)) * float(a)
        J0 = C / r**nu
        if i in (N-2, N-2) or j in (N-2, N-2):
            return alph * J0
        return J0
    for i in range(N):
        for j in range(N):
            if i != j:
                H[i, j] = -2 * J(i, j)
            else:
                H[i, j] = - 2 * B
    return H

def HxxUniform(N, C, a, nu, B, alph):
    H = np.zeros((N, N), dtype=float)
    def J(i, j):
        if i == j:
            return 0.0
        r = float(abs(i - j)) * float(a)
        J0 = C / r**nu
        return J0
    for i in range(N):
        for j in range(N):
            if i != j:
                H[i, j] = -2 * J(i, j)
            else:
                H[i, j] = - 2 * B
    return H
    
# ---------------------------
# Calculation and computation
# ---------------------------

# # Fidelity stuff 

def transfer_amp(Hfunc, N, C, a, nu, B, alp, t_vals):
    H = Hfunc(N, C, a, nu, B, alp)
    eigvals, eigvecs = np.linalg.eigh(H)
    ck = eigvecs[-1, :] * np.conj(eigvecs[0, :])
    phase = np.exp(-1j * np.outer(eigvals, t_vals))
    return ck @ phase

def F_vs_Tvals(Hfunc, t_vals,N, C, a, nu, B, alp):
    amp = transfer_amp(Hfunc, N, C, a, nu, B, alp, t_vals)    
    F_t = np.abs(amp)**2 / 6 + np.abs(amp)/3 + 1/2
    return F_t

def amp_vs_Tvals(Hfunc,t_vals,N, C, a, nu, B, alp):
    amp = transfer_amp(Hfunc, N, C, a, nu, B, alp, t_vals)
    return np.abs(amp)

def amp_vs_N(Hfunc, t_vals,N_vals, C, a, nu, B, alp):
    max_F_all = []
    for N in N_vals:
        F_t = amp_vs_Tvals(Hfunc, t_vals, N, C, a, nu, B, alp)
        idx_num = np.argmax(F_t)
        max_F_all.append(F_t[idx_num])
    return max_F_all

def F_vs_N(Hfunc, t_vals,N_vals, C, a, nu, B, alp):
    max_F_all = []
    for N in N_vals:
        F_t = F_vs_Tvals(Hfunc, t_vals, N, C, a, nu, B, alp)
        idx_num = np.argmax(F_t)
        max_F_all.append(F_t[idx_num])
    return max_F_all

# # Ergotropy stuff

def det_rhoC2_1(fn, p, th, ph):
    q = 1
    return q * (
        p * (1 - p) * np.sin(th)**2 * np.cos(ph)**2
        + np.sin(th / 2)**4 * (1 - q)
    )

def det_rhoC3_1(fn, p, th, ph):
    q = 1
    return q * (
        p * (1 - p) * np.sin(th)**2 * np.sin(ph)**2
        + np.sin(th / 2)**4 * (1 - q)
    )

def det_rhoC2_N(fn, p, th, ph):
    q = np.abs(fn)**2
    return q * (
        p * (1 - p) * np.sin(th)**2 * np.cos(ph)**2
        + np.sin(th / 2)**4 * (1 - q)
    )

def det_rhoC3_N(fn, p, th, ph):
    q = np.abs(fn)**2
    return q * (
        p * (1 - p) * np.sin(th)**2 * np.sin(ph)**2
        + np.sin(th / 2)**4 * (1 - q)
    )

def det_rhoC3_N_alt(fn, p, th, ph):
    q = fn
    return q * (
        p * (1 - p) * np.sin(th)**2 * np.sin(ph)**2
        + np.sin(th / 2)**4 * (1 - q)
    )

# # # Time evolution

def tmax_DH_for_N(Hfunc,N,C,a,nu,B,alp):
    H = Hfunc(N, C, a, nu, B, alp)
    eigvals,_ = np.linalg.eigh(H)
    eigvals, eigvecs = np.linalg.eigh(H)

    ck = eigvecs[-1, :] * np.conj(eigvecs[0, :])

    dom = np.argsort(np.abs(ck))[::-1]
    k1, k2 = dom[:2]
    t_approx = np.pi / np.abs(eigvals[k1] - eigvals[k2])
    
    return t_approx, k1, k2

# # # # C2

def Ergs_vs_Tvals_rhoC2(Hfunc, t_vals, N, C, a, nu, B, alp, p, Th, Ph):
    fn = np.abs(transfer_amp(Hfunc, N, C, a, nu, B, alp, t_vals))
    pop = np.sin(Th/2)**2 * np.abs(fn)**2

    Erg_tot_t = B * (
        2 * pop - 1
        + np.sqrt(1-4* det_rhoC2_N(fn,p,Th,Ph))
    )
    Erg_pop_t = 2 * B * np.maximum(2 * pop - 1, 0)
    Erg_coh_t = Erg_tot_t - Erg_pop_t

    return Erg_tot_t, Erg_pop_t, Erg_coh_t

def purity_vs_Tvals_rhoC2(Hfunc, t_vals, N, C, a, nu, B, alp, p, Th, Ph):
    fn = np.abs(
        transfer_amp(Hfunc, N, C, a, nu, B, alp, t_vals)
    )
    det = det_rhoC2_N(fn,p,Th,Ph)
    return 1 - 2 * det

def Effs_vs_Tvals_rhoC2(Hfunc, t_vals, N, C, a, nu, B, alp, p, Th, Ph):
    q = np.abs(
        transfer_amp(Hfunc, N, C, a, nu, B, alp, t_vals)
    )**2

    Erg_in = B * (
        2 * np.sin(Th/2)**2
        - 1
        + np.sqrt(
            1 - 4 * (p * (1-p) * np.sin(Th)**2 * np.cos(Ph)**2)
            )
    )
    Erg_pop_in = 2 * B * np.maximum(2*np.sin(Th/2)**2 - 1, 0)

    root = np.sqrt(
        1 - 4 * q * (
            p * (1-p) * np.sin(Th)**2 * np.cos(Ph)**2
            + np.sin(Th/2)**4 * (1-q)
        )
    )
    Erg_tot_t = B * (
        2 * np.sin(Th/2)**2 * q
        - 1
        + root
    )
    Erg_pop_t = 2 * B * np.maximum(2*np.sin(Th/2)**2 * q - 1, 0)
    
    Eff_tot_t = Erg_tot_t / Erg_in
    Eff_pop_t = np.divide(
        Erg_pop_t,
        Erg_pop_in,
        out=np.zeros_like(Erg_pop_t),
        where=Erg_pop_in != 0
    )

    Eff_coh_t = Eff_tot_t - Eff_pop_t

    return Eff_tot_t, Eff_pop_t, Eff_coh_t

# # # # C3

def Ergs_vs_Tvals_rhoC3(Hfunc, t_vals, N, C, a, nu, B, alp, p, Th, Ph):
    fn = np.abs(transfer_amp(Hfunc, N, C, a, nu, B, alp, t_vals))
    pop = np.sin(Th/2)**2 * np.abs(fn)**2

    Erg_tot_t = B * (
        2 * pop - 1
        + np.sqrt(1-4* det_rhoC3_N(fn,p,Th,Ph))
    )
    Erg_pop_t = 2 * B * np.maximum(2*pop - 1, 0)
    Erg_coh_t = Erg_tot_t - Erg_pop_t

    return Erg_tot_t, Erg_pop_t, Erg_coh_t

def Energies_vs_Tvals_rhoC3(Hfunc, t_vals, N, C, a, nu, B, alp, p, Th, Ph):
    fn = np.abs(transfer_amp(Hfunc, N, C, a, nu, B, alp, t_vals))
    q = np.abs(fn)**2

    x1 = np.sin(Th) * np.cos(Ph)
    y1 = (2*p - 1) * np.sin(Th) * np.sin(Ph)
    z1 = np.cos(Th)

    zN = 1 - q * (1 - z1)

    rN = np.sqrt(zN**2 + q * (x1**2 + y1**2))

    Est_t = -B * zN
    Epas_t = -B * rN
    Erg_t = Est_t - Epas_t

    return Erg_t, Est_t, Epas_t

def purity_vs_Tvals_rhoC3(Hfunc, t_vals, N, C, a, nu, B, alp, p, Th, Ph):
    fn = np.abs(
        transfer_amp(Hfunc, N, C, a, nu, B, alp, t_vals)
    )
    det = det_rhoC3_N(fn,p,Th,Ph)
    return 1 - 2 * det

def delta_purity_vs_Tvals_rhoC3(Hfunc, t_vals, N, C, a, nu, B, alp, p, Th, Ph):
    fn = np.abs(
        transfer_amp(Hfunc, N, C, a, nu, B, alp, t_vals)
    )
    return 2*p*(1-p)*np.sin(Th)**2*np.sin(Ph)**2*(1-fn**2) - 2*fn**2*np.sin(Th/2)**4*(1-fn**2) 

def Avg_erg_vs_Tvals_En(lam,t_vals,N, C, a, nu, B, alp):
    amp = (-1j *np.sin(lam*t_vals/2))**(N-1) 

    q = np.abs(amp)**2
    x = np.sqrt(q * (1 - q))

    Erg_t = (
        q - 1
        + np.abs(1 - 2*q) / 2
        + np.arcsin(2*x) / (4*x)
    )
    print(N) 
    return Erg_t



def Avg_erg_vs_Tvals(Hfunc,t_vals,N, C, a, nu, B, alp):
    amp = transfer_amp(Hfunc, N, C, a, nu, B, alp, t_vals)

    q = np.abs(amp)**2
    x = np.sqrt(q * (1 - q))

    Erg_t = (
        q - 1
        + np.abs(1 - 2*q) / 2
        + np.arcsin(2*x) / (4*x)
    )
    
    print(N) 
    return Erg_t

# # # Max ergotropy for each N

def Ergs_rhoC2_vs_N(Hfunc, t_vals,N_vals, C, a, nu, B, alp,p, Th, Ph):
    max_Erg_tot_all = []
    max_Erg_pop_all = []
    max_Erg_coh_all = []
    for N in N_vals:
        Erg_tot_t,Erg_pop_t,Erg_coh_t = Ergs_vs_Tvals_rhoC2(HxxDHLong,t_vals,N, C, a, nu, B, alp,p,Th,Ph)

        idx_tot_num = np.argmax(Erg_tot_t)
        max_Erg_tot_all.append(Erg_tot_t[idx_tot_num])
        idx_pop_num = np.argmax(Erg_pop_t)
        max_Erg_pop_all.append(Erg_pop_t[idx_pop_num])
        idx_coh_num = np.argmax(Erg_coh_t)
        max_Erg_coh_all.append(Erg_coh_t[idx_coh_num])
    return     max_Erg_tot_all, max_Erg_pop_all, max_Erg_coh_all 

def Effs_rhoC2_vs_N(Hfunc, t_vals,N_vals, C, a, nu, B, alp,p, Th, Ph):
    max_Eff_tot_all = []
    max_Eff_pop_all = []
    max_Eff_coh_all = []
    for N in N_vals:
        Eff_tot_t, Eff_pop_t, Eff_coh_t = Effs_vs_Tvals_rhoC2(Hfunc, t_vals, N, C, a, nu, B, alp, p, Th, Ph) 

        idx_tot_num = np.argmax(Eff_tot_t)
        max_Eff_tot_all.append(Eff_tot_t[idx_tot_num])
        idx_pop_num = np.argmax(Eff_pop_t)
        max_Eff_pop_all.append(Eff_pop_t[idx_pop_num])
        idx_coh_num = np.argmax(Eff_coh_t)
        max_Eff_coh_all.append(Eff_coh_t[idx_coh_num])
    return     max_Eff_tot_all, max_Eff_pop_all, max_Eff_coh_all 

def Ergs_rhoC3_vs_N(Hfunc, t_vals,N_vals, C, a, nu, B, alp,p, Th, Ph):
    max_Erg_tot_all = []
    max_Erg_pop_all = []
    max_Erg_coh_all = []
    for N in N_vals:
        Erg_tot_t,Erg_pop_t,Erg_coh_t = Ergs_vs_Tvals_rhoC3(HxxDHLong,t_vals,N, C, a, nu, B, alp,p,Th,Ph)

        idx_tot_num = np.argmax(Erg_tot_t)
        max_Erg_tot_all.append(Erg_tot_t[idx_tot_num])
        idx_pop_num = np.argmax(Erg_pop_t)
        max_Erg_pop_all.append(Erg_pop_t[idx_pop_num])
        idx_coh_num = np.argmax(Erg_coh_t)
        max_Erg_coh_all.append(Erg_coh_t[idx_coh_num])
    return     max_Erg_tot_all, max_Erg_pop_all, max_Erg_coh_all 

def Avg_erg_vsN(Hfunc, t_vals,N_vals, C, a, nu, B, alp):
    max_Erg_all = []
    for N in N_vals:
        amp = transfer_amp(Hfunc, N, C, a, nu, B, alp, t_vals)

        q = np.abs(amp)**2
        x = np.sqrt(q * (1 - q))

        Erg_t = (
            q - 1
            + np.abs(1 - 2*q) / 2
            + np.arcsin(2*x) / (4*x)
        )

        idx_num = np.argmax(Erg_t)
        max_Erg_all.append(Erg_t[idx_num])
    return max_Erg_all

# -------------------
# Parameter variation
# -------------------

# # Amplitude

def compute_max_F(alp, Hfunc, N, C, a, nu, B, t_vals):
    F = transfer_amp(Hfunc, N, C, a, nu, B, alp, t_vals)
    return np.max(np.abs(F)**2)


def F_var_alp(Hfunc, N, C, a, nu, B,alp_vals, t_vals):
    worker = partial(
        compute_max_F,
        Hfunc=Hfunc,
        N=N,
        C=C,
        a=a,
        nu=nu,
        B=B,
        t_vals=t_vals
    )
    print("Works fine")

    with ProcessPoolExecutor() as executor:
        F_vs_alp = list(executor.map(worker, alp_vals))

    return F_vs_alp

# # Ergo

# # # Optimizing alpha

def compute_max_ergs(alp, Hfunc, N, C, a, nu, B, t_vals, p, th, ph):
    fn = np.max(np.abs(transfer_amp(Hfunc, N, C, a, nu, B, alp, t_vals)))
    pop = np.sin(th/2)**2 * np.abs(fn)**2

    Erg_tot = B * (
        2 * pop - 1
        + np.sqrt(1-4* det_rhoC3_N(fn,p,th,ph))
    )
    Erg_pop = 2 * B * np.maximum(2*pop - 1, 0)
    Erg_coh = Erg_tot - Erg_pop

    return Erg_tot, Erg_pop, Erg_coh

def ergs_var_alp(Hfunc, N, C, a, nu, B,alp_vals, t_vals, p, th, ph):
    worker = partial(
        compute_max_ergs,
        Hfunc=Hfunc,
        N=N,
        C=C,
        a=a,
        nu=nu,
        B=B,
        t_vals=t_vals,
        p=p,
        th=th,
        ph=ph
    )

    with ProcessPoolExecutor() as executor:
        results = np.array(list(executor.map(worker, alp_vals)))
    erg_tot_vs_alp = results[:, 0]
    erg_pop_vs_alp = results[:, 1]
    erg_coh_vs_alp = results[:, 2]
    print("Computing done")
    
    return erg_tot_vs_alp, erg_pop_vs_alp, erg_coh_vs_alp

# # # Using alpha

def compute_ergs_N_alp(N, alp, Hfunc, C, a, nu, B, t_vals, p, th, ph):
    f = transfer_amp(Hfunc, N, C, a, nu, B, alp, t_vals)
    tmax = t_vals[np.argmax(np.abs(f)**2)]
    q = np.max(np.abs(f)**2)
    pop = np.sin(th / 2)**2 * q

    Erg_tot = B * (2 * pop - 1+ np.sqrt(1 - 4 * det_rhoC3_N(q, p, th, ph)))
    Erg_pop = 2 * B * np.maximum(2 * pop - 1, 0)
    Erg_coh = Erg_tot - Erg_pop

    return Erg_tot, Erg_pop, Erg_coh, tmax

def Ergs_rhoC3_vs_N_var_alp(Hfunc, N_vals, alp_vals, C, a, nu, B, t_vals, p, th, ph):
    worker = partial(
            compute_ergs_N_alp,
            Hfunc=Hfunc,
            C=C,
            a=a,
            nu=nu,
            B=B,
            t_vals=t_vals,
            p=p,
            th=th,
            ph=ph
        )
    with ProcessPoolExecutor() as executor:
        results = np.array(
            list(executor.map(worker, N_vals, alp_vals))
        )
    erg_tot_vs_alp = results[:, 0]
    erg_pop_vs_alp = results[:, 1]
    erg_coh_vs_alp = results[:, 2]
    tmax_vs_N = results[:,3]
    print("Computing done")
    
    return erg_tot_vs_alp, erg_pop_vs_alp, erg_coh_vs_alp, tmax_vs_N

def amp_vs_N_var_alp(Hfunc, t_vals,N_vals, C, a, nu, B, alp_vals):
    max_F_all = []
    for i, N in enumerate(N_vals):
        alp = alp_vals[i]
        F_t = amp_vs_Tvals(Hfunc, t_vals, N, C, a, nu, B, alp)
        idx_num = np.argmax(F_t)
        max_F_all.append(F_t[idx_num])
    return max_F_all


# Energy band
def f(k, N, alpha, eta):
    rhs = alpha**2 / (2 - alpha**2)

    if eta == 1:
        return 1/np.tan(k) * 1/np.tan((N-1)*k/2) - rhs
    else:
        return -1/np.tan(k) * np.tan((N-1)*k/2) - rhs


def find_k(N, alpha, eta, ngrid=int(5e3)):
    k = np.linspace(1e-10, np.pi - 1e-10, ngrid)
    y = f(k, N, alpha, eta)

    roots = []

    for i in range(len(k) - 1):

        if not np.isfinite(y[i]) or not np.isfinite(y[i+1]):
            continue

        if y[i] * y[i+1] < 0:
            try:
                r = brentq(
                    f,
                    k[i],
                    k[i+1],
                    args=(N, alpha, eta)
                )

                if abs(f(r, N, alpha, eta)) < 1e-10:
                    if not roots or abs(r - roots[-1]) > 1e-10:
                        roots.append(r)

            except ValueError:
                pass

    # Special odd-N solution
    if N % 2 == 1:
        k0 = np.pi / 2

        if not any(abs(k0 - r) < 1e-10 for r in roots):
            roots.append(k0)

    return np.sort(np.array(roots))

def analytic_eigvec(N, k, alpha, eta):
    c2 = (
        (N - 1)
        * (
            2 * (1 - alpha**2) * np.cos(k)**2
            + alpha**4 / 2
        )
        + 2 * alpha**2
        - alpha**4
    )

    c = np.sqrt(c2)

    a = np.zeros(N)

    # Boundary sites
    a[0] = alpha * np.sin(k) / c
    a[-1] = eta * alpha * np.sin(k) / c

    # Bulk: i = 2,...,N-1
    i = np.arange(2, N)
    a[1:-1] = (
        np.sin(i * k)
        + (1 - alpha**2) * np.sin((i - 2) * k)
    ) / c

    return a