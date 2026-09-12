import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm
from scipy.special import airy, jv, jvp
import subprocess

plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "pgf.texsystem": "pdflatex",
        "pgf.rcfonts": False,
        "font.size": 20,
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

def HxDHLong(N, C, a, nu, B, alph):
    H = np.zeros((N, N), dtype=float)

    def J(i, j):
        if i == j:
            return 0.0

        if i in (1, N-2) or j in (1, N-2):
            return alph
        r = float(abs(i - j)) * float(a)
        return C / r**nu

    for i in range(N):
        for j in range(N):
            if i != j:
                H[i, j] = -2 * J(i, j)
            else:
                H[i, j] = 2 * sum(J(i, k) for k in range(N) if k != i) - 2 * B
    return H

# --------------------
# Calculation function
# --------------------

# Fidelity stuff 

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

def F_vs_N(Hfunc, t_vals,N_vals, C, a, nu, B, alp):
    max_F_all = []
    for N in N_vals:
        F_t = F_vs_Tvals(Hfunc, t_vals, N, C, a, nu, B, alp)
        idx_num = np.argmax(F_t)
        max_F_all.append(F_t[idx_num])
    return max_F_all

# Ergotropy stuff

# Time evolution

def tmax_DH_for_N(N,C,a,nu,B,alp):
    H = HxDHLong(N, C, a, nu, B, alp)
    eigvals,_ = np.linalg.eigh(H)
    eigvals, eigvecs = np.linalg.eigh(H)

    ck = eigvecs[-1, :] * np.conj(eigvecs[0, :])

    # Indices of eigenstates with largest |c_k|
    dom = np.argsort(np.abs(ck))[::-1]
    k1, k2 = dom[:2]
    # k1, k2 = [0,1]
    t_approx = np.pi / np.abs(eigvals[k1] - eigvals[k2])
    
    return t_approx, k1, k2

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

def Ergs_vs_Tvals_rhoC2(Hfunc, t_vals, N, C, a, nu, B, alp, p, Th, Ph):
    q = np.abs(
        transfer_amp(Hfunc, N, C, a, nu, B, alp, t_vals)
    )**2

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

    r = np.sin(Th/2)**2 * q

    Erg_pop_t = 2 * B * np.maximum(2*r - 1, 0)
    # Erg_pop_t = B * (2*r-1 + np.abs(1-2*r))

    Erg_coh_t = Erg_tot_t - Erg_pop_t

    return Erg_tot_t, Erg_pop_t, Erg_coh_t

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

# Max ergotropy for each N

def Ergs_rhoC2_vs_N(Hfunc, t_vals,N_vals, C, a, nu, B, alp,p, Th, Ph):
    max_Erg_tot_all = []
    max_Erg_pop_all = []
    max_Erg_coh_all = []
    for N in N_vals:
        q = np.abs(transfer_amp(Hfunc, N, C, a, nu, B, alp, t_vals))**2

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

        Erg_coh_t = Erg_tot_t - Erg_pop_t

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