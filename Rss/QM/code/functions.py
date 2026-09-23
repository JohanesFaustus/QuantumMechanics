import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm
from scipy.special import airy, jv, jvp
import subprocess
import os

plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "pgf.texsystem": "pdflatex",
        "pgf.rcfonts": False,
        "font.size": 20,
        # "font.size": 10,
        # "axes.labelsize": 10,
        # "xtick.labelsize": 9,
        # "ytick.labelsize": 9,
        # "lines.linewidth": 0.8,
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

# Ergotropy stuff

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


# # Time evolution

def tmax_DH_for_N(Hfunc,N,C,a,nu,B,alp):
    H = Hfunc(N, C, a, nu, B, alp)
    eigvals,_ = np.linalg.eigh(H)
    eigvals, eigvecs = np.linalg.eigh(H)

    ck = eigvecs[-1, :] * np.conj(eigvecs[0, :])

    # Indices of eigenstates with largest |c_k|
    dom = np.argsort(np.abs(ck))[::-1]
    k1, k2 = dom[:2]
    # k1, k2 = [0,1]
    t_approx = np.pi / np.abs(eigvals[k1] - eigvals[k2])
    
    return t_approx, k1, k2

# # # C2

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

# # Max ergotropy for each N

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

# Parameter variation

def Ergs_rhoC3_var_alp(Hfunc, t_vals,N_vals, C, a, nu, B, alp_vals,p, Th, Ph):
    max_alp_tot = []
    max_alp_pop = []
    max_alp_coh = []
    for N in N_vals:
        max_Erg_tot_all = []
        max_Erg_pop_all = []
        max_Erg_coh_all = []
        for alp in alp_vals:
            Erg_tot_t,Erg_pop_t,Erg_coh_t = Ergs_vs_Tvals_rhoC3(HxxDHLong,t_vals,N, C, a, nu, B, alp,p,Th,Ph)

            max_Erg_tot_all.append(np.max(Erg_tot_t))
            max_Erg_pop_all.append(np.max(Erg_pop_t))
            max_Erg_coh_all.append(np.max(Erg_coh_t))
        
        max_alp_tot.append((alp_vals[np.argmax(max_Erg_tot_all)], N))
        max_alp_pop.append((alp_vals[np.argmax(max_Erg_pop_all)], N))
        max_alp_coh.append((alp_vals[np.argmax(max_Erg_coh_all)], N))
    return max_alp_tot,max_alp_pop,max_alp_pop

def Ergs_rhoC3_vs_N_var_alp(Hfunc, t_vals,N_vals, C, a, nu, B, alp_vals,p, Th, Ph):
    max_Erg_tot_all = []
    max_Erg_pop_all = []
    max_Erg_coh_all = []
    for i, N in enumerate(N_vals):
        alp = alp_vals[i]
        Erg_tot_t,Erg_pop_t,Erg_coh_t = Ergs_vs_Tvals_rhoC3(HxxDHLong,t_vals,N, C, a, nu, B, alp,p,Th,Ph)

        idx_tot_num = np.argmax(Erg_tot_t)
        max_Erg_tot_all.append(Erg_tot_t[idx_tot_num])
        idx_pop_num = np.argmax(Erg_pop_t)
        max_Erg_pop_all.append(Erg_pop_t[idx_pop_num])
        idx_coh_num = np.argmax(Erg_coh_t)
        max_Erg_coh_all.append(Erg_coh_t[idx_coh_num])
    return     max_Erg_tot_all, max_Erg_pop_all, max_Erg_coh_all 