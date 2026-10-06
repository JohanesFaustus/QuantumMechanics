import functions
import importlib

importlib.reload(functions)
from functions import *

detail = int(5*1e3)
alp_diff = 0.05
N_vals = np.arange(2,101)
C = 1
a = 1
nu_vals = [20,100]
B = 1
t_vals = np.linspace(0,1000, detail)

th = np.deg2rad(100)
ph = np.deg2rad(90)
p = 0.5 + np.sqrt(2)/np.tan(th)

Erg_in = B * (2 * np.sin(th/2)**2-1+np.sqrt(1-4*p*(1-p)*np.sin(th)**2 * np.sin(ph)**2))
Erg_pop_in = 2*B * np.maximum(2*np.sin(th/2)**2 - 1, 0)
Erg_coh_in = Erg_in - Erg_pop_in
purity_1 = 1 - 2 * det_rhoC3_1(1,p,th,ph)

# Hs = [HxxSHLongReceiver2,HxxSHLongSender1,HxxSHLongSender2]
H = HxxDHLong1

for nu in nu_vals:
    max_alp_tot_Hs = []
    max_alp_tot_Hs_firstiter = []
    max_alp_pop_Hs = []
    max_alp_coh_Hs = []

    for N in N_vals:
        alp_vals = np.linspace(1e-5, 1, detail)
        
        print(f"\n{H.__name__} start at N={N}, nu={nu}")
        print("Starting First iteration")

        erg_tot_vs_alp, erg_pop_vs_alp, erg_coh_vs_alp = ergs_var_alp(H, N, C, a, nu, B,alp_vals, t_vals, p, th, ph)
        
        max_erg_tot_arg = np.argmax(erg_tot_vs_alp)
        max_erg_tot = erg_tot_vs_alp[max_erg_tot_arg]
        max_alp_erg_tot = alp_vals[max_erg_tot_arg]
        max_alp_tot_Hs_firstiter.append(max_alp_erg_tot)

        print("Starting Second iteration")
        
        second_detail = np.maximum(int(detail/10),3)
        alp_lower_bound = np.maximum(max_alp_erg_tot - alp_diff,0)
        alp_upper_bound = np.minimum(max_alp_erg_tot + alp_diff, 1)
        alp_vals = np.linspace(alp_lower_bound, alp_upper_bound,second_detail)
        
        idx = np.argmin(np.abs(alp_vals - max_alp_erg_tot))
        alp_vals[idx] = max_alp_erg_tot
        alp_vals = np.sort(alp_vals)

        erg_tot_vs_alp, erg_pop_vs_alp, erg_coh_vs_alp = ergs_var_alp(H, N, C, a, nu, B,alp_vals, t_vals, p, th, ph)
        
        max_erg_tot_arg = np.argmax(erg_tot_vs_alp)
        max_alp_erg_tot = alp_vals[max_erg_tot_arg]
        max_alp_tot_Hs.append(max_alp_erg_tot)

        max_erg_pop_arg = np.argmax(erg_pop_vs_alp)
        max_alp_erg_pop = alp_vals[max_erg_pop_arg]
        max_alp_pop_Hs.append(max_alp_erg_pop)

        max_erg_coh_arg = np.argmax(erg_coh_vs_alp)
        max_alp_erg_coh = alp_vals[max_erg_coh_arg]
        max_alp_coh_Hs.append(max_alp_erg_coh)

        print(f"{N} done")

    data_tot = np.column_stack((max_alp_tot_Hs, N_vals))
    data_tot_firstiter = np.column_stack((max_alp_tot_Hs_firstiter, N_vals))
    data_pop = np.column_stack((max_alp_pop_Hs, N_vals))
    data_coh = np.column_stack((max_alp_coh_Hs, N_vals))

    np.savez(
        f"Ergs_vs_N_{H.__name__}_nu{nu}.npz",
        max_alp_tot = data_tot, 
        max_alp_tot_firstiter = data_tot_firstiter, 
        max_alp_pop = data_pop,
        max_alp_coh = data_coh 
    )
        
    print(f"{H.__name__} done")

print("\nALL DONE!")
