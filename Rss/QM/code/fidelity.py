import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "pgf.texsystem": "pdflatex",
        "pgf.rcfonts": False,
    }
)

# Parameters
N = 50          # chain length
s = 1           # sender site
r = N           # receiver site
J = 1/2

# Symbols
t = sp.symbols('t', real=True)

# Normalization coefficient
def a_m(m, N):
    if m == 1:
        return 1 / sp.sqrt(N)
    return sp.sqrt(2 / N)

t_vals = np.linspace(0, 4000/J, int(4*4000/J))
N_vals = np.arange(2,81)
F_vals = []
max_F  = []
t_max = []
g_max = []

for N in N_vals:
    gN = 0
    r = N
    for m in range(1, N + 1):
        am = a_m(m, N)
        phase = sp.exp(-4 * J * sp.I * t * (1 - sp.cos(sp.pi * (m - 1) / N)) )
        cos_r = sp.cos(sp.pi * (m - 1) * (2 * r - 1) / (2 * N) )
        cos_s = sp.cos(sp.pi * (m - 1) * (2 * s - 1) / (2 * N) )
        gN += am**2 * phase * cos_r * cos_s
    gN_func = sp.lambdify(t, gN, modules='numpy')
    gv = gN_func(t_vals)
    mag = np.abs(gv)
    F_vals = 0.5 + mag / 3 + mag**2 / 6
    idx = np.argmax(F_vals)
    max_F.append(F_vals[idx])
    t_max.append(t_vals[idx])
    # g_max.append(np.argmax(mag))
    print("Computed N  = ", N, "max F = ", F_vals[idx] )


plt.figure(figsize=(10, 5))
plt.xlabel(r'Chain Length $N$')
plt.ylabel(r'max $F$')
plt.title('Maximum Fidelity vs Chain Length')
plt.bar(N_vals,max_F, color="black")
# plt.show()
plt.savefig("fig.pgf",dpi=300)