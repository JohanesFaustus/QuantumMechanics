from qutip import basis, jmat, expect
import numpy as np

hbar = 1
l  = 1

# Define spin-1 angular momentum matrices
Lx = hbar * jmat(l, "x")
Ly = hbar * jmat(l, "y")
Lz = hbar * jmat(l, "z")
L2=Lx**2+Ly**2+Lz**2

# Define state coefficients
psi = (
    (1 / np.sqrt(7)) * basis(3, 2)
    + (2 / np.sqrt(7)) * basis(3, 1)
    + np.sqrt(2 / 7) * basis(3, 0)
)
psi = psi.unit()

# Compute expectation values
ex_Lx = expect(Lx, psi)
ex_Ly = expect(Ly, psi)
ex_Lz = expect(Lz, psi)
ex_L2 = expect(L2, psi)

print("⟨Lx⟩ =", ex_Lx)
print("⟨Ly⟩ =", ex_Ly)
print("⟨Lz⟩ =", ex_Lz)
print("⟨L2⟩ =", ex_L2)