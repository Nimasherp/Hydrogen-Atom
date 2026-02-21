import matplotlib.pyplot as plt
import numpy as np 
from numpy.polynomial import legendre as L
import scipy
from scipy.constants import physical_constants
import math

a0 = physical_constants["Bohr radius"][0]


def spherical_harmonicY(theta, phi, l, m):
    P = scipy.special.lpmv(m, l, np.cos(theta))
    N = np.sqrt(
        ((2*l + 1)*math.factorial(l - m)) / 
        (4*math.pi*math.factorial(l + m))
    )
    return (P * N * (np.exp(1j*m*phi)) * (-1)**m)

def radial_solutionR(r, n, l):
    k = n - l - 1
    alpha = 2*l + 1
    La = scipy.special.genlaguerre(k, alpha)

    N = ((2*r)/(n*a0))

    # constante de normalisation
    Norm = np.sqrt((2/(n*a0))**3 * math.factorial(k)/(2*n*math.factorial(n+l)))


    return Norm * (N**l) * np.exp(-N/2) * La(N)

class HydrogenOrbital:
    def __init__(self, n, l, m):
        # Quantum numbers for atom

        # n : Determines energy level
        # l : Determines angular shape
        # m : Determines spatial orientation

        self.n = n
        self.l = l
        self.m = m
        self.nb_of_pos = 100

    def psi(self, r, theta, phi):
        return (
            radial_solutionR(r, self.n, self.l)
            *
            spherical_harmonicY(theta, phi, self.l, self.m)
        )

    def probability_density(self,  r, theta, phi):
        return np.abs(self.psi(r, theta, phi))**2


steps = 100

r = np.linspace(0, 20*a0, steps)
theta = np.linspace(0, np.pi, steps)
phi = np.linspace(0, 2*np.pi, steps)

# Distance between two steps
dr = r[1] - r[0]
dtheta = theta[1] - theta[0]
dphi = phi[1] - phi[0] 

R, T, P = np.meshgrid(r, theta, phi, indexing="ij")

hydrogen = HydrogenOrbital(1,0,0)

psi2 = hydrogen.probability_density(R, T, P)

# The wavefunction is normalized, 
# so the total probability (integral over all space, here we use a sum) should be one.

integral = np.sum(
    psi2 * R**2 * np.sin(T)
) * dr * dtheta * dphi

print(integral)
# space = plt.axes(projection="3d")
# proton = np.array([0, 0 ,0])
# space.scatter(proton[0], proton[1], proton[2], color="red", s=100)
# plt.show()