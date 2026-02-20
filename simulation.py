import matplotlib.pyplot as plt
import numpy as np 
from numpy.polynomial import legendre as L
import scipy
from scipy.constants import physical_constants
import math

a0 = physical_constants["Bohr radius"][0]
a0 = 1


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

    def psi(self, r, theta, phi):
        return (
            radial_solutionR(r, self.n, self.l)
            *
            spherical_harmonicY(theta, phi, self.l, self.m)
        )

    def probability_density(self,  r, theta, phi):
        return np.abs(self.psi(r, theta, phi))**2

hydrogen =  HydrogenOrbital(1, 0, 0)
print(hydrogen.psi(1 , 58, 64))

# space = plt.axes(projection="3d")
# proton = np.array([0, 0 ,0])
# space.scatter(proton[0], proton[1], proton[2], color="red", s=100)
# plt.show()