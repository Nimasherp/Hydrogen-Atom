import matplotlib.pyplot as plt
import numpy as np 
from numpy.polynomial import legendre as L
import scipy
import math



def spherical_harmonicY(theta, phi, l, m):
    P = scipy.special.lpmv(m, l, np.cos(theta))
    N = np.sqrt(
        ((2 * l + 1)*math.factorial(l - m)) / 
        (4*math.pi*math.factorial(l + m))
    )
    return (P * N * (np.exp(1j*m*phi)) * (-1)**m)

class HydrogenOrbital:
    def __init__(self, n, l, m):
        # Quantum numbers for atom

        # n : Determines energy level
        # l : Determines angular shape
        # m : Determines spatial orientation

        self.n = 1
        self.l = 0
        self.m = 0

    def psi(self, r, theta, phi):
        # calcule psi(r,theta,phi)
        pass

    def probability_density(self,  r, theta, phi):
        return np.abs(self.psi(r, theta, phi))**2

   


space = plt.axes(projection="3d")
proton = np.array([0, 0 ,0])
space.scatter(proton[0], proton[1], proton[2], color="red", s=100)
plt.show()