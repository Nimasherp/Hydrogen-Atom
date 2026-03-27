import matplotlib.pyplot as plt
import numpy as np 
from numpy.polynomial import legendre as L
import scipy
from scipy.constants import physical_constants
import math

a0 = physical_constants["Bohr radius"][0]


def spherical_harmonicY(theta, phi, l, m):
    P = scipy.special.lpmv(abs(m), l, np.cos(theta))
    N = np.sqrt(
        ((2*l + 1)*math.factorial(l - abs(m))) / 
        (4*math.pi*math.factorial(l + abs(m)))
    )
    return (P * N * (np.cos(m*phi)) * (-1)**m)

def radial_solutionR(r, n, l):
    k = n - l - 1
    alpha = 2*l + 1
    La = scipy.special.genlaguerre(k, alpha)

    N = ((2*r)/(n*a0)) 

    # constante de normalisation assuring that the integration of the radial part around space is equal to 1
    Norm = np.sqrt((2/(n * a0))**3 * math.factorial(k)/(2 * n*math.factorial(n+l)))


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


# --------------- Setting up parameters --------------- #
steps = 100
# We set up space as 100 x 100 x 100 = 1 000 000 points where the max radius is defined.

n = int(input("Enter quantum number n : "))
l = int(input(f"Enter azimuthal quantum number l (0<=l<n={n}): "))
m = int(input(f"Enter magnetic quantum number m (-l <= m <= l, l={l}): "))

# Checking if values are in their conditions
if not (0 <= l < n):
    raise ValueError(f"l must be 0 <= l < n, got l={l}, n={n}")
if not (-l <= m <= l):
    raise ValueError(f"m must be -l<= m <= l, got m={m}, l={l}")

hydrogen = HydrogenOrbital(n,l,m)

limit = n*10*a0
r = np.linspace(0, limit, steps)
theta = np.linspace(0, np.pi, steps)
phi = np.linspace(0, 2*np.pi, steps)

# Here merge will help us going from (k,k,k) to (i,j,k) positions.
R, T, P = np.meshgrid(r, theta, phi, indexing="ij")

# --------------- Monte Carlo simulation --------------- #
# We first sample spatial points uniformly.
# Then we resample them with weights proportional to |psi|^2.
# This produces a cloud distributed according to the quantum probability density. (Monte Carlo method)

# Flattening the arrays to the shape 1 000 000.
R_flat = R.ravel()
Theta_flat = T.ravel()
Phi_flat = P.ravel()

# N is the number of positions, that will eventually lower
# due to the ones eleminated because of their probability.
N = 100000

idx_randm = np.random.choice(len(R_flat), N, replace=False)

points = np.column_stack((
    R_flat[idx_randm],
    Theta_flat[idx_randm],
    Phi_flat[idx_randm]
))
probabilities = hydrogen.probability_density(points[:, 0], points[:, 1], points[:, 2])
# Normalizing the probabilies
probabilities = probabilities / np.sum(probabilities) 
chosen_idx = np.unique(np.random.choice(len(points), size=N, p=probabilities)) # removing the duplicates (usually the ones with high probability are choosen often)
final_points = points[chosen_idx]

# --------------- Convert to Cartesian --------------- #
r_vals = final_points[:,0]
theta_vals = final_points[:,1]
phi_vals = final_points[:,2]

# Converting from spherical coordinates to cartesian using some basic math.

x = r_vals * np.sin(theta_vals) * np.cos(phi_vals)
y = r_vals * np.sin(theta_vals) * np.sin(phi_vals)
z = r_vals * np.cos(theta_vals)

# --------------- Color by probability --------------- #
# Optional, a way to visualise the positions with high probabilty with colors
# (similar to heat visualisation)

final_probs = hydrogen.probability_density(r_vals, theta_vals, phi_vals)
final_probs *= r_vals**2 * np.sin(theta_vals) # probability density must be multiplied by the volume element
colors = final_probs / np.max(final_probs)  # normalized

# --------------- 3D visualisation --------------- #
fig = plt.figure()
ax = fig.add_subplot(projection="3d")

ax.scatter(0, 0, 0, color="red", s=100) # Proton

# Electron cloud colored by density
# 'jet' is a predifinied colormap, the more the value in color is close to 1
# It goes from blue -> cyan -> yellow -> red
sc = ax.scatter(x, y, z, c=colors, cmap='jet', s=5, alpha=0.6)

# Set camera 
ax.view_init(elev=0, azim=0)

ax.set_title(f'3D — orbital ({n},{l},{m})')

# --------------- 2D visualisation --------------- #
# This 2D visualisation is a cross section on the xz plan, so in this case y = 0 => phi = 0
# In order to visualise the scatter we make a grid with x*z position.

fig1, ax1 = plt.subplots()
grid = 2000
x_scatter = np.linspace(-limit, limit, grid)
z_scatter = np.linspace(-limit, limit, grid)
X2D, Z2D = np.meshgrid(x_scatter, z_scatter)

r = np.sqrt(X2D**2 + Z2D**2)
# We make sure to not divide by r = 0
division = np.divide(Z2D, r, out=np.zeros_like(r), where=(r != 0))
# Since arccos is defined on [-1;1] we don't want the division to be out of those bounds
theta = np.arccos(np.clip(division, -1, 1))
phi = np.full_like(r, 0)

density = hydrogen.probability_density(r, theta, phi) * r**2 *4*np.pi

img = ax1.imshow(density, cmap='magma')
ax1.set_title(f'Cross-section 2D — orbital ({n},{l},{m})')

plt.tight_layout()

plt.show()

# fig1.savefig(f"images/orbital ({n},{l},{m})")
# --------------------------------------------------------------------------------------------------#

