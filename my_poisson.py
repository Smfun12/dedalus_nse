import numpy as np
import matplotlib.pyplot as plt
import dedalus.public as d3
from dedalus.extras.plot_tools import plot_bot_2d

# Basis
xcoord = d3.CartesianCoordinates('x')
dist = d3.Distributor(xcoord, dtype=np.float32)
xbasis = d3.Chebyshev(xcoord['x'], 1024, bounds=(0, 1))
x = dist.local_grid(xbasis)
# Fields
u = dist.Field(name='u', bases=xbasis)
tau1 = dist.Field(name='tau1', bases=xbasis)
tau2 = dist.Field(name='tau2', bases=xbasis)
f = dist.Field(name='f', bases=xbasis)
f['g'] = np.exp(-1000 * x ** 2)


lift_basis = xbasis
lift = lambda A, n: d3.Lift(A, lift_basis, n)
tau_basis = xbasis.derivative_basis(2)
p1 = dist.Field(bases=tau_basis)
p2 = dist.Field(bases=tau_basis)
p1['c'][-1] = 1
p2['c'][-1] = 1

# Problem
problem = d3.LBVP([u, tau1, tau2, f], namespace=locals())

problem.add_equation("-lap(u) + tau1*p1 + tau2*p2 = f")
problem.add_equation("u(x='left') = 0")
problem.add_equation("u(x='right') = 0")

solver = problem.build_solver()
solver.solve()

ug = u.allgather_data('g')

plt.figure(figsize=(6, 4))
plt.pcolormesh(x.ravel(), ug.T)
plt.gca().set_aspect('equal')
plt.xlabel('x')
plt.ylabel('u')
plt.title("Randomly forced Poisson equation")
plt.tight_layout()
# plt.savefig('my_poisson.pdf')
plt.savefig('my_poisson.png', dpi=200)
