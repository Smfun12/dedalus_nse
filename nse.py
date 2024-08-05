import copy

import numpy as np
import matplotlib.pyplot as plt
from dedalus import public as d3
import logging

import particles

logger = logging.getLogger(__name__)

# Parameters
Lx, Ly = 1, 2
Nx, Ny = 128, 256
Reynolds = 5e4
stop_sim_time = 5
timestepper = d3.RK222
max_timestep = 1e-2
dtype = np.float64

# Bases
coords = d3.CartesianCoordinates('x', 'y')
dist = d3.Distributor(coords, dtype=dtype)
xbasis = d3.RealFourier(coords['x'], size=Nx, bounds=(0, Lx), dealias=3 / 2)
ybasis = d3.Chebyshev(coords['y'], size=Ny, bounds=(-Ly / 2, Ly / 2), dealias=3 / 2)

# Fields
p = dist.Field(name='p', bases=(xbasis, ybasis))
l = dist.VectorField(coords, name='l', bases=(xbasis, ybasis))
u = dist.VectorField(coords, name='u', bases=(xbasis, ybasis))
tau_p = dist.Field(name='tau_p')

# Forcing
# x, y = dist.local_grids(xbasis, ybasis)
f = dist.Field(name='f', bases=(xbasis, ybasis))

# Substitutions
nu = 1 / Reynolds
ex, ez = coords.unit_vector_fields(dist)
D = nu
x, y = dist.local_grids(xbasis, ybasis)
f['g'] = np.exp(-1000 * (0.5 - x) ** 2 - 1000 * (0.5 - y) ** 2)

# Problem
problem = d3.IVP([u, l, p, tau_p], namespace=locals())
# problem.add_equation("dt(u) + grad(p) - nu*lap(u) = - u@grad(u)")
problem.add_equation("dt(u) + grad(p) - nu*lap(u) = - u@grad(u)")
problem.add_equation("dt(l) = u")
problem.add_equation("div(u) + tau_p = 0")
problem.add_equation("integ(p) = 0")  # Pressure gauge

# Solver
solver = problem.build_solver(timestepper)
solver.stop_sim_time = stop_sim_time
# Initial conditions
u['g'][0] = 1 / 2 + 1 / 2 * (np.tanh((y - 0.5) / 0.1) - np.tanh((y + 0.5) / 0.1))
# Match tracer to shear
l['g'] = u['g'][0]
# Add small vertical velocity perturbations localized to the shear layers
u['g'][1] += 0.1 * np.sin(2 * np.pi * x / Lx) * np.exp(-(y - 0.5) ** 2 / 0.01)
u['g'][1] += 0.1 * np.sin(2 * np.pi * x / Lx) * np.exp(-(y + 0.5) ** 2 / 0.01)

# Analysis
snapshots = solver.evaluator.add_file_handler('snapshots', sim_dt=0.1, max_writes=10)
snapshots.add_task(d3.div(d3.skew(l)), name='tracer')
snapshots.add_task(p, name='pressure')
snapshots.add_task(-d3.div(d3.skew(u)), name='vorticity')

# CFL
CFL = d3.CFL(solver, initial_dt=max_timestep, cadence=10, safety=0.2, threshold=0.1,
             max_change=1.5, min_change=0.5, max_dt=max_timestep)
CFL.add_velocity(u)

# Initiate particles (N particles)
domain = Domain(2, distributor=dist)
domain.basis_object = [xbasis, ybasis]
N = 100
particleTracker = particles.particles(N, domain)

# Equispaced locations
n = int(np.sqrt(particleTracker.N))
xn = np.linspace(0, particleTracker.coordLength[0], n + 1)[:-1]
dx = xn[1] - xn[0]
xn += dx / 2.
yn = np.linspace(0, particleTracker.coordLength[1], n + 1)[:-1]
dy = yn[1] - yn[0]
yn += dy / 2.
particleTracker.positions = np.array([(xn[i], yn[j]) for i in range(n) for j in range(n)])

locs = []
pos = copy.copy(particleTracker.positions)
locs.append(pos)
savet = 0
savedt = 0.25
times = [0.]
savet += savedt

# Flow properties
flow = d3.GlobalFlowProperty(solver, cadence=10)
flow.add_property((u @ ez) ** 2, name='w2')

# Main loop
try:
    logger.info('Starting main loop')
    while solver.proceed:
        timestep = CFL.compute_timestep()
        solver.step(timestep)
        particleTracker.step(timestep, (u))
        if (solver.sim_time >= savet):
            pos = copy.copy(particleTracker.positions)
            locs.append(pos)
            times.append(solver.sim_time)
            savet += savedt
        if (solver.iteration - 1) % 10 == 0:
            max_w = np.sqrt(flow.max('w2'))
            logger.info(
                'Iteration=%i, Time=%e, dt=%e, max(w)=%f' % (solver.iteration, solver.sim_time, timestep, max_w))
except Exception:
    logger.error('Exception raised, triggering end of main loop.')
    raise
finally:
    solver.log_stats()
