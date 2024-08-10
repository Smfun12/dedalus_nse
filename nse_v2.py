import copy
import os
import random
import time

import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI
from dedalus import public as de
from dedalus.core import operators
import logging
from dedalus.extras import flow_tools
from dedalus.tools import post
import scipy as sp
import particles
from scipy.linalg import solve

logger = logging.getLogger(__name__)


def P_N(F, particle_locations, x, y, scale=False):
    x_flatten = x.flatten()
    y_flatten = y.flatten()
    x_sensors = particle_locations[:, 0].flatten().T
    y_sensors = particle_locations[:, 1].flatten().T

    X, Y = np.meshgrid(x_flatten, y_flatten, indexing='ij')

    points = np.array(list(zip(x_sensors, y_sensors)))

    grid_points = np.array([X.flatten(), Y.flatten()]).T
    interpolated_values = sp.interpolate.griddata(grid_points, F['g'].flatten(), points, method='linear')

    Z_extrapolated = sp.interpolate.griddata(points, interpolated_values, (X, Y), method='linear')
    if np.isnan(Z_extrapolated).any():
        Z_extrapolated = sp.interpolate.griddata(points, interpolated_values, (X, Y), method='nearest')
    Z_extrapolated = np.nan_to_num(Z_extrapolated, nan=0)

    F['g'] = Z_extrapolated.reshape((128, 128))
    # print(F['g'])
    # F['c'][(X >= N) | (Y >= N)] = 0
    # F['g'] = sp.interpolate.RectBivariateSpline(x, y, F['g'])
    if scale:
        F.set_scales(1)

    return F['g']


def P_N_w(F, particle_locations, x, y, scale=False):
    x_flatten = x.flatten()
    y_flatten = y.flatten()
    x_sensors = particle_locations[:, 0].flatten().T
    y_sensors = particle_locations[:, 1].flatten().T
    # x_sensors, y_sensors = np.unique(x_sensors), np.unique(y_sensors)
    X, Y = np.meshgrid(x_flatten, y_flatten, indexing='ij')

    points = np.array(list(zip(x_sensors, y_sensors)))

    grid_points = np.array([X.flatten(), Y.flatten()]).T
    interpolated_values = sp.interpolate.griddata(grid_points, F['g'].flatten(), points, method='linear')

    Z_extrapolated = sp.interpolate.griddata(points, interpolated_values, (X, Y), method='linear')
    if np.isnan(Z_extrapolated).any():
        Z_extrapolated = sp.interpolate.griddata(points, interpolated_values, (X, Y), method='nearest')
    Z_extrapolated = np.nan_to_num(Z_extrapolated, nan=0)

    F['g'] = Z_extrapolated.reshape((128, 128))
    # print(F['g'])
    # F['c'][(X >= N) | (Y >= N)] = 0
    # F['g'] = sp.interpolate.RectBivariateSpline(x, y, F['g'])
    if scale:
        F.set_scales(1)

    return F['g']


# Parameters
Lx, Lz = 2, 2
Nx, Nz = 128, 128
Reynolds = 5e4
stop_sim_time = 3
timestepper = de.timesteppers.RK222
max_timestep = 1e-2
dtype = np.float64

# Bases
x_basis = de.Fourier('x', Nx, interval=(-1, 1), dealias=1)
z_basis = de.Fourier('z', Nz, interval=(-1, 1), dealias=1)
domain = de.Domain([x_basis, z_basis], grid_dtype=np.float64)

# Substitutions
nu = 1 / Reynolds
D = nu

# Problem
driving = operators.GeneralFunction(domain, 'g', P_N, args=[])
driving_v = operators.GeneralFunction(domain, 'g', P_N_w, args=[])
problem = de.IVP(domain, variables=['p', 'p_', 'u', 'w', 'uz', 'wz', 'u_', 'w_', 'uz_', 'wz_'])
problem.parameters['nu'] = nu
problem.parameters['mu'] = 0.5
problem.parameters['driving'] = driving
problem.parameters['driving_v'] = driving_v
# Nudge solution
problem.add_equation("dx(u_) + wz_ = 0", condition="(nx != 0) or (nz != 0)")
problem.add_equation("dt(u_) - nu*(dx(dx(u_)) + dz(uz_)) +dx(p_)= -(dx(u_)*u_ + w_*uz_) - mu*driving")
problem.add_equation("dt(w_) - nu*(dx(dx(w_)) + dz(wz_)) +dz(p_)= -(dx(w_)*u_ + w_*wz_) - mu*driving_v")
problem.add_equation("uz_ - dz(u_) = 0")
problem.add_equation("wz_ - dz(w_) = 0")
problem.add_equation("p_ = 0", condition="(nx == 0) and (nz == 0)")
# Real solution
problem.add_equation("dx(u) + wz = 0", condition="(nx != 0) or (nz != 0)")
problem.add_equation("dt(u) - nu*(dx(dx(u)) + dz(uz)) + dx(p) = -(dx(u)*u + w*uz)")
problem.add_equation("dt(w) - nu*(dx(dx(w)) + dz(wz)) + dz(p) = -(dx(w)*u + w*wz)")
problem.add_equation("uz - dz(u) = 0")
problem.add_equation("wz - dz(w) = 0")
problem.add_equation("p = 0", condition="(nx == 0) and (nz == 0)")

# Solver
solver = problem.build_solver(timestepper)
solver.stop_sim_time = stop_sim_time

# Initial conditions
x, z = domain.all_grids()
u = solver.state['u']
u_ = solver.state['u_']
# uz = solver.state['uz']
w = solver.state['w']
w_ = solver.state['w_']
# wz = solver.state['wz']

u.set_scales(1)
u['g'] = 0.1 * np.sin(2 * np.pi * x / Lx) * np.exp(-z ** 2 / 0.01)
u['g'] += 0.1 * np.sin(2 * np.pi * (x - 0.5) / Lx) * np.exp(-(z - 0.5) ** 2 / 0.01)
u['g'] += 0.1 * np.sin(2 * np.pi * (x - 0.5) / Lx) * np.exp(-(z + 0.5) ** 2 / 0.01)

u_.set_scales(1)
u_['g'] = 0.1 * np.sin(2 * np.pi * x / Lx) * np.exp(-z ** 2 / 0.01)
u_['g'] += 0.1 * np.sin(2 * np.pi * (x - 0.5) / Lx) * np.exp(-(z - 0.5) ** 2 / 0.01)
u_['g'] += 0.1 * np.sin(2 * np.pi * (x - 0.5) / Lx) * np.exp(-(z + 0.5) ** 2 / 0.01)

# Timestepping and output
dt = 0.125
fh_mode = 'overwrite'

solver.stop_sim_time = stop_sim_time

analysis_tasks = []
# Analysis
snapshots = solver.evaluator.add_file_handler('snapshots', sim_dt=0.25, max_writes=50, mode=fh_mode)
# snapshots.add_system(solver.state)
snapshots.add_task('p')
snapshots.add_task('u')
snapshots.add_task('w')
snapshots.add_task('p_')
snapshots.add_task('u_')
snapshots.add_task('w_')

analysis_tasks.append(snapshots)

# CFL
CFL = flow_tools.CFL(solver, initial_dt=dt, cadence=10, safety=0.5, threshold=0.05,
                     max_change=1.5, min_change=0.5, max_dt=max_timestep)
CFL.add_velocities(("u", "w"))
# CFL.add_velocities(("u_", "w_"))

# Initiate particles (N particles)
N = 49
particleTracker = particles.particles(N, domain)

# Equispaced locations
# n = int(np.sqrt(particleTracker.N))
# xn = np.linspace(0, particleTracker.coordLength[0], n + 1)[:-1]
# dx = xn[1] - xn[0]
# xn += dx / 2.
# yn = np.linspace(0, particleTracker.coordLength[1], n + 1)[:-1]
# dy = yn[1] - yn[0]
# yn += dy / 2.
# xn = np.linspace(-1, 1, N)
# yn = np.linspace(-1, 1, N)
# xn = [-1, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, -0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
# xn = [-1, -0.9, 0, 1]
# yn = [-1, -0.9, 0, 1]
# xn = [-1, 1, -1, 1, -0.75, 0.75, -0.75, 0.75, 0, 0.75, -0.75]
every_n_x_sensor = 20
every_n_y_sensor = 20
xn, yn = x[0:128:every_n_x_sensor], z.T[0:128:every_n_y_sensor]
X, Y = np.meshgrid(xn, yn)
# particleTracker.positions = np.array([(xn[i], yn[j]) for i in range(n) for j in range(n)])
particleTracker.positions = np.column_stack([X.ravel(), Y.ravel()])

# np.random.shuffle(xn), np.random.shuffle(yn)
# yn = [-1, -1, 1, 1, -0.75, 0.75, 0.75, -0.75, 0, 0, 0]
# xn = np.linspace(-1, 1, 100)
# yn = np.linspace(-1, 1, 100)
# xn = np.linspace(-1, 1, 20)
# yn = np.linspace(-1, 1, 40)
# yn = [-1, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, -0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
# particleTracker.positions = np.array([(xn[i], yn[j]) for i in range(n) for j in range(n)])
# particleTracker.positions = grid_points
# print(particleTracker.positions, xn, yn)
locs = []
pos = copy.copy(particleTracker.positions)
locs.append(pos)
savet = 0
savedt = 0.25
times = [0.]
savet += savedt
dT = problem.domain.new_field(name='dT')
dT_v = problem.domain.new_field(name='dT_v')
# Flow properties
flow = flow_tools.GlobalFlowProperty(solver, cadence=10)
flow.add_property("u*u/10", name='w2')
# flow.add_property("u_*u_/10", name='w2')
epochs = []
u_errors = []
w_errors = []
# Main loop
try:
    logger.info('Starting main loop')
    start_time = time.time()
    while solver.proceed:
        u_ = solver.state['u_']
        u = solver.state['u']

        dT['g'] = solver.state['u_']['g'] - solver.state['u']['g']
        ground_truth = solver.state['u']['g']
        estimate = solver.state['u_']['g']

        problem.parameters["driving"].args = [dT, particleTracker.positions, x, z]
        problem.parameters["driving"].original_args = [dT, particleTracker.positions, x, z]

        dT_v['g'] = solver.state['w_']['g'] - solver.state['w']['g']
        ground_truth_w = solver.state['w']['g']
        estimate_w = solver.state['w_']['g']

        problem.parameters["driving_v"].args = [dT_v, particleTracker.positions, x, z]
        problem.parameters["driving_v"].original_args = [dT_v, particleTracker.positions, x, z]

        dt = CFL.compute_dt()
        dt = solver.step(dt)

        u_error = np.linalg.norm(ground_truth - estimate)
        w_error = np.linalg.norm(ground_truth_w - estimate_w)

        u_errors.append(u_error)
        w_errors.append(w_error)
        epochs.append(solver.sim_time)

        particleTracker.step(dt, (u_, w_))
        if solver.sim_time >= savet:
            pos = copy.copy(particleTracker.positions)
            locs.append(pos)
            times.append(solver.sim_time)
            savet += savedt
        if (solver.iteration - 1) % 10 == 0:
            print("Norm of u", np.linalg.norm(ground_truth - estimate))
            print("Norm of w", np.linalg.norm(ground_truth_w - estimate_w))
            max_w = np.sqrt(flow.max('w2'))
            logger.info(
                'Iteration=%i, Time=%e, dt=%e, max(w)=%f' % (solver.iteration, solver.sim_time, dt, max_w))
except Exception as e:
    logger.error('Exception raised, triggering end of main loop.')
    print(e)
    raise
finally:
    end_time = time.time()
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    locs = np.array(locs)
    locs = np.transpose(locs, axes=(1, 0, 2))

    if rank == 0:
        np.save('rbLocs', locs)
        np.save('rbTimes', times)
    if not os.path.isdir("plots"):
        os.makedirs("plots")
    np.save('plots/rbErrors', u_errors)
    np.save('plots/wbErrors', w_errors)
    np.save('plots/epochs', epochs)
    logger.info('Iterations: %i' % solver.iteration)
    logger.info('Sim end time: %f' % solver.sim_time)
    logger.info('Run time: %.2f sec' % (end_time - start_time))
    logger.info('Run time: %f cpu-hr' % ((end_time - start_time) / 60 / 60 * domain.dist.comm_cart.size))

    logger.info('beginning join operation')
    for task in analysis_tasks:
        logger.info(task.base_path)
        post.merge_analysis(task.base_path)
    solver.log_stats()
