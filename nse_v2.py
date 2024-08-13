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

logger = logging.getLogger(__name__)

every_n_sensor = 1


def P_N(F, particle_locations, x, y, scale=False):
    """Calculate the Fourier mode projection of F with N terms."""
    # Set the c_n to zero wherever n > N (in both axes).

    x_flatten = x.flatten()
    y_flatten = y.flatten()
    # print(y_flatten.shape)
    # print(particle_locations)
    # x_sensors = particle_locations[:, 0].flatten().T
    # y_sensors = particle_locations[:, 1].flatten().T
    x_sensors, y_sensors = np.array([-0.7, -0.5, -0.3, -0.1, 0.1, 0.3, 0.5, 0.7]), np.array(
        [-0.7, -0.5, -0.3, -0.1, 0.1, 0.3, 0.5, 0.7])
    x_sensors, y_sensors = x_flatten[0:128:every_n_sensor], y_flatten[0:128:every_n_sensor]
    # Interpolate from grid data onto target points
    interp = sp.interpolate.RegularGridInterpolator((x_flatten, y_flatten), F['g'], bounds_error=False, fill_value=None,
                                                    method='nearest')
    xg, yg = np.meshgrid(x_sensors, y_sensors, indexing='ij')
    interpolated_points = interp((xg, yg))
    data = interpolated_points
    # Interpolate from target points onto full-grid
    interp = sp.interpolate.RegularGridInterpolator((x_sensors, y_sensors), data, bounds_error=False, fill_value=None,
                                                    method='nearest')

    xf, yf = np.meshgrid(x_flatten, y_flatten, indexing='ij')
    interp1 = interp((xf, yf))
    F['g'] = interp1
    # print(F['g'])
    # sp.interpolate.griddata(zipped_array, func_s, (grid_x, grid_y), method='cubic')

    # F['c'][(X >= N) | (Y >= N)] = 0
    # F['g'] = sp.interpolate.RectBivariateSpline(x, y, F['g'])
    if scale:
        F.set_scales(1)

    return F['g']


def P_N_w(F, particle_locations, x, y, scale=False):
    """Calculate the Fourier mode projection of F with N terms."""
    # Set the c_n to zero wherever n > N (in both axes).

    x_flatten = x.flatten()
    y_flatten = y.flatten()
    # print(y_flatten.shape)
    # print(particle_locations)
    # x_sensors = particle_locations[:, 0].flatten().T
    # y_sensors = particle_locations[:, 1].flatten().T
    x_sensors, y_sensors = np.array([-0.7, -0.5, -0.3, -0.1, 0.1, 0.3, 0.5, 0.7]), np.array(
        [-0.7, -0.5, -0.3, -0.1, 0.1, 0.3, 0.5, 0.7])
    x_sensors, y_sensors = x_flatten[0:128:every_n_sensor], y_flatten[0:128:every_n_sensor]

    # Interpolate from grid data onto target points
    interp = sp.interpolate.RegularGridInterpolator((x_flatten, y_flatten), F['g'], bounds_error=False, fill_value=None,
                                                    method='nearest')
    xg, yg = np.meshgrid(x_sensors, y_sensors, indexing='ij')
    interpolated_points = interp((xg, yg))
    data = interpolated_points

    # Interpolate from target points onto full-grid
    interp = sp.interpolate.RegularGridInterpolator((x_sensors, y_sensors), data, bounds_error=False, fill_value=None,
                                                    method='nearest')

    xf, yf = np.meshgrid(x_flatten, y_flatten, indexing='ij')
    interp1 = interp((xf, yf))
    F['g'] = interp1
    # print(F['g'])
    # sp.interpolate.griddata(zipped_array, func_s, (grid_x, grid_y), method='cubic')

    # F['c'][(X >= N) | (Y >= N)] = 0
    # F['g'] = sp.interpolate.RectBivariateSpline(x, y, F['g'])
    if scale:
        F.set_scales(1)

    return F['g']


# Parameters
Lx, Lz = 2 * np.pi, 2 * np.pi
Nx, Nz = 128, 128
Reynolds = 5e4
stop_sim_time = 10
timestepper = de.timesteppers.RK111
max_timestep = 1e-2
dtype = np.float64

# Bases
x_basis = de.Fourier('x', Nx, interval=(-np.pi, np.pi), dealias=1)
z_basis = de.Fourier('z', Nz, interval=(-np.pi, np.pi), dealias=1)
domain = de.Domain([x_basis, z_basis], grid_dtype=np.float64)

# Substitutions
nu = 1 / Reynolds
D = nu

# Problem
driving = operators.GeneralFunction(domain, 'g', P_N, args=[])
driving_v = operators.GeneralFunction(domain, 'g', P_N_w, args=[])
problem = de.IVP(domain, variables=['p', 'p_', 'u', 'w', 'uz', 'wz', 'u_', 'w_', 'uz_', 'wz_'])
# problem = de.IVP(domain, variables=['p', 'u_', 'w_', 'uz_', 'wz_'])
problem.parameters['nu'] = nu
problem.parameters['mu'] = 2
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
# problem.add_bc("u(z='left') = 0")
# problem.add_bc("w(z='left') = 0")
# problem.add_bc("u(z='right') = 0")
# problem.add_bc("w(z='right') = 0", condition="(nx != 0)")
# problem.add_bc("integ(p) = 0", condition="(nx == 0)")  # Pressure gauge

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

# Initial condition
u.set_scales(1)
ic = sp.io.loadmat("ic.m")
u['g'] = np.array(ic['u1_cut'])

u_.set_scales(1)
u_['g'] = 0.1 * np.array(ic['u1_cut'])

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
N = 16384
particleTracker = particles.particles(N, domain)

xn, yn = x[0:128:every_n_sensor], z.T[0:128:every_n_sensor]
X, Y = np.meshgrid(xn, yn)
particleTracker.positions = np.column_stack([X.ravel(), Y.ravel()])

locs = []
pos = copy.copy(particleTracker.positions)
locs.append(pos)
savet = 0
savedt = 0.25
times = [0.]
savet += savedt
dT = problem.domain.new_field(name='dT')
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

        dT = solver.state['u_'] - solver.state['u']
        ground_truth = solver.state['u']['g']
        estimate = solver.state['u_']['g']

        dT_w = solver.state['w_'] - solver.state['w']
        ground_truth_w = solver.state['w']['g']
        estimate_w = solver.state['w_']['g']

        problem.parameters["driving"].args = [dT, particleTracker.positions, x, z]
        problem.parameters["driving"].original_args = [dT, particleTracker.positions, x, z]

        problem.parameters["driving_v"].args = [dT_w, particleTracker.positions, x, z]
        problem.parameters["driving_v"].original_args = [dT_w, particleTracker.positions, x, z]
        dt = CFL.compute_dt()
        dt = solver.step(dt)

        u_error = np.linalg.norm(ground_truth - estimate)
        w_error = np.linalg.norm(ground_truth_w - estimate_w)

        u_errors.append(u_error)
        w_errors.append(w_error)
        epochs.append(solver.sim_time)

        particleTracker.step(dt, (u_, w_))
        particleTracker.positions = np.column_stack([X.ravel(), Y.ravel()])
        if solver.sim_time >= savet:
            pos = copy.copy(particleTracker.positions)
            locs.append(pos)
            times.append(solver.sim_time)
            savet += savedt
        if (solver.iteration - 1) % 10 == 0:
            print("Norm of u error", u_error)
            print("Norm of w error", w_error)
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
