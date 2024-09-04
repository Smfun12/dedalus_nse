import copy
import math
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


def P_N(u_hat, u_obs, particle_locations, x, y, scale=False):
    x_flatten = x.flatten()
    y_flatten = y.flatten()

    interp = sp.interpolate.RegularGridInterpolator((x_flatten, y_flatten), u_hat['g'], bounds_error=False,
                                                    fill_value=None,
                                                    method='linear')

    x_grid, y_grid = np.meshgrid(x_flatten, y_flatten, indexing='ij')
    u_hat_inter = interp((x_grid, y_grid))

    X, Y = np.meshgrid(x_flatten, y_flatten, indexing='ij')
    grid_points = np.vstack([X.flatten(), Y.flatten()]).T
    points = copy.deepcopy(particle_locations)
    values = u_obs['g'].flatten()

    Z_inter = sp.interpolate.griddata(grid_points, values, points, method='linear')
    if np.isnan(Z_inter).any():
        nan_mask = np.isnan(Z_inter)

        nan_points = points[nan_mask]

        # Interpolate NaN points using nearest method
        values_nearest = sp.interpolate.griddata(grid_points, values, nan_points, method='nearest')
        Z_inter[nan_mask] = values_nearest

    if np.isnan(Z_inter).any():
        print("nan VALUE!!!")
    interpolated_values = Z_inter
    u_obs_inter = sp.interpolate.griddata(points, interpolated_values, grid_points, method='linear')
    u_obs_inter = u_obs_inter.reshape(Nx, Nz)
    # rbf = sp.interpolate.RBFInterpolator(in)
    if np.isnan(u_obs_inter).any():

        nan_mask = np.isnan(u_obs_inter)

        nan_points = np.vstack([X[nan_mask], Y[nan_mask]]).T

        # Interpolate NaN points using nearest method
        values_nearest = sp.interpolate.griddata(points, interpolated_values, nan_points, method='nearest')

        # Replace NaN values in Z with nearest neighbor interpolated values
        u_obs_inter[nan_mask] = values_nearest

        result = u_hat_inter - u_obs_inter
    else:
        result = u_hat_inter - u_obs_inter

    # if scale:
    #     F.set_scales(1)

    return result


def P_N_w(u_hat, u_obs, particle_locations, x, y, scale=False):
    x_flatten = x.flatten()
    y_flatten = y.flatten()

    interp = sp.interpolate.RegularGridInterpolator((x_flatten, y_flatten), u_hat['g'], bounds_error=False,
                                                    fill_value=None,
                                                    method='linear')

    x_grid, y_grid = np.meshgrid(x_flatten, y_flatten, indexing='ij')
    u_hat_inter = interp((x_grid, y_grid))

    X, Y = np.meshgrid(x_flatten, y_flatten, indexing='ij')
    grid_points = np.vstack([X.flatten(), Y.flatten()]).T
    points = copy.deepcopy(particle_locations)
    values = u_obs['g'].flatten()

    Z_inter = sp.interpolate.griddata(grid_points, values, points, method='linear')
    if np.isnan(Z_inter).any():
        nan_mask = np.isnan(Z_inter)

        nan_points = points[nan_mask]

        # Interpolate NaN points using nearest method
        values_nearest = sp.interpolate.griddata(grid_points, values, nan_points, method='nearest')
        Z_inter[nan_mask] = values_nearest

    if np.isnan(Z_inter).any():
        print("nan VALUE!!!")
    interpolated_values = Z_inter
    u_obs_inter = sp.interpolate.griddata(points, interpolated_values, grid_points, method='linear')
    # rbf = sp.interpolate.RBFInterpolator(in)
    u_obs_inter = u_obs_inter.reshape(Nx, Nz)
    if np.isnan(u_obs_inter).any():

        nan_mask = np.isnan(u_obs_inter)

        nan_points = np.vstack([X[nan_mask], Y[nan_mask]]).T

        # Interpolate NaN points using nearest method
        values_nearest = sp.interpolate.griddata(points, interpolated_values, nan_points, method='nearest')

        # Replace NaN values in Z with nearest neighbor interpolated values
        u_obs_inter[nan_mask] = values_nearest

        result = u_hat_inter - u_obs_inter
    else:
        result = u_hat_inter - u_obs_inter

    # if scale:
    #     F.set_scales(1)

    return result


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

# Solver
solver = problem.build_solver(timestepper)
solver.stop_sim_time = stop_sim_time

# Initial conditions
x, z = domain.all_grids()
u = solver.state['u']
u_ = solver.state['u_']
w = solver.state['w']
w_ = solver.state['w_']

u.set_scales(1)
w.set_scales(1)
ic = sp.io.loadmat("ic.m")
ic2 = sp.io.loadmat("ic2.m")
u['g'] = np.array(ic['u1_cut'])
w['g'] = np.array(ic2['u2_cut'])

u_.set_scales(1)
w_.set_scales(1)
u_['g'] = 0.5 * np.array(ic['u1_cut'])
w_['g'] = 0.7 * np.array(ic2['u2_cut'])

# Timestepping and output
dt = 0.125
fh_mode = 'overwrite'

solver.stop_sim_time = stop_sim_time

analysis_tasks = []
# Analysis
snapshots = solver.evaluator.add_file_handler('snapshots', sim_dt=0.25, max_writes=50, mode=fh_mode)
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

every_n_x_sensor = 1
every_n_y_sensor = 1
N = math.ceil(Nx / every_n_x_sensor) * math.ceil(Nz / every_n_y_sensor)

# Initiate particles (N particles)
particleTracker = particles.particles(N, domain)

xn, yn = np.squeeze(x[0:Nx:every_n_x_sensor]), np.squeeze(z.T[0:Nz:every_n_y_sensor])
X, Y = np.meshgrid(xn, yn)

# particleTracker.positions = np.column_stack([X.ravel(), Y.ravel()])
particleTracker.positions = np.vstack([X.ravel(), Y.ravel()]).T
# particleTracker.positions = np.random.rand(N, 2)*np.pi - np.pi
init_particle_pos = copy.deepcopy(particleTracker.positions)

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

        problem.parameters["driving"].args = [solver.state['u_'], solver.state['u'], particleTracker.positions, x, z]
        problem.parameters["driving"].original_args = [solver.state['u_'], solver.state['u'], particleTracker.positions,
                                                       x, z]

        problem.parameters["driving_v"].args = [solver.state['w_'], solver.state['w'], particleTracker.positions, x, z]
        problem.parameters["driving_v"].original_args = [solver.state['w_'], solver.state['w'],
                                                         particleTracker.positions, x, z]

        u_error = np.linalg.norm(ground_truth - estimate) / np.linalg.norm(ground_truth)
        w_error = np.linalg.norm(ground_truth_w - estimate_w) / np.linalg.norm(ground_truth_w)

        u_errors.append(u_error)
        w_errors.append(w_error)
        epochs.append(solver.sim_time)

        dt = CFL.compute_dt()
        dt = solver.step(dt)

        particleTracker.step(dt, (u, w))
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
    print("difference", np.linalg.norm(init_particle_pos - particleTracker.positions))
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
