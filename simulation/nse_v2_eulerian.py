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
    x_pts_per = np.array([])
    y_pts_per = np.array([])
    u_obss = np.array([])
    X, Y = np.meshgrid(x_flatten, y_flatten, indexing='ij')
    points = copy.deepcopy(particle_locations)
    lagrangian_x = particle_locations[:,0]
    lagrangian_y = particle_locations[:,1]

    interp_lag = sp.interpolate.RegularGridInterpolator((x_flatten, y_flatten), u_obs['g'], bounds_error=False,
                                                    fill_value=None,
                                                    method='linear')
    Z_inter = interp_lag(points)
    for i in range(-1, 2):
        for j in range(-1, 2):
            x_pts_per = np.concatenate((x_pts_per, lagrangian_x + 2*np.pi*i))
            y_pts_per = np.concatenate((y_pts_per, lagrangian_y + 2*np.pi*j))
            u_obss = np.concatenate((u_obss, Z_inter))

    # print("Extended domain:", x_pts_per.shape, y_pts_per.shape, u_obss.shape)

    # Nudged solution
    periodic_values = sp.interpolate.griddata((x_pts_per, y_pts_per), u_obss, (X, Y), method='linear')

    # Ground truth solution
    interp = sp.interpolate.RegularGridInterpolator((x_flatten, y_flatten), u_hat['g'], bounds_error=False,
                                                    fill_value=None,
                                                    method='linear')

    x_grid, y_grid = np.meshgrid(x_flatten, y_flatten, indexing='ij')
    u_hat_inter = interp((x_grid, y_grid))
    result = u_hat_inter - periodic_values
    return result


def P_N_w(u_hat, u_obs, particle_locations, x, y, scale=False):
    x_flatten = x.flatten()
    y_flatten = y.flatten()
    x_pts_per = np.array([])
    y_pts_per = np.array([])
    u_obss = np.array([])
    X, Y = np.meshgrid(x_flatten, y_flatten, indexing='ij')
    points = copy.deepcopy(particle_locations)
    lagrangian_x = particle_locations[:,0]
    lagrangian_y = particle_locations[:,1]

    interp_lag = sp.interpolate.RegularGridInterpolator((x_flatten, y_flatten), u_obs['g'], bounds_error=False,
                                                    fill_value=None,
                                                    method='linear')
    Z_inter = interp_lag(points)
    # print(np.isnan(Z_inter).any())
    for i in range(-1, 2):
        for j in range(-1, 2):
            x_pts_per = np.concatenate((x_pts_per, lagrangian_x + 2*np.pi*i))
            y_pts_per = np.concatenate((y_pts_per, lagrangian_y + 2*np.pi*j))
            u_obss = np.concatenate((u_obss, Z_inter))

    # print("Extended domain:", x_pts_per.shape, y_pts_per.shape, u_obss.shape)

    # Nudged solution
    periodic_values = sp.interpolate.griddata((x_pts_per, y_pts_per), u_obss, (X, Y), method='linear')
    # print(periodic_values.shape)

    # Ground-truth solution
    interp = sp.interpolate.RegularGridInterpolator((x_flatten, y_flatten), u_hat['g'], bounds_error=False,
                                                    fill_value=None,
                                                    method='linear')

    x_grid, y_grid = np.meshgrid(x_flatten, y_flatten, indexing='ij')
    u_hat_inter = interp((x_grid, y_grid))
    result = u_hat_inter - periodic_values

    return result



# Parameters
Lx, Lz = 2 * np.pi, 2 * np.pi
Nx, Nz = 128, 128
Reynolds = 2000
stop_sim_time = 50
timestepper = de.timesteppers.RK222
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
problem.parameters['mu'] = 10
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

# Initial condition
u.set_scales(1)
w.set_scales(1)
ic = sp.io.loadmat("ic.m")
ic2 = sp.io.loadmat("ic2.m")

strength1 = 7
x_sink1, y_sink1 = 0, 0.0
Xss, Yss = np.meshgrid(x, z, indexing='ij')
u1 = -strength1 * (Xss - x_sink1) / (np.sqrt((Xss - x_sink1) ** 2 + (Yss - y_sink1) ** 2) + 1e-9)
v1 = -strength1 * (Yss - y_sink1) / (np.sqrt((Xss - x_sink1) ** 2 + (Yss - y_sink1) ** 2) + 1e-9)

# frequency = 0.5  # Frequency factor, < 1 makes the vortices larger
# u2 = np.sin(frequency * Xss) * np.cos(frequency * Yss)
# v2 = -np.cos(frequency * Xss) * np.sin(frequency * Yss)
# u['g'] = 0.1 * u1
# w['g'] = 0.1 * v1

u['g'] = 0.1 * np.array(ic['u1_cut'])
w['g'] = 0.1 * np.array(ic2['u2_cut'])

u_.set_scales(1)
w_.set_scales(1)
u_['g'] = np.zeros(u['g'].shape)
w_['g'] = np.zeros(w['g'].shape)

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
# wx = w.differentiate('x')
# uz = u.differentiate('z')
# snapshots.add_task(wx - uz, name='vorticity')
# wx_ = w_.differentiate('x')
# uz_ = u_.differentiate('z')
# snapshots.add_task(wx_ - uz_, name='vorticity_')

analysis_tasks.append(snapshots)

# CFL
CFL = flow_tools.CFL(solver, initial_dt=dt, cadence=10, safety=0.5, threshold=0.05,
                     max_change=1.5, min_change=0.5, max_dt=max_timestep)
CFL.add_velocities(("u", "w"))

# Initiate particles (N particles)
every_n_x_sensor = 3
every_n_y_sensor = 3
N = math.ceil(Nx / every_n_x_sensor) * math.ceil(Nz / every_n_y_sensor)
particleTracker = particles.particles(N, domain)

xn, yn = x[0:Nx:every_n_x_sensor], z.T[0:Nz:every_n_y_sensor]
X, Y = np.meshgrid(xn, yn)
particleTracker.positions = np.column_stack([X.ravel(), Y.ravel()])
temp_pos = particleTracker.positions
locs = []
pos = copy.copy(particleTracker.positions)
init_pos = copy.copy(particleTracker.positions)
locs.append(pos)
savet = 0
savedt = 0.25
times = [0.]
savet += savedt
dT = problem.domain.new_field(name='dT')
# Flow properties
flow = flow_tools.GlobalFlowProperty(solver, cadence=10)
flow.add_property("u*u/10", name='w2')

epochs = []
u_errors = []
w_errors = []
# Main loop
gate = 1    
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
        problem.parameters["driving"].original_args = [solver.state['u_'], solver.state['u'], particleTracker.positions,x, z]

        problem.parameters["driving_v"].args = [solver.state['w_'], solver.state['w'], particleTracker.positions, x, z]
        problem.parameters["driving_v"].original_args = [solver.state['w_'], solver.state['w'], particleTracker.positions, x, z]

        u_error = np.linalg.norm(ground_truth - estimate) / np.linalg.norm(ground_truth)
        w_error = np.linalg.norm(ground_truth_w - estimate_w) / np.linalg.norm(ground_truth_w)

        u_errors.append(u_error)
        w_errors.append(w_error)
        epochs.append(solver.sim_time)

        dt = CFL.compute_dt()
        dt = solver.step(dt)
        
        # if (solver.iteration - 1) % 10 == 0:
            # gate = 0
        particleTracker.step(dt, (u, w), Lx, Nx, gate)
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
