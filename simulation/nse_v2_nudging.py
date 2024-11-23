import copy
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


def P_N(F, N, scale=False):
    """Calculate the Fourier mode projection of F with N terms."""
    # Set the c_n to zero wherever n > N (in both axes).

    # X, Y = np.indices(F['g'].shape)
    # F['c'][(X >= N) | (Y >= N)] = 0
    # print("F shape=", F['g'].shape)
    # F['g'] = sp.interpolate.RectBivariateSpline(X, Y, F['g'])
    if scale:
        F.set_scales(1)

    return F['g'] * 0


# Parameters
Lx, Lz = 1.2, 1
Nx, Nz = 128, 256
Reynolds = 5e4
stop_sim_time = 10
timestepper = de.timesteppers.RK222
max_timestep = 1e-2
dtype = np.float64

# Bases
x_basis = de.Fourier('x', Nx, interval=(-1, 1), dealias=3 / 2)
z_basis = de.Fourier('z', Nz, interval=(-1, 1), dealias=3 / 2)
domain = de.Domain([x_basis, z_basis], grid_dtype=np.float64)

# Substitutions
nu = 1 / Reynolds
D = nu

# Problem
# driving = operators.GeneralFunction(domain, 'g', P_N, args=[])
problem = de.IVP(domain, variables=['p_', 'u_', 'w_', 'uz_', 'wz_'])
problem.parameters['nu'] = nu
problem.parameters['mu'] = 0.5
# problem.parameters['driving'] = driving
# Nudge solution
problem.add_equation("dx(u_) + wz_ = 0", condition="(nx != 0) or (nz != 0)")
problem.add_equation("dt(u_) - nu*(dx(dx(u_)) + dz(uz_)) + dx(p_) = -(dx(u_)*u_ + w_*uz_)")
problem.add_equation("dt(w_) - nu*(dx(dx(w_)) + dz(wz_)) + dz(p_) = -(dx(w_)*u_ + w_*wz_)")
problem.add_equation("uz_ - dz(u_) = 0")
problem.add_equation("wz_ - dz(w_) = 0")
problem.add_equation("p_ = 0", condition="(nx == 0) and (nz == 0)")

# Solver
solver = problem.build_solver(timestepper)
solver.stop_sim_time = stop_sim_time

# Initial conditions
x, z = domain.all_grids()
u_ = solver.state['u_']
w_ = solver.state['w_']

u_.set_scales(1)
u_['g'] = 0.1 * np.sin(2 * np.pi * x / Lx) * np.exp(-z ** 2 / 0.01)
u_['g'] += 0.1 * np.sin(2 * np.pi * (x - 0.5) / Lx) * np.exp(-(z - 0.5) ** 2 / 0.01)
u_['g'] += 0.1 * np.sin(2 * np.pi * (x - 0.5) / Lx) * np.exp(-(z + 0.5) ** 2 / 0.01)
# u.differentiate('z', out=uz)
# w.differentiate('z', out=wz)
# u['g'][1] += 0.1 * np.sin(2 * np.pi * x / Lx) * np.exp(-(z - 0.5) ** 2 / 0.01)
# u['g'][1] += 0.1 * np.exp(-(z + 0.5) ** 2 / 0.01)

# Timestepping and output
dt = 0.125
fh_mode = 'overwrite'

solver.stop_sim_time = stop_sim_time

analysis_tasks = []
# Analysis
snapshots = solver.evaluator.add_file_handler('snapshots', sim_dt=0.25, max_writes=50, mode=fh_mode)
# snapshots.add_system(solver.state)
snapshots.add_task('p_')
snapshots.add_task('u_')
snapshots.add_task('w_')

analysis_tasks.append(snapshots)

# CFL
CFL = flow_tools.CFL(solver, initial_dt=dt, cadence=10, safety=0.5, threshold=0.05,
                     max_change=1.5, min_change=0.5, max_dt=max_timestep)
CFL.add_velocities(("u_", "w_"))

# Initiate particles (N particles)
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
# xn = [0.5]
# yn = [0.5]
particleTracker.positions = np.array([(xn[i], yn[j]) for i in range(n) for j in range(n)])

locs = []
pos = copy.copy(particleTracker.positions)
locs.append(pos)
savet = 0
savedt = 0.25
times = [0.]
savet += savedt

# Flow properties
flow = flow_tools.GlobalFlowProperty(solver, cadence=10)
flow.add_property("u_*u_/10", name='w2')

# Main loop
try:
    logger.info('Starting main loop')
    start_time = time.time()
    while solver.proceed:
        # u_ = solver.state['u_']
        # dT = solver.state['u_'] - solver.state['u']
        # print('DT shape=', dT['g'].shape)
        # print('U shape=', u['g'].shape)
        # print('U_ shape=', u_['g'].shape)
        # problem.parameters["driving"].args = [dT, N]
        # problem.parameters["driving"].original_args = [dT, N]
        # print(dT['g'], N)
        # print(dT['g'])
        dt = CFL.compute_dt()
        dt = solver.step(dt)
        # "dt(u) - nu*(dx(dx(u)) + dz(uz)) + dx(p) = -(dx(u)*u + w*uz)"
        # "dt(w) - nu*(dx(dx(w)) + dz(wz)) + dz(p) = -(dx(w)*u + w*wz)"
        # print('dt: ', dt)
        # u_ = solver.state['u']
        # p = solver.state['p']
        # uz = solver.state['uz']
        # wz = solver.state['wz']
        # u_x = u_.differentiate('x')
        # w_x = w.differentiate('x')
        # w_x_x = w_x.differentiate('x')
        # u_x_x = u_x.differentiate('x')
        # p_x = p.differentiate('x')
        # p_z = p.differentiate('z')
        # uz_z = uz.differentiate('z')
        # wz_z = wz.differentiate('z')
        # wz = solver.state['wz']
        # "dx(u) + wz = 0"
        # print("U equation: ", np.max(-nu * (u_x_x['g'] + uz_z['g']) + p_x['g'] + u_x['g'] * u['g'] + w['g'] * uz['g']))
        # print("W equation: ", np.max(-nu * (w_x_x['g'] + wz_z['g']) + p_z['g'] + w_x['g'] * u['g'] + w['g'] * wz['g']))
        particleTracker.step(dt, (u_, w_))
        if solver.sim_time >= savet:
            pos = copy.copy(particleTracker.positions)
            locs.append(pos)
            times.append(solver.sim_time)
            savet += savedt
        if (solver.iteration - 1) % 10 == 0:
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

    logger.info('Iterations: %i' % solver.iteration)
    logger.info('Sim end time: %f' % solver.sim_time)
    logger.info('Run time: %.2f sec' % (end_time - start_time))
    logger.info('Run time: %f cpu-hr' % ((end_time - start_time) / 60 / 60 * domain.dist.comm_cart.size))

    logger.info('beginning join operation')
    for task in analysis_tasks:
        logger.info(task.base_path)
        post.merge_analysis(task.base_path)
    solver.log_stats()
