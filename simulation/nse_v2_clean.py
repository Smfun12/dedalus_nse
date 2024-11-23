import time

import numpy as np
from mpi4py import MPI
from dedalus import public as de
import logging
from dedalus.extras import flow_tools
from dedalus.tools import post

logger = logging.getLogger(__name__)

every_n_sensor = 1

# Parameters
Lx, Lz = 2, 2
Nx, Nz = 128, 128
Reynolds = 1e4
stop_sim_time = 10
timestepper = de.timesteppers.RK111
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
problem = de.IVP(domain, variables=['p', 'u', 'w', 'uz', 'wz'])
problem.parameters['nu'] = nu
problem.parameters['mu'] = 2
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
uz = solver.state['uz']
w = solver.state['w']
wz = solver.state['wz']

u.set_scales(1)
# Initial conditions
gshape = domain.dist.grid_layout.global_shape(scales=1)
slices = domain.dist.grid_layout.slices(scales=1)
rand = np.random.RandomState(seed=42)
noise = rand.standard_normal(gshape)[slices]

# Linear background + perturbations damped at walls
zb, zt = z_basis.interval
pert = 1e-3 * noise * (zt - z) * (z - zb)
u['g'] = pert
# u['g'] = uinit
# w['g'] = amp*np.sin(2.0*np.pi*x/Lx)*np.exp(-(z*z)/(sigma*sigma))
# u.differentiate('z',out=uz)
# w.differentiate('z',out=wz)
# u['g'] = u_init
# u['g'] = 0.1 * np.sin(2 * np.pi * x / Lx) * np.exp(-z ** 2 / 0.01)
# u['g'] += 0.1 * np.sin(2 * np.pi * (x - 0.5) / Lx) * np.exp(-(z - 0.5) ** 2 / 0.01)
# u['g'] += 0.1 * np.sin(2 * np.pi * (x - 0.5) / Lx) * np.exp(-(z + 0.5) ** 2 / 0.01)


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

analysis_tasks.append(snapshots)

# CFL
CFL = flow_tools.CFL(solver, initial_dt=dt, cadence=10, safety=0.5, threshold=0.05,
                     max_change=1.5, min_change=0.5, max_dt=max_timestep)
CFL.add_velocities(("u", "w"))

# Flow properties
flow = flow_tools.GlobalFlowProperty(solver, cadence=10)
flow.add_property("u*u/10", name='w2')

# Main loop
try:
    logger.info('Starting main loop')
    start_time = time.time()
    while solver.proceed:
        dt = CFL.compute_dt()
        dt = solver.step(dt)

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

    logger.info('Iterations: %i' % solver.iteration)
    logger.info('Sim end time: %f' % solver.sim_time)
    logger.info('Run time: %.2f sec' % (end_time - start_time))
    logger.info('Run time: %f cpu-hr' % ((end_time - start_time) / 60 / 60 * domain.dist.comm_cart.size))

    logger.info('beginning join operation')
    for task in analysis_tasks:
        logger.info(task.base_path)
        post.merge_analysis(task.base_path)
    solver.log_stats()
