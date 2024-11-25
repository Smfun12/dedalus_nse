import numpy as np
import copy
import scipy as sp
def P_N(u_hat, u_obs, particle_locations, x, y, scale=False):
    x_flatten = x.flatten()
    y_flatten = y.flatten()
    x_pts_per = np.array([])
    y_pts_per = np.array([])
    u_obss = np.array([])
    X, Y = np.meshgrid(x_flatten, y_flatten, indexing='ij')
    grid_points = np.vstack([X.flatten(), Y.flatten()]).T
    points = copy.deepcopy(particle_locations)
    lagrangian_x = particle_locations[:,0]
    lagrangian_y = particle_locations[:,1]
    values = u_obs['g'].flatten()

    interp_lag = sp.interpolate.RegularGridInterpolator((x_flatten, y_flatten), u_obs['g'], bounds_error=False,
                                                    fill_value=None,
                                                    method='linear')
    Z_inter = interp_lag(points)
    for i in range(-1, 2):
        for j in range(-1, 2):
            x_pts_per = np.concatenate((x_pts_per, lagrangian_x + 2*np.pi*i))
            y_pts_per = np.concatenate((y_pts_per, lagrangian_y + 2*np.pi*j))
            u_obss = np.concatenate((u_obss, Z_inter))

    periodic_values = sp.interpolate.griddata((x_pts_per, y_pts_per), u_obss, (X, Y), method='linear')

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
    grid_points = np.vstack([X.flatten(), Y.flatten()]).T
    points = copy.deepcopy(particle_locations)
    lagrangian_x = particle_locations[:,0]
    lagrangian_y = particle_locations[:,1]
    values = u_obs['g'].flatten()

    interp_lag = sp.interpolate.RegularGridInterpolator((x_flatten, y_flatten), u_obs['g'], bounds_error=False,
                                                    fill_value=None,
                                                    method='linear')
    Z_inter = interp_lag(points)
    
    for i in range(-1, 2):
        for j in range(-1, 2):
            x_pts_per = np.concatenate((x_pts_per, lagrangian_x + 2*np.pi*i))
            y_pts_per = np.concatenate((y_pts_per, lagrangian_y + 2*np.pi*j))
            u_obss = np.concatenate((u_obss, Z_inter))


    periodic_values = sp.interpolate.griddata((x_pts_per, y_pts_per), u_obss, (X, Y), method='linear')

    interp = sp.interpolate.RegularGridInterpolator((x_flatten, y_flatten), u_hat['g'], bounds_error=False,
                                                    fill_value=None,
                                                    method='linear')

    x_grid, y_grid = np.meshgrid(x_flatten, y_flatten, indexing='ij')
    u_hat_inter = interp((x_grid, y_grid))

    result = u_hat_inter - periodic_values

    return result