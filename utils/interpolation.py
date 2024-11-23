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

    print("Extended domain:", x_pts_per.shape, y_pts_per.shape, u_obss.shape)

    periodic_values = sp.interpolate.griddata((x_pts_per, y_pts_per), u_obss, (X, Y), method='linear')
    # print(np.isnan(periodic_values).any())

    interp = sp.interpolate.RegularGridInterpolator((x_flatten, y_flatten), u_hat['g'], bounds_error=False,
                                                    fill_value=None,
                                                    method='linear')

    x_grid, y_grid = np.meshgrid(x_flatten, y_flatten, indexing='ij')
    u_hat_inter = interp((x_grid, y_grid))

    
    # if np.isnan(Z_inter).any():
    #     nan_mask = np.isnan(Z_inter)

    #     nan_points = points[nan_mask]

    #     # Interpolate NaN points using nearest method
    #     values_nearest = sp.interpolate.griddata(grid_points, values, nan_points, method='nearest')
    #     Z_inter[nan_mask] = values_nearest

    # if np.isnan(Z_inter).any():
    #     print("nan VALUE!!!")
    # interpolated_values = Z_inter
    # u_obs_inter = sp.interpolate.griddata(points, interpolated_values, grid_points, method='linear')
    # u_obs_inter = u_obs_inter.reshape(Nx, Nz)
    # # rbf = sp.interpolate.RBFInterpolator(in)
    # if np.isnan(u_obs_inter).any():

    #     nan_mask = np.isnan(u_obs_inter)

    #     nan_points = np.vstack([X[nan_mask], Y[nan_mask]]).T

    #     # Interpolate NaN points using nearest method
    #     values_nearest = sp.interpolate.griddata(points, interpolated_values, nan_points, method='nearest')

    #     # Replace NaN values in Z with nearest neighbor interpolated values
    #     u_obs_inter[nan_mask] = values_nearest

    #     result = u_hat_inter - u_obs_inter
    # else:
    result = u_hat_inter - periodic_values

    # if scale:
    #     F.set_scales(1)

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
    # print(np.isnan(Z_inter).any())
    for i in range(-1, 2):
        for j in range(-1, 2):
            x_pts_per = np.concatenate((x_pts_per, lagrangian_x + 2*np.pi*i))
            y_pts_per = np.concatenate((y_pts_per, lagrangian_y + 2*np.pi*j))
            u_obss = np.concatenate((u_obss, Z_inter))

    # print("Extended domain:", x_pts_per.shape, y_pts_per.shape, u_obss.shape)

    periodic_values = sp.interpolate.griddata((x_pts_per, y_pts_per), u_obss, (X, Y), method='linear')
    # print(periodic_values.shape)

    interp = sp.interpolate.RegularGridInterpolator((x_flatten, y_flatten), u_hat['g'], bounds_error=False,
                                                    fill_value=None,
                                                    method='linear')

    x_grid, y_grid = np.meshgrid(x_flatten, y_flatten, indexing='ij')
    u_hat_inter = interp((x_grid, y_grid))

    
    # if np.isnan(Z_inter).any():
    #     nan_mask = np.isnan(Z_inter)

    #     nan_points = points[nan_mask]

    #     # Interpolate NaN points using nearest method
    #     values_nearest = sp.interpolate.griddata(grid_points, values, nan_points, method='nearest')
    #     Z_inter[nan_mask] = values_nearest

    # if np.isnan(Z_inter).any():
    #     print("nan VALUE!!!")
    # interpolated_values = Z_inter
    # u_obs_inter = sp.interpolate.griddata(points, interpolated_values, grid_points, method='linear')
    # u_obs_inter = u_obs_inter.reshape(Nx, Nz)
    # # rbf = sp.interpolate.RBFInterpolator(in)
    # if np.isnan(u_obs_inter).any():

    #     nan_mask = np.isnan(u_obs_inter)

    #     nan_points = np.vstack([X[nan_mask], Y[nan_mask]]).T

    #     # Interpolate NaN points using nearest method
    #     values_nearest = sp.interpolate.griddata(points, interpolated_values, nan_points, method='nearest')

    #     # Replace NaN values in Z with nearest neighbor interpolated values
    #     u_obs_inter[nan_mask] = values_nearest

    #     result = u_hat_inter - u_obs_inter
    # else:
    result = u_hat_inter - periodic_values

    # if scale:
    #     F.set_scales(1)

    return result