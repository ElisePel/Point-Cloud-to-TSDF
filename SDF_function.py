import numpy as np
import matplotlib
import skfmm

def SDF(list_data, discret_grid, limit, borne):
    res = []
    for df in list_data:
        p = matplotlib.path.Path(df)
        x, y = np.meshgrid(np.linspace(-borne, borne, discret_grid), np.linspace(-borne, borne, discret_grid))
        points = np.vstack((x.flatten(), y.flatten())).T

        grid = p.contains_points(points)
        grid = grid.reshape(discret_grid, discret_grid)

        grid_test = -1 * np.ones((discret_grid, discret_grid))
        grid_test[grid] = 1

        sdf = skfmm.distance(grid_test)
        sdf = np.clip(sdf, -limit, limit)
        sdf = sdf/(limit*10)
        res.append(sdf[::-1,:]) #car on lit une image dans le sens inverse d'une matrice normale
    return res
