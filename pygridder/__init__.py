"""
Simple Python class to grid points, lines, and polygons

"""
import numpy as np
import scipy.spatial as ss
from skimage import draw as skdraw


__version__ = "1.1.0.dev0"


class Gridder(object):
    """
    A simple class that uses a KDTree to allow for gridding of points, lines, and polygons on a regular grid.
    
    """
    def __init__(self, tx, ty, dx=np.inf, centered=False):
        """
        Create a KDTree lookup object.

        :param [2D list/array] tx: x-coordiantes of the grid
        :param [2D list/array] ty: y-coordiantes of the grid
        :param [number] dx: the delta between grid grid coordinates [assumed regular grid]; default is infinity
        :param [bool] centered: flag indicating whether gridpoints denote center or lower-left corner of grid
                                if not center, move grid points to center to facility lookup. default=False
        """
        tx = np.array(tx, copy=True, subok=True)
        ty = np.array(ty, copy=True, subok=True)
        if not centered:
            x = tx.copy()
            y = ty.copy()
            x[:, :-1] = tx[:, :-1] + (tx[:, 1:] - tx[:, :-1]) / 2.
            x[:, -1] += (tx[:, -1] - tx[:, -2]) / 2.
            y[:-1, :] = ty[:-1, :] + (ty[1:, :] - ty[:-1, :]) / 2.
            y[-1, :] += (ty[-1, :] - ty[-2, :]) / 2.
            tx = x
            ty = y
            del x
            del y
        self.tx = tx
        self.ty = ty
        self.dx = dx
        self.tpoints = np.asarray(list(zip(self.tx.ravel(), self.ty.ravel())))
        self.tree = ss.cKDTree(self.tpoints)


    def _kdtree_query(self, x, y):
        """
        Internal method to do spatial lookup and remove points outside the domain of the grid.

        :param [number or iterable of numbers] x: x-coordinates to be gridded 
        :param [number or iterable of numbers] y: y-coordinates to be gridded 
        :return: Tuple containing x-indices [first return] and y-indices [second return] denoting which 
                 grid points have "hits" 
        """
        try:
            points = np.asarray(list(zip(x, y)))
        except TypeError:
            points = np.asarray(list(zip([x], [y])))
        dists, inds = self.tree.query(points, k=1, distance_upper_bound=self.dx)
        bad_inds = np.where(inds >= len(self.tpoints))[0]
        inds = np.delete(inds, bad_inds)
        return np.unravel_index(inds, self.tx.shape)


    def make_empty_grid(self, dtype="int"):
        """
        Create a grid of zeros of the size of the grid used to initialize the gridder.
        
        :param dtype dtype: the datatype to use in the construction of the numpy grid; default=integer 
        :return: a numpy grid of zeros of the type provided
        """
        return np.zeros(self.tx.shape, dtype=dtype)


    def grid_points(self, xs, ys):
        """
        Take a single point or a list of points and return the grid indices that are hits. 
        
        :param [number or iterable of numbers] xs: x-coordinate(s)
        :param [number or iterable of numbers] ys: y-coordiante(s) 
        :return: A list of grid indices corresponding to hits by the supplied points
        """
        xinds, yinds = self._kdtree_query(x=xs, y=ys)
        points = list(zip(xinds, yinds))
        return points


    def grid_lines(self, sxs, sys, exs, eys):
        """
        Take a single line or list of lines and return the grid indices that are hits.
        
        :param [number or iterable of numbers] sxs: starting x-coordinates 
        :param [number or iterable of numbers] sys: starting y-coordinates 
        :param [number or iterable of numbers] exs: ending x-coordinates 
        :param [number or iterable of numbers] eys: ending y-coordinates 
        :return: A list of grid indices corresponding to hits by the lines
        """
        sxinds, syinds = self._kdtree_query(x=sxs, y=sys)
        exinds, eyinds = self._kdtree_query(x=exs, y=eys)
        lines = [skdraw.line(sx, sy, ex, ey) for sx, sy, ex, ey in zip(sxinds, syinds, exinds, eyinds)]
        return lines


    def grid_polygons(self, xs, ys, fill=True):
        """
        Take a single polygon or a list of polygons and return the grid indices that are hits. 
        If you only want to grid the perimeter, set fill keyward to False.
        
        :param [iterable of numbers or iterable of iterables of numbers] xs: x-coordinates 
        :param [iterable of numbers or iterable of iterables of numbers] ys: y-coordinates 
        :param [bool] fill: flag to determine whether or not to fill. True=Yes; False=No; Default=True
        :return: A list of grid indices corresponding to the hits by the polygons
        """
        xinds = []
        yinds = []
        for x, y in zip(xs, ys):
            _xinds, _yinds = self._kdtree_query(x=x, y=y)
            xinds.append(_xinds)
            yinds.append(_yinds)
        if fill:
            polys = []
            for _x, _y in zip(xinds, yinds):
                try:
                    polys.append(skdraw.polygon(_x, _y))
                except ValueError:
                    continue
        else:
            polys = []
            for _x, _y in zip(xinds, yinds):
                try:
                    polys.append(skdraw.polygon_perimeter(_x, _y))
                except ValueError:
                    continue
        return polys