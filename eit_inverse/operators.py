"""Sparse finite differences in C ordering, with no wrap or zero padding."""
import numpy as np
from scipy import sparse


def triangle_regularization_operators(points, triangles):
    """Return area-weighted L2(0) and edge-weighted P0 TV operators.

    ``points`` has shape (n_vertices, 2); integer ``triangles`` has shape
    (n_parameters, 3), with rows in parameter order. For a straight triangular
    mesh, ||M*x||_2^2 = sum(area*x^2), and ||D*x||_1 equals the total variation
    of the piecewise constant field inside the domain. Boundary edges carry
    no penalty: a constant field therefore has zero TV. This uses no FEM API.
    Curved element geometry is approximated by its vertex triangles.
    """
    points = np.asarray(points, dtype=float)
    cells = np.asarray(triangles)
    if (points.ndim != 2 or points.shape[1] != 2 or
            not np.all(np.isfinite(points))):
        raise ValueError("points must be finite 2D coordinates")
    if (cells.ndim != 2 or cells.shape[1] != 3 or len(cells) == 0 or
            not np.issubdtype(cells.dtype, np.integer) or
            cells.min() < 0 or cells.max() >= len(points)):
        raise ValueError("triangles must contain valid integer vertex indices")
    coordinates = points[cells]
    a, b = coordinates[:, 1] - coordinates[:, 0], coordinates[:, 2] - coordinates[:, 0]
    areas = np.abs(a[:, 0]*b[:, 1] - a[:, 1]*b[:, 0]) / 2
    if np.any(areas <= 0):
        raise ValueError("triangles must have positive area")
    owners = {}
    for cell, vertices in enumerate(cells):
        for i, j in ((0, 1), (1, 2), (2, 0)):
            edge = tuple(sorted((int(vertices[i]), int(vertices[j]))))
            owners.setdefault(edge, []).append(cell)
    rows, columns, values = [], [], []
    n_edges = 0
    for (i, j), adjacent in owners.items():
        if len(adjacent) > 2:
            raise ValueError("nonmanifold edge shared by more than two triangles")
        if len(adjacent) == 2:
            length = np.linalg.norm(points[i] - points[j])
            rows.extend((n_edges, n_edges))
            columns.extend(adjacent)
            values.extend((length, -length))
            n_edges += 1
    mass = sparse.diags(np.sqrt(areas), format="csr")
    jumps = sparse.csr_matrix((values, (rows, columns)), shape=(n_edges, len(cells)))
    return mass, jumps


def grid_operator(shape, order, *, spacing=1.0, mask=None):
    """Identity (0), stacked gradient (1), or stacked Hessian (2).

    shape=(ny,nx) or (n,). spacing follows shape: (hy,hx).
    Second order in 2D stacks Dyy, Dxx and sqrt(2)*Dxy. L2 therefore
    penalizes a discrete Hessian Frobenius norm. L1 is componentwise weighted
    Hessian sparsity, not TGV. Only stencils entirely inside mask are retained.
    Penalties are discrete sums; quadrature/volume weighting is caller-owned.
    """
    shape = tuple(shape)
    if len(shape) not in (1, 2) or any(int(n) != n or n < 1 for n in shape):
        raise ValueError("shape must have one or two positive integer dimensions")
    if order not in (0, 1, 2):
        raise ValueError("order must be 0, 1, or 2")
    h = np.broadcast_to(np.asarray(spacing, dtype=float), (len(shape),))
    if not np.all(np.isfinite(h)) or np.any(h <= 0):
        raise ValueError("spacing must be finite and positive")
    active = np.ones(shape, dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    if active.shape != shape or not np.any(active):
        raise ValueError("mask must match shape and contain active cells")
    indices = np.full(shape, -1, dtype=int)
    indices[active] = np.arange(active.sum())
    n = int(active.sum())
    if order == 0:
        return sparse.eye(n, format="csr")
    rows, cols, values = [], [], []
    row = 0

    def append(points, weights):
        nonlocal row
        if all(active[p] for p in points):
            for p, w in zip(points, weights):
                rows.append(row); cols.append(indices[p]); values.append(w)
            row += 1

    for axis in range(len(shape)):
        for p in np.ndindex(shape):
            if p[axis] + order >= shape[axis]:
                continue
            points = [tuple(v + (k if d == axis else 0) for d, v in enumerate(p)) for k in range(order + 1)]
            weights = np.array([-1., 1.]) / h[axis] if order == 1 else np.array([1., -2., 1.]) / h[axis]**2
            append(points, weights)
    if order == 2 and len(shape) == 2:
        for iy in range(shape[0] - 1):
            for ix in range(shape[1] - 1):
                append([(iy, ix), (iy, ix+1), (iy+1, ix), (iy+1, ix+1)],
                       np.array([1., -1., -1., 1.]) * np.sqrt(2) / np.prod(h))
    return sparse.csr_matrix((values, (rows, cols)), shape=(row, n))
