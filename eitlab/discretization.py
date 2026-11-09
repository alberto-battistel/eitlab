"""Affine reconstruction -> FEM mapping and geometric area restriction.

Rows of P MUST match the parameter/DOF order of the FEM Jacobian. This module
does not import a FEM library. Its geometry helper supports straight 2D triangles.
"""
import numpy as np
from scipy import sparse

from ._numeric import real_matrix, real_vector


class ParameterSpace:
    def __init__(self, prolongation, *, restriction=None, offset=None):
        self.P = real_matrix(prolongation, "prolongation")
        self.n_fem, self.ndof = self.P.shape
        if min(self.P.shape) < 1:
            raise ValueError("prolongation must have nonempty dimensions")
        self.R = None if restriction is None else real_matrix(restriction, "restriction")
        if self.R is not None and self.R.shape != (self.ndof, self.n_fem):
            raise ValueError("restriction must have shape (ndof, n_fem)")
        self.offset = np.zeros(self.n_fem) if offset is None else real_vector(offset, self.n_fem, "offset")

    @classmethod
    def identity(cls, ndof, *, offset=None):
        I = sparse.eye(ndof, format="csr")
        return cls(I, restriction=I, offset=offset)

    def to_fem(self, parameters):
        return self.offset + self.P @ real_vector(parameters, self.ndof, "parameters")

    def from_fem(self, field):
        if self.R is None:
            raise ValueError("No restriction was supplied")
        return self.R @ (real_vector(field, self.n_fem, "field") - self.offset)

    def project_jacobian(self, jacobian):
        J = real_matrix(jacobian, "Jacobian")
        if J.shape[1] != self.n_fem:
            raise ValueError("Jacobian columns must match prolongation rows")
        return (J @ self.P).tocsr()


class PixelSpace(ParameterSpace):
    def __init__(self, prolongation, restriction, offset, overlap, fem_areas,
                 pixel_areas, mask, x_edges, y_edges):
        super().__init__(prolongation, restriction=restriction, offset=offset)
        self.overlap = overlap
        self.fem_areas = fem_areas
        # Covered physical areas of active pixels, not full rectangle areas.
        self.pixel_areas = pixel_areas
        self.mask = mask
        self.shape = mask.shape
        self.x_edges, self.y_edges = x_edges, y_edges
        self.active_indices = np.flatnonzero(mask.ravel(order="C"))

    @property
    def centers(self):
        X, Y = np.meshgrid((self.x_edges[1:]+self.x_edges[:-1])/2,
                           (self.y_edges[1:]+self.y_edges[:-1])/2)
        return np.column_stack([X.ravel()[self.active_indices], Y.ravel()[self.active_indices]])

    def to_image(self, parameters, fill=np.nan):
        x = real_vector(parameters, self.ndof, "parameters")
        out = np.full(self.shape, fill, dtype=float)
        out[self.mask] = x
        return out


def _area(polygon):
    if len(polygon) < 3:
        return 0.0
    p = np.asarray(polygon)
    p = p-p[0]  # avoid cancellation for translated meshes
    return abs(float(np.dot(p[:, 0], np.roll(p[:, 1], -1)) -
                     np.dot(p[:, 1], np.roll(p[:, 0], -1)))) / 2


def _clip(polygon, axis, edge, keep_greater):
    if len(polygon) == 0:
        return []
    output = []
    previous = polygon[-1]
    previous_inside = previous[axis] >= edge if keep_greater else previous[axis] <= edge
    for current in polygon:
        inside = current[axis] >= edge if keep_greater else current[axis] <= edge
        if inside != previous_inside:
            t = (edge-previous[axis])/(current[axis]-previous[axis])
            output.append(previous+t*(current-previous))
        if inside:
            output.append(current)
        previous, previous_inside = current, inside
    return output


def triangle_pixel_space(triangles, x_edges, y_edges, *, offset=None):
    """Exact polygon overlap for affine P0 triangles and rectangular pixels.

    triangles[k] is the triangle belonging to FEM parameter k. Grid must cover
    every triangle. A[e,p]=intersection area; P=D_f^-1 A; R=D_p^-1 A.T.
    Empty pixels are removed. P/R preserve constants but are not inverses.
    Curved elements, 3D meshes and higher-order parameter spaces need another
    overlap/projection implementation; do not approximate them silently here.
    """
    t = np.asarray(triangles, dtype=float)
    if t.ndim != 3 or t.shape[1:] != (3, 2) or len(t) == 0 or not np.all(np.isfinite(t)):
        raise ValueError("triangles must be finite with shape (n_fem, 3, 2)")
    xe, ye = [np.asarray(v, dtype=float) for v in (x_edges, y_edges)]
    for e in (xe, ye):
        if e.ndim != 1 or len(e) < 2 or not np.all(np.isfinite(e)) or np.any(np.diff(e) <= 0):
            raise ValueError("Pixel edges must be finite, strictly increasing vectors")
    fem_areas = np.array([_area(tri) for tri in t])
    if np.any(fem_areas <= 0):
        raise ValueError("Degenerate triangles are unsupported")
    nx, ny = len(xe)-1, len(ye)-1
    rows, cols, values = [], [], []
    for k, tri in enumerate(t):
        low, high = tri.min(axis=0), tri.max(axis=0)
        ix0 = max(0, np.searchsorted(xe, low[0], side="right")-1)
        ix1 = min(nx, np.searchsorted(xe, high[0], side="left")+1)
        iy0 = max(0, np.searchsorted(ye, low[1], side="right")-1)
        iy1 = min(ny, np.searchsorted(ye, high[1], side="left")+1)
        for iy in range(iy0, iy1):
            for ix in range(ix0, ix1):
                poly = list(tri)
                for axis, edge, greater in [(0, xe[ix], True), (0, xe[ix+1], False),
                                            (1, ye[iy], True), (1, ye[iy+1], False)]:
                    poly = _clip(poly, axis, edge, greater)
                area = _area(poly)
                if area > 0:
                    rows.append(k); cols.append(iy*nx+ix); values.append(area)
    A = sparse.csr_matrix((values, (rows, cols)), shape=(len(t), nx*ny))
    covered = np.asarray(A.sum(axis=1)).ravel()
    if not np.allclose(covered, fem_areas, rtol=1e-9, atol=0):
        raise ValueError("Pixel grid does not cover every FEM triangle")
    pixel_areas = np.asarray(A.sum(axis=0)).ravel()
    active = pixel_areas > 0
    A = A[:, active]
    pixel_areas = pixel_areas[active]
    P = (sparse.diags(1/fem_areas) @ A).tocsr()
    R = (sparse.diags(1/pixel_areas) @ A.T).tocsr()
    return PixelSpace(P, R, offset, A, fem_areas, pixel_areas,
                      active.reshape(ny, nx), xe.copy(), ye.copy())
