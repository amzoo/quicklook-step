use std::cmp::min;
use nalgebra_glm::{DVec2, DVec3, TVec};
use crate::{KnotVector, VecF};

#[derive(Debug, Clone)]
pub struct NDBSplineSurface<const D: usize> {
    pub u_open: bool,
    pub v_open: bool,
    pub u_knots: KnotVector,
    pub v_knots: KnotVector,
    control_points: Vec<Vec<TVec<f64, D>>>,
}

/// Non-rational b-spline surface with 3D control points
impl<const D: usize> NDBSplineSurface<D> {
    pub fn new(
        u_open: bool,
        v_open: bool,
        u_knots: KnotVector,
        v_knots: KnotVector,
        control_points: Vec<Vec<TVec<f64, D>>>,
    ) -> Self {
        Self {
            u_open,
            v_open,
            u_knots,
            v_knots,
            control_points,
        }
    }

    pub fn min_u(&self) -> f64 {
        self.u_knots.min_t()
    }
    pub fn max_u(&self) -> f64 {
        self.u_knots.max_t()
    }
    pub fn min_v(&self) -> f64 {
        self.v_knots.min_t()
    }
    pub fn max_v(&self) -> f64 {
        self.v_knots.max_t()
    }

    /// Converts a point at position uv onto the 3D mesh, using basis functions
    /// of order `p + 1` and `q + 1` respectively.
    ///
    /// ALGORITHM A3.5
    pub fn surface_point(&self, uv: DVec2) -> TVec<f64, D> {
        let uspan = self.u_knots.find_span(uv.x);
        let Nu = self.u_knots.basis_funs_for_span(uspan, uv.x);

        let vspan = self.v_knots.find_span(uv.y);
        let Nv = self.v_knots.basis_funs_for_span(vspan, uv.y);

        self.surface_point_from_basis(uspan, &Nu, vspan, &Nv)
    }

    pub fn surface_point_from_basis(&self,
        uspan: usize, Nu: &VecF,
        vspan: usize, Nv: &VecF) -> TVec<f64, D>
    {
        let p = self.u_knots.degree();
        let q = self.v_knots.degree();

        let uind = uspan - p;
        let mut S = TVec::zeros();
        for l in 0..=q {
            let mut temp = TVec::zeros();
            let vind = vspan - q + l;
            for k in 0..=p {
                temp += Nu[k] * self.control_points[uind + k][vind];
            }
            S += Nv[l] * temp;
        }
        S
    }

    /// Returns all derivatives of the surface.  If `D = surface_derivs()`,
    /// `D[k][l]` is the derivative of the surface `k` times in the `u`
    /// direction and `l` times in the `v` direction.
    ///
    /// We compute derivatives up to and including the `d`'th order derivatives.
    ///
    /// ALGORITHM A3.6
    pub fn surface_derivs<const E: usize>(&self, uv: DVec2) -> Vec<Vec<TVec<f64, D>>> {
        let p = self.u_knots.degree();
        let q = self.v_knots.degree();

        // Simple initialization of du
        let du = min(E, p);
        let dv = min(E, q);

        // The output matrix goes all the way to order d, even if some of the
        // surfaces are lower order (those values will be locked at 0)
        let mut SKL = vec![vec![TVec::zeros(); E + 1]; E + 1];

        let uspan = self.u_knots.find_span(uv.x);
        let Nu_deriv = self.u_knots.basis_funs_derivs_for_span(uspan, uv.x, du);

        let vspan = self.v_knots.find_span(uv.y);
        let Nv_deriv = self.v_knots.basis_funs_derivs_for_span(vspan, uv.y, dv);

        let mut temp = vec![TVec::zeros(); q + 1];
        for k in 0..=du {
            for s in 0..=q {
                temp[s] = TVec::zeros();
                for r in 0..=p {
                    temp[s] += Nu_deriv[k][r] * self.control_points[uspan - p + r][vspan - q + s];
                }
            }
            let dd = min(E - k, dv);
            for l in 0..=dd {
                for s in 0..=q {
                    SKL[k][l] += Nv_deriv[l][s] * temp[s];
                }
            }
        }
        SKL
    }

    // Computes the relative scale of U and V, based on average distance between
    // control points in 3D space
    pub fn aspect_ratio(&self) -> f64 {
        let mut u_sum = 0.0;
        let mut v_sum = 0.0;
        // Helper function to find 3-distance even if this is 4D.  Rational
        // (4D) control points are stored in homogeneous form (x*w, y*w, z*w, w),
        // so they must be divided by their weight first; otherwise models
        // that sit far from the origin get wildly wrong distances.
        let to_3d = |a: TVec<f64, D>| {
            let w = if D == 4 { a[3] } else { 1.0 };
            DVec3::new(a[0] / w, a[1] / w, a[2] / w)
        };
        let distance = |a: TVec<f64, D>, b: TVec<f64, D>| (to_3d(a) - to_3d(b)).norm();
        for i in 0..self.control_points.len() {
            for j in 0..self.control_points[i].len() {
                if i > 0 {
                    v_sum += distance(self.control_points[i - 1][j],
                                      self.control_points[i][j]);
                }
                if j > 0 {
                    u_sum += distance(self.control_points[i][j - 1],
                                      self.control_points[i][j]);
                }
            }
        }
        let u_mean = u_sum / self.control_points.len() as f64;
        let v_mean = v_sum / self.control_points[0].len() as f64;

        u_mean / v_mean
    }

    /// Control point in 3D, dividing out the weight for rational surfaces
    fn point_3d(&self, i: usize, j: usize) -> DVec3 {
        let a = self.control_points[i][j];
        let w = if D == 4 { a[3] } else { 1.0 };
        DVec3::new(a[0] / w, a[1] / w, a[2] / w)
    }

    /// Distance below which two control points count as the same point,
    /// scaled to the model so it works far from the origin.
    fn closure_tolerance(&self) -> f64 {
        let mut scale: f64 = 1.0;
        for i in 0..self.control_points.len() {
            for j in 0..self.control_points[i].len() {
                scale = scale.max(self.point_3d(i, j).abs().max());
            }
        }
        scale * 1e-7
    }

    /// True when the surface meets itself along its u = min and u = max
    /// edges (e.g. a full tube), so U wraps around.
    pub fn closed_in_u(&self) -> bool {
        let n = self.control_points.len();
        let tol = self.closure_tolerance();
        n > 1 && (0..self.control_points[0].len()).all(|j|
            (self.point_3d(0, j) - self.point_3d(n - 1, j)).norm() < tol)
    }

    /// True when the surface meets itself along its v = min and v = max edges
    pub fn closed_in_v(&self) -> bool {
        let tol = self.closure_tolerance();
        self.control_points.iter().enumerate().all(|(i, row)| row.len() > 1 &&
            (self.point_3d(i, 0) - self.point_3d(i, row.len() - 1)).norm() < tol)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra_glm::DVec4;

    fn surface<const D: usize>(pts: Vec<Vec<TVec<f64, D>>>) -> NDBSplineSurface<D> {
        let k = KnotVector::from_multiplicities(1, &[0.0, 1.0], &[2, 2]);
        NDBSplineSurface::new(true, true, k.clone(), k, pts)
    }

    /// A unit square patch far from the origin, with rational weights != 1.
    /// The aspect ratio must be 1 regardless of weights or offset, which
    /// requires dividing homogeneous control points by their weight.
    #[test]
    fn aspect_ratio_ignores_weights_and_offset() {
        let (ox, oy) = (5600.0, -4900.0);
        let p = |x: f64, y: f64, w: f64| DVec4::new((ox + x) * w, (oy + y) * w, 0.0, w);
        let weighted = surface::<4>(vec![
            vec![p(0.0, 0.0, 1.0), p(1.0, 0.0, 0.5)],
            vec![p(0.0, 1.0, 0.5), p(1.0, 1.0, 1.0)],
        ]);
        assert!((weighted.aspect_ratio() - 1.0).abs() < 1e-9,
                "got {}", weighted.aspect_ratio());
    }
}
