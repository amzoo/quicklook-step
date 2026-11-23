use nalgebra_glm as glm;
use glm::{DVec3, DVec4, DMat4};

use nurbs::{AbstractCurve, NDBSplineCurve, SampledCurve};
use crate::surface::{Surface, MAX_SEGMENT_ANGLE};

#[derive(Debug)]
pub enum Curve {
    // TODO: move this to a standalone struct?
    Ellipse {
        eplane_from_world: DMat4,
        world_from_eplane: DMat4,
        closed: bool,
        dir: bool
    },
    Line,
    BSplineCurveWithKnots(SampledCurve<3>, Sense),
    NURBSCurve(SampledCurve<4>, Sense),
    Polyline(Vec<DVec3>),
}

/// How an edge travels along its underlying spline curve.
#[derive(Debug, Clone, Copy)]
pub struct Sense {
    /// The edge starts and ends at the same vertex (e.g. a hole's circle)
    pub closed: bool,
    /// The curve itself is closed, so an edge may wrap past its seam
    pub periodic: bool,
    /// Travel in the direction of increasing curve parameter
    pub dir: bool,
}

impl Curve {
    pub fn new_ellipse(location: DVec3, axis: DVec3, ref_direction: DVec3,
                       radius1: f64, radius2: f64, closed: bool, dir: bool)
        -> Self
    {
        // Build a rotation matrix to go from flat (XY) to 3D space
        let world_from_eplane = Surface::make_affine_transform(axis,
            radius1 * ref_direction,
            radius2 * axis.cross(&ref_direction),
            location);
        let eplane_from_world = world_from_eplane
            .try_inverse()
            .expect("Could not invert");
        Self::Ellipse {
            world_from_eplane,
            eplane_from_world,
            closed, dir
        }
    }

    pub fn new_circle(location: DVec3, axis: DVec3, ref_direction: DVec3,
                      radius: f64, closed: bool, dir: bool) -> Self {
        Self::new_ellipse(location, axis, ref_direction,
                          radius, radius, closed, dir)
    }

    pub fn new_line() -> Self {
        Self::Line
    }

    fn curve_points<const N: usize>(u: DVec3, v: DVec3, curve: &SampledCurve<N>,
                                    sense: Sense, density: usize) -> Vec<DVec3>
        where NDBSplineCurve<N>: AbstractCurve
    {
        let (t_min, t_max) = (curve.min_u(), curve.max_u());
        let t_start = curve.u_from_point(u);
        let t_end = curve.u_from_point(v);
        // Points per knot span, so that no segment turns by more than
        // MAX_SEGMENT_ANGLE (straight spans need only their end points)
        let per_knot = |a: f64, b: f64| {
            let turn = curve.turning(a, b);
            let n = (turn / MAX_SEGMENT_ANGLE).ceil() as usize;
            // Round up: truncating here gave segments over the angle limit
            let spans = curve.spans_between(a, b);
            ((n * density + spans - 1) / spans).clamp(1, 32 * density)
        };
        let (whole, fwd) = (per_knot(t_min, t_max), per_knot(t_start, t_end));

        let mut c = if sense.closed {
            // A closed edge has the same start and end vertex, so it covers
            // the whole curve.  Walk from the vertex to the end of the
            // domain, then from the start of the domain back to the vertex.
            let mut c = curve.as_polyline(t_start, t_max, whole);
            c.extend(curve.as_polyline(t_min, t_start, whole).into_iter().skip(1));
            if !sense.dir {
                c.reverse();
            }
            c
        } else if sense.periodic && (t_end < t_start) == sense.dir {
            // The edge crosses the seam of a closed curve
            let (a, b) = if sense.dir { (t_start, t_end) } else { (t_end, t_start) };
            let mut c = curve.as_polyline(a, t_max, whole);
            c.extend(curve.as_polyline(t_min, b, whole).into_iter().skip(1));
            if !sense.dir {
                c.reverse();
            }
            c
        } else {
            curve.as_polyline(t_start, t_end, fwd)
        };
        // Drop repeated points (e.g. where the walk wraps past the seam)
        c.dedup_by(|a, b| (*a - *b).norm() < 1e-9);
        if c.len() < 2 {
            return vec![u, v];
        }
        c[0] = u;
        *c.last_mut().unwrap() = v;
        c
    }

    /// The part of a polyline between the vertices nearest `u` and `v`
    fn polyline_points(u: DVec3, v: DVec3, pts: &[DVec3]) -> Vec<DVec3> {
        let nearest = |p: DVec3| (0..pts.len())
            .min_by(|&a, &b| (pts[a] - p).norm()
                .total_cmp(&(pts[b] - p).norm()))
            .unwrap_or(0);
        let (i, j) = (nearest(u), nearest(v));
        let mut c: Vec<DVec3> = if i <= j {
            pts[i..=j].to_vec()
        } else {
            pts[j..=i].iter().rev().cloned().collect()
        };
        if c.len() < 2 {
            return vec![u, v];
        }
        c[0] = u;
        *c.last_mut().unwrap() = v;
        c
    }

    /// Samples the edge from `u` to `v`.  `density` multiplies the number of
    /// points (1 normally; higher to retry a face whose boundary crossed).
    pub fn build(&self, u: DVec3, v: DVec3, density: usize) -> Vec<DVec3> {
        let density = density.max(1);
        match self {
            Self::Line => vec![u, v],
            Self::BSplineCurveWithKnots(curve, sense) =>
                Self::curve_points(u, v, curve, *sense, density),
            Self::NURBSCurve(curve, sense) =>
                Self::curve_points(u, v, curve, *sense, density),
            Self::Polyline(pts) => Self::polyline_points(u, v, pts),
            Self::Ellipse {
                eplane_from_world, world_from_eplane, closed, dir
            } => {
                // Project from 3D into the "ellipse plane".  In the "eplane",
                // the ellipse lies on the unit circle.
                let u_eplane = eplane_from_world *
                               DVec4::new(u.x, u.y, u.z, 1.0);
                let v_eplane = eplane_from_world *
                               DVec4::new(v.x, v.y, v.z, 1.0);

                // Pick the starting angle in the circle's flat plane
                let u_ang = u_eplane.y.atan2(u_eplane.x);
                let mut v_ang = v_eplane.y.atan2(v_eplane.x);
                const PI2: f64 = 2.0 * std::f64::consts::PI;
                if *closed {
                    if *dir {
                        v_ang = u_ang + PI2;
                    } else {
                        v_ang = u_ang - PI2;
                    }
                } else if *dir && v_ang <= u_ang {
                    v_ang += PI2;
                } else if !*dir && v_ang >= u_ang {
                    v_ang -= PI2;
                }

                // One point per MAX_SEGMENT_ANGLE of arc, plus the ends
                let count = 4.max(
                    ((u_ang - v_ang).abs() / MAX_SEGMENT_ANGLE).ceil() as usize * density + 1);

                let mut out_world = vec![u];
                // Walk around the circle, using the true positions for start
                // and end points to improve numerical accuracy.
                for i in 1..(count - 1) {
                    let frac = (i as f64) / ((count - 1) as f64);
                    let ang = u_ang * (1.0 - frac) + v_ang * frac;
                    let pos_eplane = DVec4::new(ang.cos(), ang.sin(), 0.0, 1.0);

                    // Project back into 3D
                    let p = world_from_eplane * pos_eplane;
                    out_world.push(glm::vec4_to_vec3(&p));
                }
                out_world.push(v);
                out_world
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nurbs::{KnotVector, NURBSCurve};

    /// No segment of a sampled full circle may turn by more than the limit
    #[test]
    fn spline_circle_segments_stay_within_angle_limit() {
        let h = std::f64::consts::FRAC_1_SQRT_2;
        let pts = [(1., 0., 1.), (1., 1., h), (0., 1., 1.), (-1., 1., h), (-1., 0., 1.),
                   (-1., -1., h), (0., -1., 1.), (1., -1., h), (1., 0., 1.)];
        let control = pts.iter().map(|&(x, y, w)| DVec4::new(x * w, y * w, 0.0, w)).collect();
        let knots = KnotVector::from_multiplicities(2, &[0., 1., 2., 3., 4.], &[3, 2, 2, 2, 3]);
        let curve = Curve::NURBSCurve(
            SampledCurve::new(NURBSCurve::new(false, knots, control)),
            Sense { closed: true, periodic: true, dir: true });
        let start = DVec3::new(1.0, 0.0, 0.0);
        let p = curve.build(start, start, 1);
        let worst = p.windows(2)
            .map(|w| w[0].normalize().dot(&w[1].normalize()).clamp(-1.0, 1.0).acos())
            .fold(0.0, f64::max);
        assert!(worst <= MAX_SEGMENT_ANGLE + 1e-9,
                "segment turns {} degrees", worst.to_degrees());
    }
}
