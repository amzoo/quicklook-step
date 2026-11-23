//! Swept surfaces (`SURFACE_OF_LINEAR_EXTRUSION`, `SURFACE_OF_REVOLUTION`).
//!
//! Both are converted into an exact rational B-spline (NURBS) surface, so they
//! reuse the existing NURBS lowering, seam handling and Steiner points.

use std::convert::TryInto;

use nalgebra_glm::{DVec3, DVec4};
use step::{ap214, ap214::Entity, step_file::StepFile};

use nurbs::{KnotVector, NURBSSurface, SampledSurface};

use crate::{
    Error,
    surface::Surface,
    triangulate::{axis2_placement_3d, cartesian_point, control_points_1d, direction},
};

/// A profile curve as a NURBS curve with homogeneous control points
/// (x*w, y*w, z*w, w).
struct Profile {
    knots: KnotVector,
    points: Vec<DVec4>,
}

const HALF_SQRT_2: f64 = std::f64::consts::FRAC_1_SQRT_2;

/// Unit-circle control points and weights of a full rational quadratic circle
const CIRCLE: [(f64, f64, f64); 9] = [
    (1.0, 0.0, 1.0), (1.0, 1.0, HALF_SQRT_2), (0.0, 1.0, 1.0),
    (-1.0, 1.0, HALF_SQRT_2), (-1.0, 0.0, 1.0), (-1.0, -1.0, HALF_SQRT_2),
    (0.0, -1.0, 1.0), (1.0, -1.0, HALF_SQRT_2), (1.0, 0.0, 1.0),
];

fn circle_knots() -> KnotVector {
    KnotVector::from_multiplicities(2, &[0.0, 1.0, 2.0, 3.0, 4.0], &[3, 2, 2, 2, 3])
}

fn homogeneous(p: DVec3, w: f64) -> DVec4 {
    DVec4::new(p.x * w, p.y * w, p.z * w, w)
}

fn euclidean(p: &DVec4) -> DVec3 {
    DVec3::new(p.x / p.w, p.y / p.w, p.z / p.w)
}

fn bspline_knots(degree: i64, knots: &[f64], mults: &[i64]) -> Result<KnotVector, Error> {
    let degree: usize = degree.try_into().map_err(|_| Error::UnknownCurveType)?;
    let mults = mults.iter()
        .map(|&m| m.try_into().map_err(|_| Error::UnknownCurveType))
        .collect::<Result<Vec<usize>, _>>()?;
    Ok(KnotVector::from_multiplicities(degree, knots, &mults))
}

/// Converts a profile curve into NURBS form.  `hint` holds points of the face
/// being triangulated, used to bound curves that are infinite (LINE).
fn profile(s: &StepFile, curve: ap214::Curve, hint: &[DVec3]) -> Result<Profile, Error> {
    Ok(match &s[curve] {
        Entity::Line(l) => {
            let p0 = cartesian_point(s, l.pnt);
            let v = s.entity(l.dir).ok_or(Error::UnknownCurveType)?;
            let d = direction(s, v.orientation).normalize();
            // A LINE is infinite; make it long enough to cover the face
            let reach = hint.iter().map(|p| (p - p0).norm()).fold(0.0, f64::max) + 1.0;
            Profile {
                knots: KnotVector::from_multiplicities(1, &[0.0, 1.0], &[2, 2]),
                points: vec![homogeneous(p0 - d * reach, 1.0),
                             homogeneous(p0 + d * reach, 1.0)],
            }
        },
        Entity::Circle(c) => {
            let r = c.radius.0.0.0;
            ellipse(s, c.position, r, r)
        },
        Entity::Ellipse(c) => ellipse(s, c.position, c.semi_axis_1.0.0.0, c.semi_axis_2.0.0.0),
        Entity::BSplineCurveWithKnots(c) => {
            let knots: Vec<f64> = c.knots.iter().map(|k| k.0).collect();
            Profile {
                knots: bspline_knots(c.degree, &knots, &c.knot_multiplicities)?,
                points: control_points_1d(s, &c.control_points_list)
                    .into_iter().map(|p| homogeneous(p, 1.0)).collect(),
            }
        },
        Entity::ComplexEntity(v) if v.len() == 2 => {
            match (&v[0], &v[1]) {
                (Entity::BSplineCurveWithKnots(b), Entity::RationalBSplineCurve(r)) => {
                    let knots: Vec<f64> = b.knots.iter().map(|k| k.0).collect();
                    Profile {
                        knots: bspline_knots(b.degree, &knots, &b.knot_multiplicities)?,
                        points: control_points_1d(s, &b.control_points_list)
                            .into_iter()
                            .zip(r.weights_data.iter())
                            .map(|(p, w)| homogeneous(p, *w))
                            .collect(),
                    }
                },
                _ => return Err(Error::UnknownCurveType),
            }
        },
        // The face's own edges do the trimming, so the full basis is enough
        Entity::TrimmedCurve(t) => profile(s, t.basis_curve, hint)?,
        _ => return Err(Error::UnknownCurveType),
    })
}

fn ellipse(s: &StepFile, position: ap214::Axis2Placement, r1: f64, r2: f64) -> Profile {
    let (location, axis, ref_direction) = axis2_placement_3d(s, position.cast());
    let x = ref_direction * r1;
    let y = axis.cross(&ref_direction) * r2;
    Profile {
        knots: circle_knots(),
        points: CIRCLE.iter()
            .map(|&(cx, cy, w)| homogeneous(location + x * cx + y * cy, w))
            .collect(),
    }
}

/// `SURFACE_OF_LINEAR_EXTRUSION`: the profile swept along a straight vector.
pub fn extrusion(s: &StepFile, e: &ap214::SurfaceOfLinearExtrusion_,
                 hint: &[DVec3]) -> Result<Surface, Error> {
    if hint.is_empty() {
        return Err(Error::UnknownSurfaceType);
    }
    let prof = profile(s, e.swept_curve, hint)?;
    let v = s.entity(e.extrusion_axis).ok_or(Error::UnknownSurfaceType)?;
    let d = direction(s, v.orientation).normalize();

    // The surface is infinite along `d`.  Bound it so it covers every point
    // of the face: offset = p·d - C(u)·d, and C(u) stays inside the hull of
    // its control points.
    let along = |p: &DVec3| p.dot(&d);
    let (mut c_min, mut c_max) = (f64::INFINITY, f64::NEG_INFINITY);
    for p in &prof.points {
        let a = along(&euclidean(p));
        c_min = c_min.min(a);
        c_max = c_max.max(a);
    }
    let (mut p_min, mut p_max) = (f64::INFINITY, f64::NEG_INFINITY);
    for p in hint {
        p_min = p_min.min(along(p));
        p_max = p_max.max(along(p));
    }
    let pad = 1e-3 + 0.01 * (p_max - p_min).abs();
    let (v_min, v_max) = (p_min - c_max - pad, p_max - c_min + pad);

    let control_points = prof.points.iter().map(|p| {
        let w = p.w;
        vec![p + DVec4::new(d.x, d.y, d.z, 0.0) * (v_min * w),
             p + DVec4::new(d.x, d.y, d.z, 0.0) * (v_max * w)]
    }).collect();
    let surf = NURBSSurface::new(
        true, true,
        prof.knots,
        KnotVector::from_multiplicities(1, &[v_min, v_max], &[2, 2]),
        control_points,
    );
    Ok(Surface::NURBS(SampledSurface::new(surf), None))
}

/// `SURFACE_OF_REVOLUTION`: the profile swept a full turn around an axis.
pub fn revolution(s: &StepFile, r: &ap214::SurfaceOfRevolution_,
                  hint: &[DVec3]) -> Result<Surface, Error> {
    let prof = profile(s, r.swept_curve, hint)?;
    let ax = s.entity(r.axis_position).ok_or(Error::UnknownSurfaceType)?;
    let origin = cartesian_point(s, ax.location);
    let axis = ax.axis
        .map(|a| direction(s, a))
        .unwrap_or_else(|| DVec3::new(0.0, 0.0, 1.0));
    revolve(prof, origin, axis)
}

/// `TOROIDAL_SURFACE` as an exact NURBS: a circle of `minor` radius,
/// `major` away from the axis, revolved around it.  Same (u, v) layout as the
/// STEP definition (u around the axis, v around the tube).
pub fn torus(location: DVec3, axis: DVec3, ref_direction: DVec3,
             major: f64, minor: f64) -> Result<Surface, Error> {
    if axis.norm() < 1e-12 || ref_direction.norm() < 1e-12 {
        return Err(Error::DegenerateSurface);
    }
    let (axis, x) = (axis.normalize(), ref_direction.normalize());
    let centre = location + x * major;
    let tube = Profile {
        knots: circle_knots(),
        points: CIRCLE.iter()
            .map(|&(cx, cy, w)| homogeneous(centre + x * (minor * cx) + axis * (minor * cy), w))
            .collect(),
    };
    revolve(tube, location, axis)
}

/// Sweeps a profile one full turn around an axis.  Following STEP, u is the
/// angle around the axis and v runs along the profile, which keeps the
/// surface normal on the side STEP expects.
fn revolve(prof: Profile, origin: DVec3, axis: DVec3) -> Result<Surface, Error> {
    if axis.norm() < 1e-12 {
        return Err(Error::DegenerateSurface);
    }
    let axis = axis.normalize();

    // Each profile control point sweeps out one circle of control points
    let rings: Vec<Vec<DVec4>> = prof.points.iter().map(|p| {
        let pos = euclidean(p);
        let centre = origin + axis * (pos - origin).dot(&axis);
        let x = pos - centre;
        let y = axis.cross(&x);
        CIRCLE.iter()
            .map(|&(cx, cy, w)| homogeneous(centre + x * cx + y * cy, w * p.w))
            .collect()
    }).collect();
    // control_points[u][v]: u indexes the circle, v the profile
    let control_points = (0..CIRCLE.len())
        .map(|k| rings.iter().map(|ring| ring[k]).collect())
        .collect();
    let surf = NURBSSurface::new(
        false, true,
        circle_knots(),
        prof.knots,
        control_points,
    );
    Ok(Surface::NURBS(SampledSurface::new(surf), None))
}
