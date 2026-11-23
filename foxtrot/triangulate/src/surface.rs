use std::f64::{EPSILON, consts::PI};
use std::ops::Range;

use nalgebra_glm as glm;
use glm::{DVec2, DVec3, DVec4, DMat4};

use nurbs::{AbstractSurface, NDBSplineSurface, SampledSurface};
use crate::{Error, mesh::Vertex};

/// Largest angle (radians) that one mesh segment may turn through, along a
/// boundary curve or across a curved face.  This is the single density knob:
/// 12 degrees keeps the chord within ~0.55% of the local radius, and since
/// vertex normals are exact, shading stays smooth; only silhouettes facet.
pub const MAX_SEGMENT_ANGLE: f64 = 12.0 * std::f64::consts::PI / 180.0;

/// Upper bound on interior grid rows per direction
const MAX_STEINER_ROWS: usize = 64;

/// Interior rows along the straight direction of a patch that is curved only
/// one way
const STRAIGHT_STEINER_ROWS: usize = 4;

// Represents a surface in 3D space, with a function to project a 3D point
// on the surface down to a 2D space.
#[derive(Debug, Clone)]
pub enum Surface {
    Cylinder {
        location: DVec3,
        axis: DVec3,
        mat: DMat4,
        mat_i: DMat4,
        radius: f64,
        z_min: f64,
        z_max: f64,
    },
    Plane {
        normal: DVec3,
        mat_i: DMat4,
    },
    Cone {
        mat: DMat4,
        mat_i: DMat4,
        angle: f64,
    },
    BSpline(SampledSurface<3>, Option<Annulus>),
    NURBS(SampledSurface<4>, Option<Annulus>),
    Sphere {
        location: DVec3,
        mat: DMat4,     // uv to world
        mat_i: DMat4,   // world to uv
        radius: f64,
    },
    Torus {
        axis: DVec3,
        location: DVec3,
        mat: DMat4,
        mat_i: DMat4,
        major_radius: f64,
        minor_radius: f64,
    },
}

/// Maps a face that wraps all the way around a closed spline surface (e.g. a
/// fillet band on a torus, bounded by two full circles and no seam edge)
/// onto a flat ring: the wrapping parameter becomes the angle and the other
/// parameter the radius.  A plain (u, v) rectangle can't hold such a face,
/// because each boundary circle would become an open line.
#[derive(Debug, Clone, Copy)]
pub struct Annulus {
    /// True if the face wraps around in U (else in V)
    around_u: bool,
    start: f64,
    period: f64,
    /// Range of the other parameter, mapped onto radius 1..2
    r0: f64,
    r1: f64,
}

impl Annulus {
    fn radial_span(&self) -> f64 {
        if self.r1 - self.r0 > 1e-12 { self.r1 - self.r0 } else { 1.0 }
    }

    /// Surface parameters to a point on the ring
    fn to_plane(&self, uv: DVec2) -> (f64, f64) {
        let (a, r) = if self.around_u { (uv.x, uv.y) } else { (uv.y, uv.x) };
        let theta = 2.0 * PI * (a - self.start) / self.period;
        let t = (r - self.r0) / self.radial_span();
        // The radius runs against V when wrapping in U (and with U when
        // wrapping in V) so triangles keep the surface's orientation.
        let rho = if self.around_u { 2.0 - t } else { 1.0 + t };
        (rho * theta.cos(), rho * theta.sin())
    }

    /// Grid of surface parameters covering the ring
    fn grid(&self, num_a: usize, num_r: usize) -> Vec<DVec2> {
        let mut out = Vec::new();
        for i in 0..num_a {
            let a = self.start + self.period * (i as f64) / (num_a as f64);
            for j in 0..num_r {
                let r = self.r0 + self.radial_span() * (j as f64 + 1.0) / (num_r as f64 + 1.0);
                out.push(if self.around_u { DVec2::new(a, r) } else { DVec2::new(r, a) });
            }
        }
        out
    }
}

impl Surface {
    pub fn new_sphere(location: DVec3, radius: f64) -> Self {
        Surface::Sphere {
            // mat and mat_i are built in prepare()
            mat: DMat4::identity(),
            mat_i: DMat4::identity(),
            location, radius,
        }
    }
    pub fn new_cylinder(axis: DVec3, ref_direction: DVec3, location: DVec3, radius: f64)
        -> Result<Self, Error>
    {
        let mat = Self::make_rigid_transform(axis, ref_direction, location);
        let mat_i = mat.try_inverse().ok_or(Error::DegenerateSurface)?;
        Ok(Surface::Cylinder {
            mat,
            mat_i,
            axis, radius, location,
            z_min: 0.0,
            z_max: 0.0,
        })
    }

    pub fn new_torus(location: DVec3, axis: DVec3,
                     major_radius: f64, minor_radius: f64) -> Self
    {
        Surface::Torus {
            // mat and mat_i are built in prepare()
            mat: DMat4::identity(),
            mat_i: DMat4::identity(),
            location, axis, major_radius, minor_radius
        }
    }

    pub fn new_plane(axis: DVec3, ref_direction: DVec3, location: DVec3)
        -> Result<Self, Error>
    {
        let mat_i = Self::make_rigid_transform(axis, ref_direction, location)
            .try_inverse()
            .ok_or(Error::DegenerateSurface)?;
        Ok(Surface::Plane { mat_i, normal: axis })
    }

    pub fn new_cone(axis: DVec3, ref_direction: DVec3, location: DVec3, angle: f64)
        -> Result<Self, Error>
    {
        let mat = Self::make_rigid_transform(axis, ref_direction, location);
        let mat_i = mat.try_inverse().ok_or(Error::DegenerateSurface)?;
        Ok(Surface::Cone { mat, mat_i, angle })
    }

    pub fn make_affine_transform(z_world: DVec3, x_world: DVec3, y_world: DVec3, origin_world: DVec3) -> DMat4 {
        let mut mat = DMat4::identity();
        mat.set_column(0, &glm::vec3_to_vec4(&x_world));
        mat.set_column(1, &glm::vec3_to_vec4(&y_world));
        mat.set_column(2, &glm::vec3_to_vec4(&z_world));
        mat.set_column(3, &glm::vec3_to_vec4(&origin_world));
        *mat.get_mut((3, 3)).unwrap() =  1.0;
        mat
    }

    fn make_rigid_transform(z_world: DVec3, x_world: DVec3, origin_world: DVec3) -> DMat4 {
        let mut mat = DMat4::identity();
        mat.set_column(0, &glm::vec3_to_vec4(&x_world));
        mat.set_column(1, &glm::vec3_to_vec4(&z_world.cross(&x_world)));
        mat.set_column(2, &glm::vec3_to_vec4(&z_world));
        mat.set_column(3, &glm::vec3_to_vec4(&origin_world));
        *mat.get_mut((3, 3)).unwrap() =  1.0;
        mat
    }

    fn surf_lower<const N: usize>(p: DVec3, surf: &SampledSurface<N>) -> Result<DVec2, Error>
        where NDBSplineSurface<N>: AbstractSurface
    {
        surf.uv_from_point(p).ok_or(Error::CouldNotLower)
    }

    /// Lowers a 3D point on a specific surface into a 2D space defined by
    /// the surface type.  This should only be called from `lower_verts`,
    /// to ensure that `prepare` is called first.
    fn lower(&self, p: DVec3) -> Result<DVec2, Error> {
        let p_ = DVec4::new(p.x, p.y, p.z, 1.0);
        match self {
            Surface::Plane { mat_i, .. } => {
                Ok(glm::vec4_to_vec2(&(mat_i * p_)))
            },
            Surface::Cone { mat_i, .. } => {
                let xy = glm::vec4_to_vec2(&(mat_i * p_));
                Ok(DVec2::new(-xy.x, xy.y))
            },

            Surface::Cylinder { mat_i, z_min, z_max, .. } => {
                let p = mat_i * p_;
                // We convert the Z coordinates to either add or subtract from
                // the radius, so that we maintain the right topology (instead
                // of doing something like theta-z coordinates, which wrap
                // around awkwardly).

                // Scale from radius=1 to radius=0.5 based on Z
                let z = (p.z - z_min) / (z_max - z_min);
                let scale = 1.0 / (1.0 + z);
                Ok(DVec2::new(p.x * scale, p.y * scale))
            },
            Surface::Torus { mat_i, major_radius, minor_radius, .. } => {
                let p = mat_i * p_;
                /*
                         ^ Y
                         |
                    /---------\
                   /     |     \
                   |   -----   |
                   |   | O |- -|- - >Z
                   |   -----   |
                   \           /
                    \---------/

                    (X axis points into the screen)
                */
                let major_angle = p.y.atan2(p.z);

                // Rotate the point so that it's got Y = 0, so we can calculate
                // the minor angle
                let z = DVec3::new(0.0, major_angle.sin(), major_angle.cos());
                let new_mat = Self::make_rigid_transform(
                    z, DVec3::new(1.0, 0.0, 0.0), z * *major_radius);
                let new_mat_i = new_mat.try_inverse()
                    .expect("Could not invert");
                let new_p = new_mat_i * DVec4::new(p.x, p.y, p.z, 1.0);

                let minor_angle = new_p.x.atan2(new_p.z);

                // Construct nested circles with a scale based on the ratio
                // of radiuses (to make an _attempt_ to match 3D distance)
                let scale = 1.0 + (major_radius / minor_radius) *
                                  (major_angle + PI) / (2.0 * PI);

                let x = if *major_radius > 0.0 {
                    -minor_angle.cos()
                } else {
                    minor_angle.cos()
                };
                Ok(scale * DVec2::new(x, minor_angle.sin()))
            },
            Surface::BSpline(surf, _) => Self::surf_lower(p, surf),
            Surface::NURBS(surf, _) => Self::surf_lower(p, surf),
            Surface::Sphere { mat_i, radius, .. } => {
                // mat_i is constructed in prepare to be a reasonable basis
                let p = (mat_i * p_).xyz() / *radius;
                let r = p.yz().norm();

                // Angle from 0 to PI
                let angle = r.atan2(p.x);
                let yz = p.yz();
                Ok(if yz.norm() < EPSILON {
                    yz
                } else {
                    yz * angle / yz.norm()
                })
            },
        }
    }

    fn prepare(&mut self, verts: &[Vertex], loops: &[Range<usize>], same_sense: bool) {
        match self {
            Surface::Cylinder { mat_i, z_min, z_max, .. } => {
                *z_min = std::f64::INFINITY;
                *z_max = -std::f64::INFINITY;
                for v in verts {
                    let p = (*mat_i) * DVec4::new(v.pos.x, v.pos.y, v.pos.z, 1.0);
                    if p.z < *z_min {
                        *z_min = p.z;
                    }
                    if p.z > *z_max {
                        *z_max = p.z;
                    }
                }
            },
            Surface::Sphere { mat, mat_i, location, .. } => {
                // Centre the projection inside the face, so the far side of
                // the sphere (which maps to the edge of the projection disc)
                // is as far from the face as possible.  Only the boundary's
                // orientation says which side of a loop is the face (a small
                // circle can bound a cap or everything but the cap), so use
                // its circulation axis, flipped when the face runs against
                // the surface normal.
                let dirs: Vec<DVec3> = verts.iter()
                    .map(|v| (v.pos - *location).normalize())
                    .collect();
                let mean = dirs.iter().sum::<DVec3>() / dirs.len().max(1) as f64;
                let mut circulation = DVec3::zeros();
                for l in loops {
                    for i in l.clone() {
                        let j = if i + 1 == l.end { l.start } else { i + 1 };
                        circulation += dirs[i].cross(&dirs[j]);
                    }
                }
                let ref_direction = if circulation.norm() > 1e-9 {
                    let c = circulation.normalize();
                    if same_sense { c } else { -c }
                } else if mean.norm() > 1e-9 {
                    mean.normalize()
                } else {
                    dirs[0]
                };
                // Any direction perpendicular to it completes the frame
                let least = if ref_direction.x.abs() < 0.5 {
                    DVec3::new(1.0, 0.0, 0.0)
                } else {
                    DVec3::new(0.0, 1.0, 0.0)
                };
                let axis = ref_direction.cross(&least).normalize();

                *mat = Self::make_rigid_transform(
                        axis, ref_direction, *location);
                *mat_i = mat
                    .try_inverse()
                    .expect("Could not invert");
            },
            Surface::Torus { axis, mat, mat_i, location, .. } => {
                let mean_dir = verts.iter()
                    .map(|v| v.pos - *location)
                    .sum::<DVec3>()
                    .normalize();
                let mean_perp_dir = (mean_dir - *axis * mean_dir.dot(axis)).normalize();
                *mat = Self::make_rigid_transform(
                    mean_perp_dir, *axis, *location);
                *mat_i = mat
                    .try_inverse()
                    .expect("Could not invert");
            },
            _ => (),
        }
    }

    /// Lowers boundary vertices to 2D.  `loops` gives the index range of
    /// each closed boundary loop within `verts`.
    pub fn lower_verts(&mut self, verts: &mut [Vertex], loops: &[Range<usize>],
                       same_sense: bool)
        -> Result<Vec<(f64, f64)>, Error>
    {
        self.prepare(verts, loops, same_sense);
        let mut pts = Vec::with_capacity(verts.len());
        let mut prev: Option<DVec2> = None;
        for v in verts {
            // Project to the 2D subspace for triangulation.  Spline surfaces
            // march along the boundary, seeding each point from the last.
            let proj = match self {
                Surface::BSpline(s, _) => s.uv_from_point_near(v.pos, prev)
                    .ok_or(Error::CouldNotLower)?,
                Surface::NURBS(s, _) => s.uv_from_point_near(v.pos, prev)
                    .ok_or(Error::CouldNotLower)?,
                _ => self.lower(v.pos)?,
            };
            prev = Some(proj);
            // Update the surface normal
            v.norm = self.normal(v.pos, proj);
            pts.push((proj.x, proj.y));
        }
        if let Surface::Cone { .. } = self {
            Self::nudge_apex(&mut pts);
        }
        if !matches!(self, Surface::BSpline(..) | Surface::NURBS(..)) {
            return Ok(pts);
        }

        self.unwrap_seams(&mut pts, loops);
        if let Some(annulus) = self.find_annulus(&pts, loops) {
            for p in pts.iter_mut() {
                *p = annulus.to_plane(DVec2::new(p.0, p.1));
            }
            match self {
                Surface::BSpline(_, a) | Surface::NURBS(_, a) => *a = Some(annulus),
                _ => unreachable!(),
            }
            return Ok(pts);
        }

        // Calculate an aspect ratio based on the control points net, then use
        // it to transform projected points.  This means that positions in 2D
        // (UV) space are closer to positions in 3D space, so the
        // triangulation is better.
        let aspect_ratio = match self {
            Surface::NURBS(surf, _) => surf.surf.aspect_ratio(),
            Surface::BSpline(surf, _) => surf.surf.aspect_ratio(),
            _ => unreachable!(),
        };
        for p in pts.iter_mut() {
            p.1 *= aspect_ratio;
        }
        Ok(pts)
    }

    /// A cone's apex lowers to the origin.  When the face's two straight
    /// edges meet there at 180 degrees (a half cone), the apex sits exactly on
    /// a straight boundary line and triangulation rejects it, so move it a
    /// hair toward the inside of the face.
    fn nudge_apex(pts: &mut [(f64, f64)]) {
        let size = pts.iter().map(|p| p.0.hypot(p.1)).fold(0.0, f64::max);
        if size <= 0.0 {
            return;
        }
        let n = pts.len() as f64;
        let (cx, cy) = pts.iter().fold((0.0, 0.0), |(x, y), p| (x + p.0 / n, y + p.1 / n));
        let len = cx.hypot(cy);
        if len <= 0.0 {
            return;
        }
        let step = 1e-3 * size;
        for p in pts.iter_mut() {
            if p.0.hypot(p.1) < 1e-9 * size {
                *p = (cx / len * step, cy / len * step);
            }
        }
    }

    /// If a boundary loop winds all the way around a closed direction, the
    /// face needs the ring mapping instead of a (u, v) rectangle.
    fn find_annulus(&self, pts: &[(f64, f64)], loops: &[Range<usize>]) -> Option<Annulus> {
        let (pu, pv) = self.seam_periods();
        let turns = |period: f64, coord: fn(&(f64, f64)) -> f64| loops.iter().any(|l| {
            if l.len() < 2 { return false; }
            let (first, last) = (coord(&pts[l.start]), coord(&pts[l.end - 1]));
            let close = (first - last + period / 2.0).rem_euclid(period) - period / 2.0;
            ((last - first + close) / period).round() != 0.0
        });
        let (around_u, (start, period)) = if let Some(p) = pu.filter(|p| turns(p.1, |q| q.0)) {
            (true, p)
        } else if let Some(p) = pv.filter(|p| turns(p.1, |q| q.1)) {
            (false, p)
        } else {
            return None;
        };
        let radial = |q: &(f64, f64)| if around_u { q.1 } else { q.0 };
        let (mut r0, mut r1) = (f64::INFINITY, f64::NEG_INFINITY);
        for q in pts {
            r0 = r0.min(radial(q));
            r1 = r1.max(radial(q));
        }
        Some(Annulus { around_u, start, period, r0, r1 })
    }

    /// For a B-spline / NURBS surface that closes on itself (e.g. a full
    /// tube), the (start, period) of each closed parameter direction.
    fn seam_periods(&self) -> (Option<(f64, f64)>, Option<(f64, f64)>) {
        fn periods<const N: usize>(s: &SampledSurface<N>)
            -> (Option<(f64, f64)>, Option<(f64, f64)>)
            where NDBSplineSurface<N>: AbstractSurface
        {
            let f = &s.surf;
            (s.closed_u.then(|| (f.min_u(), f.max_u() - f.min_u())),
             s.closed_v.then(|| (f.min_v(), f.max_v() - f.min_v())))
        }
        match self {
            Surface::BSpline(s, _) => periods(s),
            Surface::NURBS(s, _) => periods(s),
            _ => (None, None),
        }
    }

    /// Points on the seam of a closed surface can lower to either end of the
    /// parameter range, which tears the boundary loop apart in 2D.  Walk each
    /// loop and shift every point by whole periods so it stays next to the
    /// previous one, then shift whole loops so holes sit inside their face.
    fn unwrap_seams(&self, pts: &mut [(f64, f64)], loops: &[Range<usize>]) {
        let (pu, pv) = self.seam_periods();
        // Value of `x` shifted by whole periods to lie closest to `target`
        let near = |target: f64, x: f64, period: f64| {
            x - ((x - target) / period).round() * period
        };
        let get = |q: &(f64, f64), axis: usize| if axis == 0 { q.0 } else { q.1 };
        let set = |q: &mut (f64, f64), axis: usize, x: f64| {
            if axis == 0 { q.0 = x } else { q.1 = x }
        };
        let mean = |pts: &[(f64, f64)], axis: usize|
            pts.iter().map(|q| get(q, axis)).sum::<f64>() / pts.len() as f64;

        for (axis, period) in [(0, pu), (1, pv)] {
            let Some((_, period)) = period else { continue };
            for (i, l) in loops.iter().enumerate() {
                for j in (l.start + 1)..l.end {
                    let x = near(get(&pts[j - 1], axis), get(&pts[j], axis), period);
                    set(&mut pts[j], axis, x);
                }
                // Move inner loops by whole periods to sit by the first loop
                if i > 0 && !l.is_empty() && !loops[0].is_empty() {
                    let m = mean(&pts[l.clone()], axis);
                    let shift = near(mean(&pts[loops[0].clone()], axis), m, period) - m;
                    for q in &mut pts[l.clone()] {
                        set(q, axis, get(q, axis) + shift);
                    }
                }
            }
        }
    }

    /// Bring an unwrapped (u, v) back into the surface's parameter domain
    fn wrap_uv(&self, uv: DVec2) -> DVec2 {
        let (pu, pv) = self.seam_periods();
        let wrap = |x: f64, p: Option<(f64, f64)>| match p {
            Some((start, period)) if period > 0.0 =>
                start + (x - start).rem_euclid(period),
            _ => x,
        };
        DVec2::new(wrap(uv.x, pu), wrap(uv.y, pv))
    }

    pub fn raise(&self, uv: DVec2) -> Option<DVec3> {
        match self {
            Surface::Sphere { mat, radius, .. } => {
                let angle = uv.norm();
                if angle > PI {
                    return None;
                }
                let x = angle.cos();

                // Calculate pre-transformed position
                let pos = (*radius) * if uv.norm() < EPSILON {
                    DVec3::new(x, 0.0, 0.0)
                } else {
                    let yz = uv.normalize() * angle.sin();
                    DVec3::new(x, yz.x, yz.y)
                };
                // Transform into world space
                let pos = (mat * DVec4::new(pos.x, pos.y, pos.z, 1.0))
                    .xyz();
                Some(pos)
            },
            Surface::BSpline(s, _) => Some(s.surf.point(self.wrap_uv(uv))),
            Surface::NURBS(s, _) => Some(s.surf.point(self.wrap_uv(uv))),
            Surface::Torus { mat, minor_radius, major_radius, .. } => {
                let mut uv = uv;
                if *major_radius > 0.0 {
                    uv.x *= -1.0;
                }
                let minor_angle = uv.y.atan2(uv.x);
                let major_angle = (uv.norm() - 1.0) /
                                  (major_radius / minor_radius) * 2.0 * PI - PI;
                let new_p = DVec3::new(minor_angle.sin(), 0.0, minor_angle.cos()) * *minor_radius;

                let z = DVec3::new(0.0, major_angle.sin(), major_angle.cos());
                let new_mat = Self::make_rigid_transform(
                    z, DVec3::new(1.0, 0.0, 0.0), z * *major_radius);
                let p = new_mat * DVec4::new(new_p.x, new_p.y, new_p.z, 1.0);

                Some((mat * p).xyz())
            },
            _ => unimplemented!(),
        }
    }

    fn bbox(pts: &[(f64, f64)]) -> (f64, f64, f64, f64) {
        let (mut xmin, mut xmax) = (std::f64::INFINITY, -std::f64::INFINITY);
        let (mut ymin, mut ymax) = (std::f64::INFINITY, -std::f64::INFINITY);
        for (px, py) in pts {
            xmin = px.min(xmin);
            ymin = py.min(ymin);
            xmax = px.max(xmax);
            ymax = py.max(ymax);
        }
        (xmin, xmax, ymin, ymax)
    }

    /// Grid rows needed to follow a turn of `angle` radians
    fn rows_for(angle: f64) -> usize {
        ((angle / MAX_SEGMENT_ANGLE).ceil() as usize).min(MAX_STEINER_ROWS)
    }

    /// How far the surface normal turns across the parameter box
    /// [u0, u1] x [v0, v1], along U and along V (radians, worst row/column).
    fn turning(&self, u0: f64, u1: f64, v0: f64, v1: f64) -> (f64, f64) {
        const K: usize = 6;
        let at = |i: usize, j: usize| {
            let uv = DVec2::new(u0 + (u1 - u0) * i as f64 / K as f64,
                                v0 + (v1 - v0) * j as f64 / K as f64);
            let n = self.normal(DVec3::zeros(), uv);
            if n.iter().all(|x| x.is_finite()) { Some(n) } else { None }
        };
        let angle = |a: Option<DVec3>, b: Option<DVec3>| match (a, b) {
            (Some(a), Some(b)) => a.dot(&b).clamp(-1.0, 1.0).acos(),
            _ => 0.0,
        };
        let (mut tu, mut tv) = (0.0f64, 0.0f64);
        for j in 0..=K {
            tu = tu.max((0..K).map(|i| angle(at(i, j), at(i + 1, j))).sum());
        }
        for i in 0..=K {
            tv = tv.max((0..K).map(|j| angle(at(i, j), at(i, j + 1))).sum());
        }
        (tu, tv)
    }

    /// Adds interior points so the mesh follows the surface's curvature.
    /// The number of rows in each direction is set by how much the surface
    /// turns across the face (see `MAX_SEGMENT_ANGLE`), so flat faces and
    /// small fillets stay light and strongly curved free-form faces get dense.
    pub fn add_steiner_points(&self, pts: &mut Vec<(f64, f64)>,
                                     verts: &mut Vec<Vertex>)
    {
        if pts.is_empty() {
            return;
        }
        let push = |pts: &mut Vec<(f64, f64)>, verts: &mut Vec<Vertex>,
                    plane: (f64, f64), uv: DVec2| {
            if let Some(pos) = self.raise(uv) {
                pts.push(plane);
                verts.push(Vertex {
                    pos,
                    norm: self.normal(pos, uv),
                    color: DVec3::new(0.0, 0.0, 0.0),
                });
            }
        };
        let (xmin, xmax, ymin, ymax) = Self::bbox(&pts);
        let grid = |n: usize, lo: f64, hi: f64, k: usize|
            lo + (hi - lo) * (k as f64 + 1.0) / (n as f64 + 1.0);

        match self {
            // Azimuthal projection: distance from the centre is the angle
            Surface::Sphere { .. } => {
                let n = Self::rows_for((xmax - xmin).max(ymax - ymin)).max(2);
                for x in 0..n {
                    for y in 0..n {
                        let uv = DVec2::new(grid(n, xmin, xmax, x), grid(n, ymin, ymax, y));
                        push(pts, verts, (uv.x, uv.y), uv);
                    }
                }
            },
            Surface::Torus { .. } => {
                for x in 0..32 {
                    for y in 0..32 {
                        let uv = DVec2::new(grid(32, xmin, xmax, x), grid(32, ymin, ymax, y));
                        push(pts, verts, (uv.x, uv.y), uv);
                    }
                }
            },
            Surface::BSpline(_, annulus) | Surface::NURBS(_, annulus) => {
                // Parameter box covered by the face
                let aspect_ratio = match self {
                    Surface::NURBS(surf, _) => surf.surf.aspect_ratio(),
                    Surface::BSpline(surf, _) => surf.surf.aspect_ratio(),
                    _ => unreachable!(),
                };
                let (u0, u1, v0, v1) = match annulus {
                    Some(a) if a.around_u => (a.start, a.start + a.period, a.r0, a.r1),
                    Some(a) => (a.r0, a.r1, a.start, a.start + a.period),
                    None => (xmin, xmax, ymin / aspect_ratio, ymax / aspect_ratio),
                };
                let (tu, tv) = self.turning(u0, u1, v0, v1);
                let (num_u, num_v) = (Self::rows_for(tu), Self::rows_for(tv));
                // A patch straight in one direction but curved in the other
                // (a hole wall, a rolled edge) still needs a few rows along
                // the straight direction, so triangles follow the curve.
                let (num_u, num_v) = match (num_u, num_v) {
                    (0, n) if n > 0 => (STRAIGHT_STEINER_ROWS, n),
                    (n, 0) if n > 0 => (n, STRAIGHT_STEINER_ROWS),
                    other => other,
                };
                if let Some(a) = annulus {
                    // Ring mapping: grid in parameters, mapped the same way
                    let (num_a, num_r) = if a.around_u { (num_u, num_v) } else { (num_v, num_u) };
                    for uv in a.grid(num_a.max(4), num_r) {
                        push(pts, verts, a.to_plane(uv), uv);
                    }
                } else {
                    // Keep grid cells at most 2:1 in the lowered plane (which
                    // approximates 3D distance); long thin cells make the
                    // triangulation cut across concave regions.
                    let (eu, ev) = (xmax - xmin, ymax - ymin);
                    let (num_u, num_v) = if num_u > 0 && num_v > 0 && eu > 0.0 && ev > 0.0 {
                        let cell = (eu / (num_u + 1) as f64).min(ev / (num_v + 1) as f64);
                        let fit = |extent: f64, n: usize|
                            n.max(((extent / (2.0 * cell)).ceil() as usize).saturating_sub(1))
                             .min(MAX_STEINER_ROWS);
                        (fit(eu, num_u), fit(ev, num_v))
                    } else {
                        (num_u, num_v)
                    };
                    for x in 0..num_u {
                        let u = grid(num_u, xmin, xmax, x);
                        for y in 0..num_v {
                            let v = grid(num_v, ymin, ymax, y);
                            push(pts, verts, (u, v), DVec2::new(u, v / aspect_ratio));
                        }
                    }
                }
            },
            _ => (),
        }
    }

    fn surf_normal<const N: usize>(uv: DVec2, surf: &SampledSurface<N>) -> DVec3
        where NDBSplineSurface<N>: AbstractSurface
    {
        // Calculate first order derivs, then cross them to get normal
        let derivs = surf.surf.derivs::<1>(uv);
        let n = derivs[1][0].cross(&derivs[0][1]);
        n.normalize()
    }

    // Calculate the surface normal, using either the 3D or 2D position
    pub fn normal(&self, p: DVec3, uv: DVec2) -> DVec3 {
        match self {
            Surface::Plane { normal, .. } => *normal,
            Surface::Cone { mat, mat_i, angle, .. } => {
                // Project into CONE SPACE
                let pos = mat_i * DVec4::new(p.x, p.y, p.z, 1.0);
                let xy = if pos.xy().norm() > std::f64::EPSILON {
                    pos.xy().normalize()
                } else {
                    return DVec3::zeros();
                };
                let normal = DVec4::new(xy.x * angle.cos(),
                                        xy.y * angle.cos(), -angle.sin(), 0.0);
                // Deproject back into world space
                (mat * normal).xyz()
            }
            Surface::Sphere { location, .. } => (p - location).normalize(),
            Surface::Cylinder { mat, mat_i, .. } => {
                // Project the point onto the axis
                let proj = mat_i * DVec4::new(p.x, p.y, p.z, 1.0);

                // Then the normal is just pointing along that direction
                // (same hack as below)
                let norm = DVec3::new(proj.x, proj.y, 0.0).normalize();
                (mat * norm.to_homogeneous()).xyz()
            },
            Surface::BSpline(surf, _) => Self::surf_normal(self.wrap_uv(uv), surf),
            Surface::NURBS(surf, _) => Self::surf_normal(self.wrap_uv(uv), surf),
            Surface::Torus { mat, mat_i, major_radius, .. } => {
                let p = (*mat_i * DVec4::new(p.x, p.y, p.z, 1.0)).xyz();
                let major_angle = p.y.atan2(p.z);

                let z = DVec3::new(0.0, major_angle.sin(), major_angle.cos()) * *major_radius;
                let norm = (p - z).normalize();

                (mat * norm.to_homogeneous()).xyz()
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn degenerate_frames_return_error_instead_of_panicking() {
        let zero = DVec3::new(0.0, 0.0, 0.0);
        let x = DVec3::new(1.0, 0.0, 0.0);
        assert_eq!(Surface::new_cylinder(zero, x, zero, 1.0).err(), Some(Error::DegenerateSurface));
        assert_eq!(Surface::new_plane(zero, zero, zero).err(), Some(Error::DegenerateSurface));
        assert_eq!(Surface::new_cone(zero, x, zero, 0.5).err(), Some(Error::DegenerateSurface));
    }
}
