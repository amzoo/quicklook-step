use std::collections::{HashMap, HashSet};
use std::convert::TryInto;

use nalgebra_glm as glm;
use glm::{DVec3, DVec4, DMat4, U32Vec3};
use log::{info, warn, error};

#[cfg(feature = "rayon")]
use rayon::prelude::*;

use step::{
    ap214, ap214::*, step_file::{FromEntity, StepFile}, id::Id, ap214::Entity,
};
use crate::{
    Error,
    curve::{Curve, Sense},
    mesh, mesh::{Mesh, Triangle},
    stats::Stats,
    surface::Surface
};
use nurbs::{BSplineSurface, SampledCurve, SampledSurface, NURBSSurface, KnotVector};

const SAVE_DEBUG_SVGS: bool = false;
const SAVE_PANIC_SVGS: bool = false;

/// `TransformStack` is a mapping of representations to transformed children.
type TransformStack<'a> =
    HashMap<Representation<'a>, Vec<(Representation<'a>, DMat4)>>;
fn build_transform_stack<'a>(s: &'a StepFile, flip: bool) -> TransformStack<'a> {
    // Store a map of parent -> (child, transform)
    let mut transform_stack: HashMap<_, Vec<_>> = HashMap::new();
    for r in s.0.iter()
        .filter_map(|e|
            RepresentationRelationshipWithTransformation_::try_from_entity(e))
    {
        let (a, b) = if flip {
            (r.rep_2, r.rep_1)
        } else {
            (r.rep_1, r.rep_2)
        };
        let mut mat = item_defined_transformation(s, r.transformation_operator.cast());
        if flip {
            mat = mat.try_inverse().expect("Could not invert transform matrix");
        }

        transform_stack.entry(b)
            .or_default()
            .push((a, mat));
    }
    transform_stack
}

fn transform_stack_roots<'a>(transform_stack: &TransformStack<'a>) -> Vec<Representation<'a>> {
    let children: HashSet<_> = transform_stack
        .values()
        .flat_map(|v| v.iter())
        .map(|v| v.0)
        .collect();
    transform_stack
        .keys()
        .filter(|k| !children.contains(k))
        .copied()
        .collect()
}

pub fn triangulate(s: &StepFile) -> (Mesh, Stats) {
    let styled_items: Vec<_> = s.0.iter()
        .filter_map(|e| MechanicalDesignGeometricPresentationRepresentation_::try_from_entity(e))
        .flat_map(|m| m.items.iter())
        .filter_map(|item| s.entity(item.cast::<StyledItem_>()))
        .collect();
    let brep_colors: HashMap<_, DVec3> = styled_items.iter()
        .filter_map(|styled|
            if styled.styles.len() != 1 {
                None
            } else {
                presentation_style_color(s, styled.styles[0])
                    .map(|c| (styled.item, c))
            })
        .collect();

    // Construction / reference geometry (CATIA's "supplemental geometry",
    // datum planes, ...) is stored in a CONSTRUCTIVE_GEOMETRY_REPRESENTATION.
    // CAD viewers hide it, so we do too.
    let constructive: HashSet<usize> = s.0.iter()
        .filter_map(|e| match e {
            Entity::ConstructiveGeometryRepresentation(c) => Some(c.items.iter()),
            _ => None,
        })
        .flatten()
        .map(|i| i.0)
        .collect();

    // Store a map of parent -> (child, transform)
    let mut transform_stack = build_transform_stack(s, false);
    let mut roots = transform_stack_roots(&transform_stack);
    // The transformation graph isn't directional (because STEP is a Good File
    // Format), so if it's got more than one root, assume it's backwards.  We
    // are assuming that directions in the graph are consistent within the file,
    // until we find a counterexample.
    if roots.len() > 1 {
        info!("Flipping transform stack");
        transform_stack = build_transform_stack(s, true);
        roots = transform_stack_roots(&transform_stack);
    }
    let mut todo: Vec<_> = roots.into_iter()
        .map(|v| (v, DMat4::identity()))
        .collect();
    if todo.len() > 1 {
        warn!("Transformation stack has more than one root!");
    }

    // Store a map of ShapeRepresentationRelationships, which some models
    // use to map from axes to specific instances
    let mut shape_rep_relationship: HashMap<Id<_>, Vec<Id<_>>> = HashMap::new();
    for (r1, r2) in s.0.iter()
        .filter_map(|e| ShapeRepresentationRelationship_::try_from_entity(e))
        .map(|e| (e.rep_1, e.rep_2))
    {
        shape_rep_relationship.entry(r1).or_default().push(r2);
    }

    let mut to_mesh: HashMap<Id<_>, Vec<_>> = HashMap::new();
    while let Some((id, mat)) = todo.pop() {
        for child in shape_rep_relationship.get(&id).unwrap_or(&vec![]) {
            todo.push((*child, mat));
        }
        if let Some(children) = transform_stack.get(&id) {
            for (child, next_mat) in children {
                todo.push((*child, mat * next_mat));
            }
        } else {
            // Bind this transform to the RepresentationItem, which is
            // either a ManifoldSolidBrep or a ShellBasedSurfaceModel
            let items = match &s[id] {
                Entity::AdvancedBrepShapeRepresentation(b) => &b.items,
                Entity::ShapeRepresentation(b) => &b.items,
                Entity::ManifoldSurfaceShapeRepresentation(b) => &b.items,
                e => panic!("Could not get shape from {:?}", e),
            };

            for m in items.iter().filter(|m| !constructive.contains(&m.0)) {
                match &s[*m] {
                    Entity::ManifoldSolidBrep(_)
                    | Entity::BrepWithVoids(_)
                    | Entity::ShellBasedSurfaceModel(_) =>
                        to_mesh.entry(*m).or_default().push(mat),
                    Entity::Axis2Placement3d(_) => (),
                    e => warn!("Skipping {:?}", e),
                }
            }
        }
    }
    // If there are items in breps that aren't attached to a transformation
    // chain, then draw them individually (with an identity matrix)
    if to_mesh.is_empty() {
        s.0.iter()
            .enumerate()
            .filter(|(i, e)| !constructive.contains(i) &&
                match e {
                    Entity::ManifoldSolidBrep(_)
                    | Entity::BrepWithVoids(_)
                    | Entity::ShellBasedSurfaceModel(_) => true,
                    _ => false,
                }
            )
            .map(|(i, _e)| Id::new(i))
            .for_each(|i| to_mesh.entry(i).or_default().push(DMat4::identity()));
    }

    let (to_mesh_iter, empty) = {
        #[cfg(feature = "rayon")]
        { (to_mesh.par_iter(), || (Mesh::default(), Stats::default())) }
        #[cfg(not(feature = "rayon"))]
        { (to_mesh.iter(), (Mesh::default(), Stats::default())) }
    };
    let mesh_fold = to_mesh_iter
        .fold(
            // Empty constructor
            empty,

            // Fold operation
            |(mut mesh, mut stats), (id, mats)| {
                let v_start = mesh.verts.len();
                let t_start = mesh.triangles.len();
                match &s[*id] {
                    Entity::ManifoldSolidBrep(b) =>
                        closed_shell(s, b.outer, &mut mesh, &mut stats),
                    Entity::ShellBasedSurfaceModel(b) =>
                        for v in &b.sbsm_boundary {
                            shell(s, *v, &mut mesh, &mut stats);
                        },
                    Entity::BrepWithVoids(b) =>
                        // TODO: handle voids
                        closed_shell(s, b.outer, &mut mesh, &mut stats),
                    _ => {
                        warn!("Skipping {:?} (not a known solid)", s[*id]);
                        return (mesh, stats);
                    },
                };

                // Pick out a color from the color map and apply it to each
                // newly-created vertex
                let color = brep_colors.get(id)
                    .map(|c| *c)
                    .unwrap_or(DVec3::new(0.5, 0.5, 0.5));

                // Build copies of the mesh by copying and applying transforms
                let v_end = mesh.verts.len();
                let t_end = mesh.triangles.len();
                for mat in &mats[1..] {
                    for v in v_start..v_end {
                        let p = mesh.verts[v].pos;
                        let p_h = DVec4::new(p.x, p.y, p.z, 1.0);
                        let pos = (mat * p_h).xyz();

                        let n = mesh.verts[v].norm;
                        let norm = (mat * glm::vec3_to_vec4(&n)).xyz();

                        mesh.verts.push(mesh::Vertex { pos, norm, color });
                    }
                    let offset = mesh.verts.len() - v_end;
                    for t in t_start..t_end {
                        let mut tri = mesh.triangles[t];
                        tri.verts.add_scalar_mut(offset as u32);
                        mesh.triangles.push(tri);
                    }
                }

                // Now that we've built all of the other copies of the mesh,
                // re-use the original mesh and apply the first transform
                let mat = mats[0];
                for v in v_start..v_end {
                    let p = mesh.verts[v].pos;
                    let p_h = DVec4::new(p.x, p.y, p.z, 1.0);
                    mesh.verts[v].pos = (mat * p_h).xyz();

                    let n = mesh.verts[v].norm;
                    mesh.verts[v].norm = (mat * glm::vec3_to_vec4(&n)).xyz();

                    mesh.verts[v].color = color;
                }
                (mesh, stats)
            });

    let (mesh, stats) = {
        #[cfg(feature = "rayon")]
        { mesh_fold.reduce(empty,
                |a, b| (Mesh::combine(a.0, b.0), Stats::combine(a.1, b.1))) }
        #[cfg(not(feature = "rayon"))]
        {
            mesh_fold
        }
    };

    info!("num_shells: {}", stats.num_shells);
    info!("num_faces: {}", stats.num_faces);
    info!("num_errors: {}", stats.num_errors);
    info!("num_panics: {}", stats.num_panics);
    if stats.num_errors + stats.num_panics > 0 {
        warn!("{} of {} faces failed to triangulate",
              stats.num_errors + stats.num_panics, stats.num_faces);
    }
    (mesh, stats)
}

fn item_defined_transformation(s: &StepFile, t: Id<ItemDefinedTransformation_>) -> DMat4 {
    let i = s.entity(t).expect("Could not get ItemDefinedTransform");

    let (location, axis, ref_direction) = axis2_placement_3d(s,
        i.transform_item_1.cast());
    let t1 = Surface::make_affine_transform(axis,
        ref_direction,
        axis.cross(&ref_direction),
        location);

    let (location, axis, ref_direction) = axis2_placement_3d(s,
        i.transform_item_2.cast());
    let t2 = Surface::make_affine_transform(axis,
        ref_direction,
        axis.cross(&ref_direction),
        location);

    t2 * t1.try_inverse().expect("Could not invert transform matrix")
}

fn presentation_style_color(s: &StepFile, p: PresentationStyleAssignment)
    -> Option<DVec3>
{
    // AAAAAHHHHH
    s.entity(p)
        .and_then(|p: &PresentationStyleAssignment_| {
                let mut surf = p.styles.iter().filter_map(|y| {
                    // This is an ambiguous parse, so we hard-code the first
                    // Entity item in the enum
                    use PresentationStyleSelect::PreDefinedPresentationStyle;
                    if let PreDefinedPresentationStyle(u) = y {
                        s.entity(u.cast::<SurfaceStyleUsage_>())
                    } else {
                        None
                    }});
                let out = surf.next();
                out
            })
        .and_then(|surf: &SurfaceStyleUsage_|
            s.entity(surf.style.cast::<SurfaceSideStyle_>()))
        .and_then(|surf: &SurfaceSideStyle_| if surf.styles.len() != 1 {
                None
            } else {
                s.entity(surf.styles[0].cast::<SurfaceStyleFillArea_>())
            })
        .map(|surf: &SurfaceStyleFillArea_|
            s.entity(surf.fill_area).expect("Could not get fill_area"))
        .and_then(|fill: &FillAreaStyle_| if fill.fill_styles.len() != 1 {
                None
            } else {
                s.entity(fill.fill_styles[0].cast::<FillAreaStyleColour_>())
            })
        .and_then(|f: &FillAreaStyleColour_|
            s.entity(f.fill_colour.cast::<ColourRgb_>()))
        .map(|c| DVec3::new(c.red, c.green, c.blue))
}

pub(crate) fn cartesian_point(s: &StepFile, a: Id<CartesianPoint_>) -> DVec3 {
    let p = s.entity(a).expect("Could not get cartesian point");
    DVec3::new(p.coordinates[0].0, p.coordinates[1].0, p.coordinates[2].0)
}

pub(crate) fn direction(s: &StepFile, a: Direction) -> DVec3 {
    let p = s.entity(a).expect("Could not get cartesian point");
    let d = DVec3::new(p.direction_ratios[0],
                       p.direction_ratios[1],
                       p.direction_ratios[2]);
    // STEP directions need not be unit length (some exporters leave a unit
    // conversion factor in them), so normalize before building frames.
    if d.norm() > 1e-12 { d.normalize() } else { d }
}

pub(crate) fn axis2_placement_3d(s: &StepFile, t: Id<Axis2Placement3d_>) -> (DVec3, DVec3, DVec3) {
    let a = s.entity(t).expect("Could not get Axis2Placement3d");
    let location = cartesian_point(s, a.location);
    // TODO: this doesn't necessarily match the behavior of `build_axes`
    // The axis is optional and defaults to +Z
    let axis = a.axis.map(|d| direction(s, d))
        .unwrap_or_else(|| DVec3::new(0.0, 0.0, 1.0));
    let ref_direction = build_ref_direction(axis, a.ref_direction.map(|r| direction(s, r)));
    (location, axis, ref_direction)
}

/// Builds a unit reference direction perpendicular to `axis`, following the
/// STEP definition of `axis2_placement_3d`: the reference direction is
/// projected onto the plane normal to the axis.  If it is omitted (or is
/// parallel to the axis), any perpendicular direction is valid, so we pick the
/// world axis least aligned with `axis` to stay numerically stable.
fn build_ref_direction(axis: DVec3, ref_direction: Option<DVec3>) -> DVec3 {
    const EPS: f64 = 1e-9;
    let axis_n = if axis.norm() > EPS { axis.normalize() } else { axis };
    let project = |v: DVec3| v - axis_n * v.dot(&axis_n);

    if let Some(r) = ref_direction {
        let p = project(r);
        if p.norm() > EPS {
            return p.normalize();
        }
    }
    let a = axis_n.abs();
    let least = if a.x <= a.y && a.x <= a.z {
        DVec3::new(1.0, 0.0, 0.0)
    } else if a.y <= a.z {
        DVec3::new(0.0, 1.0, 0.0)
    } else {
        DVec3::new(0.0, 0.0, 1.0)
    };
    let p = project(least);
    if p.norm() > EPS { p.normalize() } else { least }
}

fn shell(s: &StepFile, c: Shell, mesh: &mut Mesh, stats: &mut Stats) {
    match &s[c] {
        Entity::ClosedShell(_) => closed_shell(s, c.cast(), mesh, stats),
        Entity::OpenShell(_) => open_shell(s, c.cast(), mesh, stats),
        h => warn!("Skipping {:?} (unknown Shell type)", h),
    }
}

fn open_shell(s: &StepFile, c: OpenShell, mesh: &mut Mesh, stats: &mut Stats) {
    let cs = s.entity(c).expect("Could not get OpenShell");
    for face in &cs.cfs_faces {
        guarded_face(s, face.0, mesh, stats);
    }
    stats.num_shells += 1;
}

fn closed_shell(s: &StepFile, c: ClosedShell, mesh: &mut Mesh, stats: &mut Stats) {
    let cs = s.entity(c).expect("Could not get ClosedShell");
    for face in &cs.cfs_faces {
        guarded_face(s, face.0, mesh, stats);
    }
    stats.num_shells += 1;
}

/// Triangulates one face so that any failure, including a panic in surface
/// or edge setup, costs only that face and leaves no partial geometry.
fn guarded_face(s: &StepFile, face: usize, mesh: &mut Mesh, stats: &mut Stats) {
    let (nv, nt) = (mesh.verts.len(), mesh.triangles.len());
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(||
        advanced_face(s, Id::new(face), mesh, stats)));
    let failed = match result {
        Ok(Ok(())) => false,
        Ok(Err(err)) => {
            error!("Failed to triangulate face #{}: {}", face, err);
            stats.num_errors += 1;
            true
        },
        Err(_) => {
            error!("Panic while triangulating face #{}", face);
            stats.num_panics += 1;
            true
        },
    };
    if failed {
        mesh.verts.truncate(nv);
        mesh.triangles.truncate(nt);
    }
}

fn advanced_face(s: &StepFile, f: AdvancedFace, mesh: &mut Mesh,
                 stats: &mut Stats) -> Result<(), Error>
{
    let face = s.entity(f).expect("Could not get AdvancedFace");
    stats.num_faces += 1;

    // Walk the face's boundary loops first: swept surfaces are infinite in
    // STEP, so they need the boundary to know how far to extend.
    let bounds = face.bounds.iter()
        .map(|b| face_bound(s, *b))
        .collect::<Result<Vec<_>, _>>()?;
    let hint: Vec<DVec3> = bounds.iter().flatten().cloned().collect();

    // Grab the surface, returning early if it's unimplemented
    let mut surf = get_surface(s, face.face_geometry, &hint)?;

    // This is the starting point at which we insert new vertices
    let offset = mesh.verts.len();

    // For each contour, project from 3D down to the surface, then
    // start collecting them as constrained edges for triangulation
    let mut edges = Vec::new();
    let v_start = mesh.verts.len();
    let mut num_pts = 0;
    // Index range (relative to v_start) of each closed boundary loop
    let mut loops = Vec::new();
    for bound_contours in bounds {

        match bound_contours.len() {
            // We should always have non-zero items in the contour
            0 => panic!("Got empty contours for {:?}", face),

            // Special case for a single-vertex point, which shows up in
            // cones: we push it as a Steiner point, but without any
            // associated contours.
            1 => {
                num_pts += 1;
                mesh.verts.push(mesh::Vertex {
                    pos: bound_contours[0],
                    norm: DVec3::zeros(),
                    color: DVec3::new(0.0, 0.0, 0.0),
                });
            },

            // Default for lists of contour points
            _ => {
                // Record the initial point to close the loop
                let start = num_pts;
                for pt in bound_contours {
                    // The contour marches forward!
                    edges.push((num_pts, num_pts + 1));

                    // Also store this vertex in the 3D triangulation
                    mesh.verts.push(mesh::Vertex {
                        pos: pt,
                        norm: DVec3::zeros(),
                        color: DVec3::new(0.0, 0.0, 0.0),
                    });
                    num_pts += 1;
                }
                // The last point is a duplicate, because it closes the
                // contours, so we skip it here and reattach the contour to
                // the start.
                num_pts -= 1;
                mesh.verts.pop();

                // Close the loop by returning to the starting point
                edges.pop();
                edges.last_mut().unwrap().1 = start;
                loops.push(start..num_pts);
            }
        }
    }

    // We inject Stiner points based on the surface type to improve curvature,
    // e.g. for spherical sections.  However, we don't want triagulation to
    // _fail_ due to these points, so if that happens, we nuke the point (by
    // assigning it to the first point in the list, which causes it to get
    // deduplicated), then retry.
    let mut pts = surf.lower_verts(&mut mesh.verts[v_start..], &loops)?;
    let bonus_points = pts.len();
    surf.add_steiner_points(&mut pts, &mut mesh.verts);
    let result = std::panic::catch_unwind(|| {
        // TODO: this is only needed because we use pts below to save a debug
        // SVG if this panics.  Once we're confident in never panicking, we
        // can remove this.
        let mut pts = pts.clone();
        loop {
            let mut t = match cdt::Triangulation::new_with_edges(&pts, &edges) {
                Err(e) => break Err(e),
                Ok(t) => t,
            };
            match t.run() {
                Ok(()) => break Ok(t),
                // If triangulation failed due to a Steiner point on a fixed
                // edge, then reassign that point to pts[0] (so it will be
                // ignored as a duplicate)
                Err(cdt::Error::PointOnFixedEdge(p)) if p >= bonus_points => {
                    pts[p] = pts[0];
                    continue;
                },
                Err(e) => {
                    if SAVE_DEBUG_SVGS {
                        let filename = format!("err{}.svg", face.face_geometry.0);
                        t.save_debug_svg(&filename)
                            .expect("Could not save debug SVG");
                    }
                    break Err(e)
                },
            }
        }
    });
    match result {
        Ok(Ok(t)) => {
            for (a, b, c) in t.triangles() {
                let a = (a + offset) as u32;
                let b = (b + offset) as u32;
                let c = (c + offset) as u32;
                mesh.triangles.push(Triangle { verts:
                    if face.same_sense {
                        U32Vec3::new(a, b, c)
                    } else {
                        U32Vec3::new(a, c, b)
                    }
                });
            }
        },
        Ok(Err(e)) => {
            error!("Got error while triangulating {}: {:?}",
                   face.face_geometry.0, e);
            stats.num_errors += 1;
        },
        Err(e) => {
            error!("Got panic while triangulating {}: {:?}",
                   face.face_geometry.0, e);
            if SAVE_PANIC_SVGS {
                let filename = format!("panic{}.svg", face.face_geometry.0);
                cdt::save_debug_panic(&pts, &edges, &filename)
                    .expect("Could not save debug SVG");
            }
            stats.num_panics += 1;
        }
    }
    // Flip normals of new vertices, depending on the same_sense flag
    if !face.same_sense {
        for v in &mut mesh.verts[v_start..] {
            v.norm = -v.norm;
        }
    }
    Ok(())
}

fn get_surface(s: &StepFile, surf: ap214::Surface, hint: &[DVec3])
    -> Result<Surface, Error>
{
    match &s[surf] {
        Entity::SurfaceOfLinearExtrusion(e) => crate::swept::extrusion(s, e, hint),
        Entity::SurfaceOfRevolution(r) => crate::swept::revolution(s, r, hint),
        Entity::CylindricalSurface(c) => {
            let (location, axis, ref_direction) = axis2_placement_3d(s, c.position);
            Surface::new_cylinder(axis, ref_direction, location, c.radius.0.0.0)
        },
        Entity::ToroidalSurface(c) => {
            let (location, axis, ref_direction) = axis2_placement_3d(s, c.position);
            crate::swept::torus(location, axis, ref_direction,
                                c.major_radius.0.0.0, c.minor_radius.0.0.0)
        },
        Entity::Plane(p) => {
            // We'll ignore axis and ref_direction in favor of building an
            // orthonormal basis later on
            let (location, axis, ref_direction) = axis2_placement_3d(s, p.position);
            Surface::new_plane(axis, ref_direction, location)
        },
        // We treat cones like planes, since that's a valid mapping into 2D
        Entity::ConicalSurface(c) => {
            let (location, axis, ref_direction) = axis2_placement_3d(s, c.position);
            Surface::new_cone(axis, ref_direction, location, c.semi_angle.0)
        },
        Entity::SphericalSurface(c) => {
            // We'll ignore axis and ref_direction in favor of building an
            // orthonormal basis later on
            let (location, _axis, _ref_direction) = axis2_placement_3d(s, c.position);
            Ok(Surface::new_sphere(location, c.radius.0.0.0))
        },
        Entity::BSplineSurfaceWithKnots(b) =>
        {
            // TODO: make KnotVector::from_multiplicies accept iterators?
            let u_knots: Vec<f64> = b.u_knots.iter().map(|k| k.0).collect();
            let u_multiplicities: Vec<usize> = b.u_multiplicities.iter()
                .map(|&k| k.try_into().expect("Got negative multiplicity"))
                .collect();
            let u_knot_vec = KnotVector::from_multiplicities(
                b.u_degree.try_into().expect("Got negative degree"),
                &u_knots, &u_multiplicities);

            let v_knots: Vec<f64> = b.v_knots.iter().map(|k| k.0).collect();
            let v_multiplicities: Vec<usize> = b.v_multiplicities.iter()
                .map(|&k| k.try_into().expect("Got negative multiplicity"))
                .collect();
            let v_knot_vec = KnotVector::from_multiplicities(
                b.v_degree.try_into().expect("Got negative degree"),
                &v_knots, &v_multiplicities);

            let control_points_list = control_points_2d(s, &b.control_points_list);

            let surf = BSplineSurface::new(
                // `.U.` (unknown) counts as open; real closure is detected
                // from the control points in SampledSurface::new
                b.u_closed.0 != Some(true),
                b.v_closed.0 != Some(true),
                u_knot_vec,
                v_knot_vec,
                control_points_list,
            );
            Ok(Surface::BSpline(SampledSurface::new(surf), None))
        },
        Entity::ComplexEntity(v) if v.len() == 2 => {
            let bspline = if let Entity::BSplineSurfaceWithKnots(b) = &v[0] {
                b
            } else {
                warn!("Could not get BSplineCurveWithKnots from {:?}", v[0]);
                return Err(Error::UnknownCurveType)
            };
            let rational = if let Entity::RationalBSplineSurface(b) = &v[1] {
                b
            } else {
                warn!("Could not get RationalBSplineCurve from {:?}", v[1]);
                return Err(Error::UnknownCurveType)
            };

            // TODO: make KnotVector::from_multiplicies accept iterators?
            let u_knots: Vec<f64> = bspline.u_knots.iter().map(|k| k.0).collect();
            let u_multiplicities: Vec<usize> = bspline.u_multiplicities.iter()
                .map(|&k| k.try_into().expect("Got negative multiplicity"))
                .collect();
            let u_knot_vec = KnotVector::from_multiplicities(
                bspline.u_degree.try_into().expect("Got negative degree"),
                &u_knots, &u_multiplicities);

            let v_knots: Vec<f64> = bspline.v_knots.iter().map(|k| k.0).collect();
            let v_multiplicities: Vec<usize> = bspline.v_multiplicities.iter()
                .map(|&k| k.try_into().expect("Got negative multiplicity"))
                .collect();
            let v_knot_vec = KnotVector::from_multiplicities(
                bspline.v_degree.try_into().expect("Got negative degree"),
                &v_knots, &v_multiplicities);

            let control_points_list = control_points_2d(
                    s, &bspline.control_points_list)
                .into_iter()
                .zip(rational.weights_data.iter())
                .map(|(ctrl, weight)|
                    ctrl.into_iter()
                        .zip(weight.into_iter())
                        .map(|(p, w)| DVec4::new(p.x * w, p.y * w, p.z * w, *w))
                        .collect())
                .collect();

            let surf = NURBSSurface::new(
                bspline.u_closed.0 != Some(true),
                bspline.v_closed.0 != Some(true),
                u_knot_vec,
                v_knot_vec,
                control_points_list,
            );
            Ok(Surface::NURBS(SampledSurface::new(surf), None))

        },
        e => {
            warn!("Unsupported surface type {}", entity_name(e));
            Err(Error::UnknownSurfaceType)
        },
    }
}

pub(crate) fn control_points_1d(s: &StepFile, row: &Vec<CartesianPoint>) -> Vec<DVec3> {
    row.iter().map(|p| cartesian_point(s, *p)).collect()
}

fn control_points_2d(s: &StepFile, rows: &Vec<Vec<CartesianPoint>>) -> Vec<Vec<DVec3>> {
    rows.iter()
        .map(|row| control_points_1d(s, row))
        .collect()
}

fn face_bound(s: &StepFile, b: FaceBound) -> Result<Vec<DVec3>, Error> {
    let (bound, orientation) = match &s[b] {
        Entity::FaceBound(b) => (b.bound, b.orientation),
        Entity::FaceOuterBound(b) => (b.bound, b.orientation),
        e => panic!("Could not get bound from {:?} at {:?}", e, b),
    };
    match &s[bound] {
        Entity::EdgeLoop(e) => {
            let mut d = edge_loop(s, &e.edge_list)?;
            if !orientation {
                d.reverse()
            }
            Ok(d)
        },
        Entity::VertexLoop(v) => {
            // This is an "edge loop" with a single vertex, which is
            // used for cones and not really anything else.
            Ok(vec![vertex_point(s, v.loop_vertex)])
        }
        e => panic!("{:?} is not an EdgeLoop", e),
    }
}

fn edge_loop(s: &StepFile, edge_list: &[OrientedEdge])
    -> Result<Vec<DVec3>, Error>
{
    let mut out = Vec::new();
    for (i, e) in edge_list.iter().enumerate() {
        // Remove the last item from the list, since it's the beginning
        // of the following list (hopefully)
        if i > 0 {
            out.pop();
        }
        let edge = s.entity(*e).expect("Could not get OrientedEdge");
        let o = edge_curve(s, edge.edge_element.cast(), edge.orientation)?;
        out.extend(o.into_iter());
    }
    Ok(out)
}

fn edge_curve(s: &StepFile, e: EdgeCurve, orientation: bool) -> Result<Vec<DVec3>, Error> {
    let edge_curve = s.entity(e).expect("Could not get EdgeCurve");
    // An edge we can't model exactly is drawn straight between its vertices,
    // so the face still renders (approximately) instead of vanishing.
    let curve = match curve(s, edge_curve, edge_curve.edge_geometry, orientation) {
        Err(Error::UnknownCurveType) => Curve::new_line(),
        r => r?,
    };

    let (start, end) = if orientation {
        (edge_curve.edge_start, edge_curve.edge_end)
    } else {
        (edge_curve.edge_end, edge_curve.edge_start)
    };
    let u = vertex_point(s, start);
    let v = vertex_point(s, end);
    Ok(curve.build(u, v))
}

fn curve(s: &StepFile, edge_curve: &ap214::EdgeCurve_,
         curve_id: ap214::Curve, orientation: bool) -> Result<Curve, Error>
{
    Ok(match &s[curve_id] {
        Entity::Circle(c) => {
            let (location, axis, ref_direction) = axis2_placement_3d(s, c.position.cast());
            Curve::new_circle(location, axis, ref_direction, c.radius.0.0.0,
                              edge_curve.edge_start == edge_curve.edge_end,
                              edge_curve.same_sense ^ !orientation)
        },
        Entity::Ellipse(c) => {
            let (location, axis, ref_direction) = axis2_placement_3d(s, c.position.cast());
            Curve::new_ellipse(location, axis, ref_direction,
                               c.semi_axis_1.0.0.0, c.semi_axis_2.0.0.0,
                               edge_curve.edge_start == edge_curve.edge_end,
                               edge_curve.same_sense ^ !orientation)
        },
        Entity::BSplineCurveWithKnots(c) => {
            // Closed curves (e.g. Rhino's hole circles) are fine; only reject
            // curves that are known to cross themselves.
            if c.self_intersect.0 == Some(true) {
                return Err(Error::SelfIntersectingCurve);
            }

            let control_points_list = control_points_1d(
                s, &c.control_points_list);

            let knots: Vec<f64> = c.knots.iter().map(|k| k.0).collect();
            let multiplicities: Vec<usize> = c.knot_multiplicities.iter()
                .map(|&k| k.try_into().expect("Got negative multiplicity"))
                .collect();
            let knot_vec = KnotVector::from_multiplicities(
                c.degree.try_into().expect("Got negative degree"),
                &knots, &multiplicities);

            let periodic = c.closed_curve.0 == Some(true);
            let curve = nurbs::BSplineCurve::new(
                !periodic,
                knot_vec,
                control_points_list,
            );
            Curve::BSplineCurveWithKnots(SampledCurve::new(curve),
                                         spline_sense(edge_curve, periodic, orientation))
        },
        Entity::ComplexEntity(v) if v.len() == 2 => {
            let bspline = if let Entity::BSplineCurveWithKnots(b) = &v[0] {
                b
            } else {
                warn!("Could not get BSplineCurveWithKnots from {:?}", v[0]);
                return Err(Error::UnknownCurveType)
            };
            let rational = if let Entity::RationalBSplineCurve(b) = &v[1] {
                b
            } else {
                warn!("Could not get RationalBSplineCurve from {:?}", v[1]);
                return Err(Error::UnknownCurveType)
            };
            let knots: Vec<f64> = bspline.knots.iter().map(|k| k.0).collect();
            let multiplicities: Vec<usize> = bspline.knot_multiplicities.iter()
                .map(|&k| k.try_into().expect("Got negative multiplicity"))
                .collect();
            let knot_vec = KnotVector::from_multiplicities(
                bspline.degree.try_into().expect("Got negative degree"),
                &knots, &multiplicities);

            let control_points_list = control_points_1d(
                    s, &bspline.control_points_list)
                .into_iter()
                .zip(rational.weights_data.iter())
                .map(|(p, w)| DVec4::new(p.x * w, p.y * w, p.z * w, *w))
                .collect();

            let periodic = bspline.closed_curve.0 == Some(true);
            let curve = nurbs::NURBSCurve::new(
                !periodic,
                knot_vec,
                control_points_list,
            );
            Curve::NURBSCurve(SampledCurve::new(curve),
                              spline_sense(edge_curve, periodic, orientation))
        },
        Entity::SurfaceCurve(v) => {
            curve(s, edge_curve, v.curve_3d, orientation)?
        },
        Entity::SeamCurve(v) => {
            curve(s, edge_curve, v.curve_3d, orientation)?
        },
        // Rhino 8 writes arcs as a TRIMMED_CURVE around a CIRCLE.  The edge's
        // own vertices already mark the ends, so only the basis curve and its
        // direction matter here.
        Entity::TrimmedCurve(t) => {
            curve(s, edge_curve, t.basis_curve, orientation ^ !t.sense_agreement)?
        },
        Entity::Polyline(p) => Curve::Polyline(
            p.points.iter().map(|pt| cartesian_point(s, *pt)).collect()),
        // The Line type ignores pnt / dir and just uses u and v
        Entity::Line(_) => Curve::new_line(),
        e => {
            warn!("Unsupported edge curve {}, drawing it straight", entity_name(e));
            return Err(Error::UnknownCurveType);
        },
    })
}

/// Direction and closure of an edge along a spline curve, mirroring how the
/// CIRCLE / ELLIPSE arms work out direction.
fn spline_sense(edge_curve: &ap214::EdgeCurve_, periodic: bool, orientation: bool) -> Sense {
    Sense {
        closed: edge_curve.edge_start == edge_curve.edge_end,
        periodic,
        dir: edge_curve.same_sense ^ !orientation,
    }
}

/// Short entity name for log messages (e.g. "OffsetSurface")
fn entity_name(e: &Entity) -> String {
    let dbg = format!("{:?}", e);
    dbg.split(|c: char| c == '(' || c == ' ').next().unwrap_or("?").to_owned()
}

fn vertex_point(s: &StepFile, v: Vertex) -> DVec3 {
    cartesian_point(s,
        s.entity(v.cast::<VertexPoint_>())
            .expect("Could not get VertexPoint")
            .vertex_geometry
            .cast())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn omitted_ref_direction_is_perpendicular_to_axis() {
        for axis in [
            DVec3::new(1.0, 0.0, 0.0), DVec3::new(0.0, 1.0, 0.0), DVec3::new(0.0, 0.0, 1.0),
            DVec3::new(-1.0, 0.0, 0.0), DVec3::new(1.0, 1.0, 1.0).normalize(),
        ] {
            let r = build_ref_direction(axis, None);
            assert!((r.norm() - 1.0).abs() < 1e-9);
            assert!(r.dot(&axis).abs() < 1e-9);
            assert!(axis.cross(&r).norm() > 0.99);
        }
    }

    #[test]
    fn parallel_ref_direction_falls_back() {
        let axis = DVec3::new(1.0, 0.0, 0.0);
        let r = build_ref_direction(axis, Some(axis));
        assert!(r.dot(&axis).abs() < 1e-9);
    }

    #[test]
    fn valid_ref_direction_is_kept() {
        let r = build_ref_direction(DVec3::new(0.0, 0.0, 1.0), Some(DVec3::new(0.0, 1.0, 0.0)));
        assert!((r - DVec3::new(0.0, 1.0, 0.0)).norm() < 1e-9);
    }
}
