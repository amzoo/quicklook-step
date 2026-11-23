//! Regression tests for STEP constructs that used to drop faces.  Each test
//! is a tiny hand-written file in the style of a real exporter, checked
//! against the exact surface area.

use std::f64::consts::PI;

use step::step_file::StepFile;
use triangulate::{mesh::Mesh, stats::Stats, triangulate::triangulate};

const HALF_SQRT_2: &str = "0.707106781186548";

fn run(data: &str) -> (Mesh, Stats) {
    let file = format!("ISO-10303-21;\nHEADER;\nFILE_DESCRIPTION((''),'2;1');\n\
        FILE_NAME('t','',(''),(''),'','','');\nFILE_SCHEMA(('AUTOMOTIVE_DESIGN'));\n\
        ENDSEC;\nDATA;\n{}\nENDSEC;\nEND-ISO-10303-21;\n", data);
    let flat = StepFile::strip_flatten(file.as_bytes());
    let entities = StepFile::parse(&flat);
    triangulate(&entities)
}

fn area(mesh: &Mesh) -> f64 {
    mesh.triangles.iter().map(|t| {
        let p = |i: u32| mesh.verts[i as usize].pos;
        let (a, b, c) = (p(t.verts.x), p(t.verts.y), p(t.verts.z));
        (b - a).cross(&(c - a)).norm() / 2.0
    }).sum()
}

fn assert_close(actual: f64, expected: f64, tolerance: f64) {
    assert!((actual - expected).abs() <= expected * tolerance,
            "area {} differs from {} by more than {}%",
            actual, expected, tolerance * 100.0);
}

/// A 10 x 10 square in the XY plane at z = `z`, edges written the way Rhino
/// and NX write lines: degree-1 B-splines, with `self_intersect` given as
/// `.F.` (Rhino) or `.U.` (NX, CATIA).
/// Entities #100-#163; the outer bound is #161 and the plane is #105.
fn square(z: f64, self_intersect: &str) -> String {
    let mut s = format!("#101=CARTESIAN_POINT('',(0.,0.,{z}));
#102=DIRECTION('',(0.,0.,1.));
#103=DIRECTION('',(1.,0.,0.));
#104=AXIS2_PLACEMENT_3D('',#101,#102,#103);
#105=PLANE('',#104);
#110=CARTESIAN_POINT('',(0.,0.,{z}));
#111=CARTESIAN_POINT('',(10.,0.,{z}));
#112=CARTESIAN_POINT('',(10.,10.,{z}));
#113=CARTESIAN_POINT('',(0.,10.,{z}));
");
    for i in 0..4 {
        let j = (i + 1) % 4;
        s += &format!("#{}=VERTEX_POINT('',#{});\n", 120 + i, 110 + i);
        s += &format!("#{}=B_SPLINE_CURVE_WITH_KNOTS('',1,(#{},#{}),.UNSPECIFIED.,.F.,{},(2,2),(0.,10.),.UNSPECIFIED.);\n",
                      130 + i, 110 + i, 110 + j, self_intersect);
        s += &format!("#{}=EDGE_CURVE('',#{},#{},#{},.T.);\n", 140 + i, 120 + i, 120 + j, 130 + i);
        s += &format!("#{}=ORIENTED_EDGE('',*,*,#{},.T.);\n", 150 + i, 140 + i);
    }
    s += "#160=EDGE_LOOP('',(#150,#151,#152,#153));\n#161=FACE_OUTER_BOUND('',#160,.T.);\n";
    s
}

/// Rhino writes a hole as one closed edge (start vertex == end vertex) over a
/// closed rational B-spline circle.  Entities #200-#230; bound is #222.
fn rhino_hole(cx: f64, cy: f64, r: f64, z: f64) -> String {
    let corners = [(1., 0.), (1., 1.), (0., 1.), (-1., 1.), (-1., 0.),
                   (-1., -1.), (0., -1.), (1., -1.), (1., 0.)];
    let mut s = String::new();
    for (k, (x, y)) in corners.iter().enumerate() {
        s += &format!("#{}=CARTESIAN_POINT('',({},{},{z}));\n", 200 + k, cx + r * x, cy + r * y);
    }
    let w = HALF_SQRT_2;
    s += &format!("#210=(BOUNDED_CURVE()B_SPLINE_CURVE(2,(#200,#201,#202,#203,#204,#205,#206,#207,#208),\
.UNSPECIFIED.,.T.,.F.)B_SPLINE_CURVE_WITH_KNOTS((3,2,2,2,3),(0.,1.,2.,3.,4.),.UNSPECIFIED.)\
CURVE()GEOMETRIC_REPRESENTATION_ITEM()RATIONAL_B_SPLINE_CURVE((1.,{w},1.,{w},1.,{w},1.,{w},1.))\
REPRESENTATION_ITEM(''));\n");
    s += "#211=VERTEX_POINT('',#200);
#212=EDGE_CURVE('',#211,#211,#210,.T.);
#220=ORIENTED_EDGE('',*,*,#212,.F.);
#221=EDGE_LOOP('',(#220));
#222=FACE_BOUND('',#221,.T.);
";
    s
}

#[test]
fn closed_bspline_hole_is_cut_out() {
    let data = format!("{}{}#300=ADVANCED_FACE('',(#161,#222),#105,.T.);
#301=OPEN_SHELL('',(#300));
#302=SHELL_BASED_SURFACE_MODEL('',(#301));",
        square(0.0, ".U."), rhino_hole(5.0, 5.0, 2.0, 0.0));
    let (mesh, stats) = run(&data);
    assert_eq!(stats.num_errors, 0);
    assert_eq!(stats.num_panics, 0);
    assert_close(area(&mesh), 100.0 - PI * 4.0, 0.01);
}

#[test]
fn trimmed_circle_edge_is_followed() {
    // Half disc of radius 3: a straight edge plus a Rhino 8 style
    // TRIMMED_CURVE(CIRCLE) arc.
    let data = "#1=CARTESIAN_POINT('',(0.,0.,0.));
#2=DIRECTION('',(0.,0.,1.));
#3=DIRECTION('',(1.,0.,0.));
#4=AXIS2_PLACEMENT_3D('',#1,#2,#3);
#5=PLANE('',#4);
#6=CIRCLE('',#4,3.);
#10=CARTESIAN_POINT('',(3.,0.,0.));
#11=CARTESIAN_POINT('',(-3.,0.,0.));
#12=VERTEX_POINT('',#10);
#13=VERTEX_POINT('',#11);
#20=TRIMMED_CURVE($,#6,(#10),(#11),.T.,.CARTESIAN.);
#21=B_SPLINE_CURVE_WITH_KNOTS('',1,(#11,#10),.UNSPECIFIED.,.F.,.F.,(2,2),(0.,6.),.UNSPECIFIED.);
#30=EDGE_CURVE('',#12,#13,#20,.T.);
#31=EDGE_CURVE('',#13,#12,#21,.T.);
#40=ORIENTED_EDGE('',*,*,#30,.T.);
#41=ORIENTED_EDGE('',*,*,#31,.T.);
#50=EDGE_LOOP('',(#40,#41));
#51=FACE_OUTER_BOUND('',#50,.T.);
#60=ADVANCED_FACE('',(#51),#5,.T.);
#61=OPEN_SHELL('',(#60));
#62=SHELL_BASED_SURFACE_MODEL('',(#61));";
    let (mesh, stats) = run(data);
    assert_eq!(stats.num_errors, 0);
    assert_close(area(&mesh), PI * 9.0 / 2.0, 0.01);
}

/// A full torus (major radius 30, minor 8) bounded by its two seam circles,
/// the way OpenCascade writes it.
fn torus_face() -> &'static str {
    "#1=CARTESIAN_POINT('',(0.,0.,0.));
#2=DIRECTION('',(0.,0.,1.));
#3=DIRECTION('',(1.,0.,0.));
#4=AXIS2_PLACEMENT_3D('',#1,#2,#3);
#5=TOROIDAL_SURFACE('',#4,30.,8.);
#6=CIRCLE('',#4,38.);
#7=CARTESIAN_POINT('',(30.,0.,0.));
#8=DIRECTION('',(0.,-1.,0.));
#9=AXIS2_PLACEMENT_3D('',#7,#8,#3);
#10=CIRCLE('',#9,8.);
#11=CARTESIAN_POINT('',(38.,0.,0.));
#12=VERTEX_POINT('',#11);
#20=EDGE_CURVE('',#12,#12,#6,.T.);
#21=EDGE_CURVE('',#12,#12,#10,.T.);
#30=ORIENTED_EDGE('',*,*,#20,.F.);
#31=ORIENTED_EDGE('',*,*,#21,.F.);
#32=ORIENTED_EDGE('',*,*,#20,.T.);
#33=ORIENTED_EDGE('',*,*,#21,.T.);
#40=EDGE_LOOP('',(#30,#31,#32,#33));
#41=FACE_BOUND('',#40,.T.);
#50=ADVANCED_FACE('',(#41),#5,.T.);
#51=OPEN_SHELL('',(#50));
#52=SHELL_BASED_SURFACE_MODEL('',(#51));"
}

#[test]
fn full_torus_is_closed_in_both_directions() {
    let (mesh, stats) = run(torus_face());
    assert_eq!(stats.num_errors, 0);
    assert_close(area(&mesh), 4.0 * PI * PI * 30.0 * 8.0, 0.01);

    // Normals must point away from the tube's centre circle
    for t in &mesh.triangles {
        let p = |i: u32| mesh.verts[i as usize].pos;
        let (a, b, c) = (p(t.verts.x), p(t.verts.y), p(t.verts.z));
        let centroid = (a + b + c) / 3.0;
        let mut ring = centroid;
        ring.z = 0.0;
        let ring = ring.normalize() * 30.0;
        assert!((b - a).cross(&(c - a)).dot(&(centroid - ring)) > 0.0);
    }
}

#[test]
fn torus_band_without_seam_uses_ring_mapping() {
    // The outer half of a torus (major 30, minor 8) bounded by two full
    // circles and no seam edge, like a fillet band from CATIA or NX.
    let data = "#1=CARTESIAN_POINT('',(0.,0.,0.));
#2=DIRECTION('',(0.,0.,1.));
#3=DIRECTION('',(1.,0.,0.));
#4=AXIS2_PLACEMENT_3D('',#1,#2,#3);
#5=TOROIDAL_SURFACE('',#4,30.,8.);
#6=CARTESIAN_POINT('',(0.,0.,8.));
#7=AXIS2_PLACEMENT_3D('',#6,#2,#3);
#8=CIRCLE('',#7,30.);
#9=CARTESIAN_POINT('',(0.,0.,-8.));
#10=AXIS2_PLACEMENT_3D('',#9,#2,#3);
#11=CIRCLE('',#10,30.);
#12=CARTESIAN_POINT('',(30.,0.,8.));
#13=CARTESIAN_POINT('',(30.,0.,-8.));
#14=VERTEX_POINT('',#12);
#15=VERTEX_POINT('',#13);
#20=EDGE_CURVE('',#14,#14,#8,.T.);
#21=EDGE_CURVE('',#15,#15,#11,.T.);
#30=ORIENTED_EDGE('',*,*,#20,.T.);
#31=ORIENTED_EDGE('',*,*,#21,.F.);
#40=EDGE_LOOP('',(#30));
#41=EDGE_LOOP('',(#31));
#42=FACE_OUTER_BOUND('',#40,.T.);
#43=FACE_BOUND('',#41,.T.);
#50=ADVANCED_FACE('',(#42,#43),#5,.T.);
#51=OPEN_SHELL('',(#50));
#52=SHELL_BASED_SURFACE_MODEL('',(#51));";
    let (mesh, stats) = run(data);
    assert_eq!(stats.num_errors, 0);
    // Outer half: v from -90 to 90 degrees, area = 2*pi*r*(pi*R + 2*r)
    assert_close(area(&mesh), 2.0 * PI * 8.0 * (PI * 30.0 + 2.0 * 8.0), 0.02);
}

#[test]
fn surface_of_revolution_is_triangulated() {
    // A cylinder wall (radius 5, height 10) written as a line revolved
    // around Z, bounded by a seam line and two closed circles.
    let data = "#1=CARTESIAN_POINT('',(0.,0.,0.));
#2=DIRECTION('',(0.,0.,1.));
#3=DIRECTION('',(1.,0.,0.));
#4=AXIS1_PLACEMENT('',#1,#2);
#5=CARTESIAN_POINT('',(5.,0.,0.));
#6=VECTOR('',#2,1.);
#7=LINE('',#5,#6);
#8=SURFACE_OF_REVOLUTION('',#7,#4);
#10=CARTESIAN_POINT('',(5.,0.,10.));
#11=VERTEX_POINT('',#5);
#12=VERTEX_POINT('',#10);
#13=AXIS2_PLACEMENT_3D('',#1,#2,#3);
#14=CIRCLE('',#13,5.);
#15=CARTESIAN_POINT('',(0.,0.,10.));
#16=AXIS2_PLACEMENT_3D('',#15,#2,#3);
#17=CIRCLE('',#16,5.);
#20=EDGE_CURVE('',#11,#12,#7,.T.);
#21=EDGE_CURVE('',#11,#11,#14,.T.);
#22=EDGE_CURVE('',#12,#12,#17,.T.);
#30=ORIENTED_EDGE('',*,*,#21,.T.);
#31=ORIENTED_EDGE('',*,*,#20,.T.);
#32=ORIENTED_EDGE('',*,*,#22,.F.);
#33=ORIENTED_EDGE('',*,*,#20,.F.);
#40=EDGE_LOOP('',(#30,#31,#32,#33));
#41=FACE_OUTER_BOUND('',#40,.T.);
#50=ADVANCED_FACE('',(#41),#8,.T.);
#51=OPEN_SHELL('',(#50));
#52=SHELL_BASED_SURFACE_MODEL('',(#51));";
    let (mesh, stats) = run(data);
    assert_eq!(stats.num_errors, 0);
    assert_close(area(&mesh), 2.0 * PI * 5.0 * 10.0, 0.01);
}

#[test]
fn construction_geometry_is_hidden() {
    let data = format!("{}#300=ADVANCED_FACE('',(#161),#105,.T.);
#301=OPEN_SHELL('',(#300));
#302=SHELL_BASED_SURFACE_MODEL('',(#301));
#303=CONSTRUCTIVE_GEOMETRY_REPRESENTATION('supplemental geometry',(#302),#104);",
        square(0.0, ".F."));
    let (mesh, _stats) = run(&data);
    assert!(mesh.triangles.is_empty());
}

#[test]
fn unknown_logical_is_parsed() {
    // `.U.` (unknown) in a LOGICAL used to make the whole entity unreadable,
    // dropping every face that touched it.
    let data = format!("{}#300=ADVANCED_FACE('',(#161),#105,.T.);
#301=OPEN_SHELL('',(#300));
#302=SHELL_BASED_SURFACE_MODEL('',(#301));", square(0.0, ".U."));
    let (mesh, stats) = run(&data);
    assert_eq!(stats.num_errors, 0);
    assert_close(area(&mesh), 100.0, 1e-9);
}

#[test]
fn unknown_surface_closure_flag_does_not_blank_the_file() {
    // NX/CATIA write `.U.` for a B-spline surface's u_closed / v_closed.  This
    // used to panic, which made the whole preview blank.
    let data = square(0.0, ".F.").replace("#105=PLANE('',#104);",
        "#170=CARTESIAN_POINT('',(0.,0.,0.));
#171=CARTESIAN_POINT('',(0.,10.,0.));
#172=CARTESIAN_POINT('',(10.,0.,0.));
#173=CARTESIAN_POINT('',(10.,10.,0.));
#105=B_SPLINE_SURFACE_WITH_KNOTS('',1,1,((#170,#171),(#172,#173)),.UNSPECIFIED.,.U.,.U.,.U.,(2,2),(2,2),(0.,1.),(0.,1.),.UNSPECIFIED.);");
    let data = format!("{}#300=ADVANCED_FACE('',(#161),#105,.T.);
#301=OPEN_SHELL('',(#300));
#302=SHELL_BASED_SURFACE_MODEL('',(#301));", data);
    let (mesh, stats) = run(&data);
    assert_eq!(stats.num_errors + stats.num_panics, 0);
    assert_close(area(&mesh), 100.0, 1e-6);
}

#[test]
fn non_unit_directions_are_normalized() {
    // ftc_08 in the NIST set writes directions scaled by 1/25.4.  The hole's
    // circle placement uses them too, so an unnormalized frame squashes it.
    let data = square(0.0, ".F.")
        .replace("#102=DIRECTION('',(0.,0.,1.));", "#102=DIRECTION('',(0.,0.,0.0393700787402));")
        .replace("#103=DIRECTION('',(1.,0.,0.));", "#103=DIRECTION('',(0.0393700787402,0.,0.));");
    let data = format!("{}#200=CARTESIAN_POINT('',(5.,5.,0.));
#201=AXIS2_PLACEMENT_3D('',#200,#102,#103);
#202=CIRCLE('',#201,2.);
#203=CARTESIAN_POINT('',(7.,5.,0.));
#204=VERTEX_POINT('',#203);
#205=EDGE_CURVE('',#204,#204,#202,.T.);
#206=ORIENTED_EDGE('',*,*,#205,.F.);
#207=EDGE_LOOP('',(#206));
#208=FACE_BOUND('',#207,.T.);
#300=ADVANCED_FACE('',(#161,#208),#105,.T.);
#301=OPEN_SHELL('',(#300));
#302=SHELL_BASED_SURFACE_MODEL('',(#301));", data);
    let (mesh, stats) = run(&data);
    assert_eq!(stats.num_errors + stats.num_panics, 0);
    assert_close(area(&mesh), 100.0 - PI * 4.0, 0.01);
}

#[test]
fn omitted_axis_defaults_to_z() {
    let data = square(0.0, ".F.")
        .replace("#104=AXIS2_PLACEMENT_3D('',#101,#102,#103);", "#104=AXIS2_PLACEMENT_3D('',#101,$,$);");
    let data = format!("{}#300=ADVANCED_FACE('',(#161),#105,.T.);
#301=OPEN_SHELL('',(#300));
#302=SHELL_BASED_SURFACE_MODEL('',(#301));", data);
    let (mesh, stats) = run(&data);
    assert_eq!(stats.num_errors + stats.num_panics, 0);
    assert_close(area(&mesh), 100.0, 1e-6);
}

/// A sphere of radius 5 at the origin, cut by one circle at height `z` that
/// runs clockwise seen from above (`.F.`), so the face is everything except
/// the cap above the circle.
fn sphere_without_cap(z: f64) -> String {
    let r = (25.0 - z * z).sqrt();
    format!("#1=CARTESIAN_POINT('',(0.,0.,0.));
#2=DIRECTION('',(0.,0.,1.));
#3=DIRECTION('',(1.,0.,0.));
#4=AXIS2_PLACEMENT_3D('',#1,#2,#3);
#5=SPHERICAL_SURFACE('',#4,5.);
#6=CARTESIAN_POINT('',(0.,0.,{z}));
#7=AXIS2_PLACEMENT_3D('',#6,#2,#3);
#8=CIRCLE('',#7,{r});
#9=CARTESIAN_POINT('',({r},0.,{z}));
#10=VERTEX_POINT('',#9);
#11=EDGE_CURVE('',#10,#10,#8,.T.);
#12=ORIENTED_EDGE('',*,*,#11,.F.);
#13=EDGE_LOOP('',(#12));
#14=FACE_OUTER_BOUND('',#13,.T.);
#15=ADVANCED_FACE('',(#14),#5,.T.);
#16=OPEN_SHELL('',(#15));
#17=SHELL_BASED_SURFACE_MODEL('',(#16));")
}

#[test]
fn sphere_face_larger_than_hemisphere() {
    // Cap above z = 4 removed: area = 4*pi*r^2 - 2*pi*r*h with h = 1
    let (mesh, stats) = run(&sphere_without_cap(4.0));
    assert_eq!(stats.num_errors + stats.num_panics, 0);
    assert_close(area(&mesh), 100.0 * PI - 10.0 * PI, 0.02);
}

#[test]
fn hemisphere_bounded_by_great_circle() {
    let (mesh, stats) = run(&sphere_without_cap(0.0));
    assert_eq!(stats.num_errors + stats.num_panics, 0);
    assert_close(area(&mesh), 50.0 * PI, 0.02);
}
