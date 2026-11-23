//! Tessellated (mesh) geometry from AP242 files.
//!
//! AP242 can store a part as triangles instead of (or as well as) a B-rep:
//! `TESSELLATED_SOLID` / `TESSELLATED_SHELL` items made of
//! `TRIANGULATED_FACE`, `COMPLEX_TRIANGULATED_FACE`, `TRIANGULATED_SURFACE_SET`
//! or `COMPLEX_TRIANGULATED_SURFACE_SET`, whose points live in a
//! `COORDINATES_LIST`.  These entities are not in the AP214 schema the parser
//! is generated from, so they are read here straight from the flattened file
//! text (see `StepFile::strip_flatten`).  The geometry is already triangles,
//! so it is copied out rather than triangulated.

use std::collections::{HashMap, HashSet};

use nalgebra_glm::{DVec3, U32Vec3};

use crate::mesh::{Mesh, Triangle, Vertex};

/// A parsed STEP parameter value
#[derive(Debug, Clone)]
enum Value {
    List(Vec<Value>),
    Num(f64),
    Ref(usize),
    Other,
}

impl Value {
    fn list(&self) -> &[Value] {
        match self {
            Value::List(v) => v,
            _ => &[],
        }
    }
    fn num(&self) -> Option<f64> {
        match self {
            Value::Num(n) => Some(*n),
            _ => None,
        }
    }
    fn reference(&self) -> Option<usize> {
        match self {
            Value::Ref(r) => Some(*r),
            _ => None,
        }
    }
    /// A list of 1-based indices
    fn indices(&self) -> Vec<usize> {
        self.list().iter()
            .filter_map(|v| v.num())
            .filter(|n| *n >= 1.0)
            .map(|n| n as usize)
            .collect()
    }
}

/// Parses one parameter value starting at `s[*i]`
fn parse_value(s: &[u8], i: &mut usize) -> Value {
    match s.get(*i) {
        Some(b'(') => {
            *i += 1;
            let mut items = Vec::new();
            while *i < s.len() && s[*i] != b')' {
                items.push(parse_value(s, i));
                if s.get(*i) == Some(&b',') {
                    *i += 1;
                }
            }
            *i += 1; // closing ')'
            Value::List(items)
        },
        Some(b'#') => {
            *i += 1;
            let start = *i;
            while *i < s.len() && s[*i].is_ascii_digit() {
                *i += 1;
            }
            std::str::from_utf8(&s[start..*i]).ok()
                .and_then(|t| t.parse().ok())
                .map(Value::Ref)
                .unwrap_or(Value::Other)
        },
        Some(b'\'') => {
            // String; '' is an escaped quote
            *i += 1;
            while *i < s.len() {
                if s[*i] == b'\'' {
                    if s.get(*i + 1) == Some(&b'\'') {
                        *i += 2;
                        continue;
                    }
                    *i += 1;
                    break;
                }
                *i += 1;
            }
            Value::Other
        },
        Some(c) if c.is_ascii_digit() || *c == b'-' || *c == b'+' || *c == b'.' => {
            let start = *i;
            while *i < s.len() && matches!(s[*i], b'0'..=b'9' | b'-' | b'+' | b'.' | b'E' | b'e') {
                *i += 1;
            }
            std::str::from_utf8(&s[start..*i]).ok()
                .and_then(|t| t.parse().ok())
                .map(Value::Num)
                .unwrap_or(Value::Other)
        },
        _ => {
            // $, *, .ENUM., typed values like NAME(..): skip to the next
            // separator at this nesting level
            let mut depth = 0;
            while *i < s.len() {
                match s[*i] {
                    b'(' => depth += 1,
                    b')' if depth == 0 => break,
                    b')' => depth -= 1,
                    b',' if depth == 0 => break,
                    _ => (),
                }
                *i += 1;
            }
            Value::Other
        },
    }
}

/// Splits flattened STEP text into `(id, NAME, args)` records, keeping only
/// the entity names we care about.
fn records<'a>(flat: &'a [u8], wanted: &[&str]) -> Vec<(usize, &'a str, Vec<Value>)> {
    let mut out = Vec::new();
    let mut start = 0;
    let mut in_str = false;
    for (i, &c) in flat.iter().enumerate() {
        match c {
            b'\'' => in_str = !in_str,
            b';' if !in_str => {
                let rec = &flat[start..i];
                start = i + 1;
                // #id=NAME(args)
                if rec.first() != Some(&b'#') {
                    continue;
                }
                let Some(eq) = rec.iter().position(|&b| b == b'=') else { continue };
                let Some(paren) = rec.iter().position(|&b| b == b'(') else { continue };
                if paren < eq {
                    continue;
                }
                let name = match std::str::from_utf8(&rec[eq + 1..paren]) {
                    Ok(n) if wanted.contains(&n) => n,
                    _ => continue,
                };
                let Some(id) = std::str::from_utf8(&rec[1..eq]).ok()
                    .and_then(|t| t.parse().ok()) else { continue };
                let mut j = paren;
                let args = match parse_value(rec, &mut j) {
                    Value::List(v) => v,
                    _ => continue,
                };
                out.push((id, name, args));
            },
            _ => (),
        }
    }
    out
}

const FACES: [&str; 4] = [
    "TRIANGULATED_FACE", "COMPLEX_TRIANGULATED_FACE",
    "TRIANGULATED_SURFACE_SET", "COMPLEX_TRIANGULATED_SURFACE_SET",
];

/// Builds a mesh from every tessellated solid or shell in the file.  Returns
/// an empty mesh if the file has no tessellated geometry.
pub fn mesh_from_flat(flat: &[u8]) -> Mesh {
    let mut wanted = vec!["COORDINATES_LIST", "TESSELLATED_SOLID", "TESSELLATED_SHELL",
                          "TESSELLATED_GEOMETRIC_SET"];
    wanted.extend(FACES.iter());
    let recs = records(flat, &wanted);

    let coords: HashMap<usize, Vec<DVec3>> = recs.iter()
        .filter(|r| r.1 == "COORDINATES_LIST")
        .map(|(id, _, a)| {
            let pts = a.get(2).map(|v| v.list()).unwrap_or(&[]).iter()
                .map(|p| {
                    let c: Vec<f64> = p.list().iter().filter_map(|x| x.num()).collect();
                    DVec3::new(*c.first().unwrap_or(&0.0), *c.get(1).unwrap_or(&0.0),
                               *c.get(2).unwrap_or(&0.0))
                })
                .collect();
            (*id, pts)
        })
        .collect();

    // Faces that make up the part: those listed by a tessellated solid or
    // shell.  Without any, take every face that isn't part of an annotation
    // (TESSELLATED_GEOMETRIC_SET holds PMI graphics).
    let items_of = |kind: &str| -> HashSet<usize> {
        recs.iter()
            .filter(|r| r.1 == kind)
            .flat_map(|(_, _, a)| a.get(1).map(|v| v.list().to_vec()).unwrap_or_default())
            .filter_map(|v| v.reference())
            .collect()
    };
    let part = {
        let mut s = items_of("TESSELLATED_SOLID");
        s.extend(items_of("TESSELLATED_SHELL"));
        s
    };
    let annotation = items_of("TESSELLATED_GEOMETRIC_SET");

    let mut mesh = Mesh::default();
    for (id, name, a) in recs.iter().filter(|r| FACES.contains(&r.1)) {
        let keep = if part.is_empty() { !annotation.contains(id) } else { part.contains(id) };
        if keep {
            add_face(&mut mesh, name, a, &coords);
        }
    }
    mesh
}

/// Appends one tessellated face (or surface set) to `mesh`
fn add_face(mesh: &mut Mesh, name: &str, a: &[Value], coords: &HashMap<usize, Vec<DVec3>>) {
    let Some(points) = a.get(1).and_then(|v| v.reference()).and_then(|r| coords.get(&r))
        else { return };
    // Faces carry a geometric_link after the normals; surface sets don't
    let has_link = name.ends_with("_FACE");
    let at = |k: usize| a.get(if has_link { k + 1 } else { k });
    let normals: Vec<DVec3> = at(3).map(|v| v.list()).unwrap_or(&[]).iter()
        .map(|n| {
            let c: Vec<f64> = n.list().iter().filter_map(|x| x.num()).collect();
            DVec3::new(*c.first().unwrap_or(&0.0), *c.get(1).unwrap_or(&0.0),
                       *c.get(2).unwrap_or(&0.0))
        })
        .collect();
    let pnindex = at(4).map(|v| v.indices()).unwrap_or_default();

    // Triangles as triples of 1-based index positions
    let mut tris: Vec<[usize; 3]> = Vec::new();
    if name.starts_with("COMPLEX_") {
        for strip in at(5).map(|v| v.list()).unwrap_or(&[]) {
            let s = strip.indices();
            for k in 0..s.len().saturating_sub(2) {
                // Every other triangle in a strip is wound the other way
                let t = if k % 2 == 0 { [s[k], s[k + 1], s[k + 2]] }
                        else { [s[k + 1], s[k], s[k + 2]] };
                tris.push(t);
            }
        }
        for fan in at(6).map(|v| v.list()).unwrap_or(&[]) {
            let f = fan.indices();
            for k in 1..f.len().saturating_sub(1) {
                tris.push([f[0], f[k], f[k + 1]]);
            }
        }
    } else {
        for t in at(5).map(|v| v.list()).unwrap_or(&[]) {
            let t = t.indices();
            if t.len() == 3 {
                tris.push([t[0], t[1], t[2]]);
            }
        }
    }

    // Index position -> point
    let point = |idx: usize| -> Option<DVec3> {
        let p = if pnindex.is_empty() { idx } else { *pnindex.get(idx - 1)? };
        points.get(p.checked_sub(1)?).copied()
    };

    // One vertex per index position used by this face
    let mut vert_of: HashMap<usize, u32> = HashMap::new();
    let first_new = mesh.verts.len();
    let mut new_tris = Vec::new();
    for t in tris {
        if t[0] == t[1] || t[1] == t[2] || t[0] == t[2] {
            continue;
        }
        let mut ids = [0u32; 3];
        let mut ok = true;
        for (k, &idx) in t.iter().enumerate() {
            let Some(pos) = point(idx) else { ok = false; break };
            ids[k] = *vert_of.entry(idx).or_insert_with(|| {
                let normal = match normals.len() {
                    1 => normals[0],
                    n if n > 0 && idx <= n => normals[idx - 1],
                    _ => DVec3::zeros(),
                };
                mesh.verts.push(Vertex { pos, norm: normal, color: DVec3::zeros() });
                (mesh.verts.len() - 1) as u32
            });
        }
        if ok {
            new_tris.push(U32Vec3::new(ids[0], ids[1], ids[2]));
        }
    }

    // Without usable normals in the file, average the triangle normals
    if mesh.verts[first_new..].iter().any(|v| v.norm.norm() < 1e-12) {
        for v in &mut mesh.verts[first_new..] {
            v.norm = DVec3::zeros();
        }
        for t in &new_tris {
            let p = |i: u32| mesh.verts[i as usize].pos;
            let n = (p(t.y) - p(t.x)).cross(&(p(t.z) - p(t.x)));
            for i in [t.x, t.y, t.z] {
                mesh.verts[i as usize].norm += n;
            }
        }
        for v in &mut mesh.verts[first_new..] {
            if v.norm.norm() > 0.0 {
                v.norm = v.norm.normalize();
            }
        }
    }
    mesh.triangles.extend(new_tris.into_iter().map(|verts| Triangle { verts }));
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::triangulate::triangulate;
    use step::step_file::StepFile;

    fn area(mesh: &Mesh) -> f64 {
        mesh.triangles.iter().map(|t| {
            let p = |i: u32| mesh.verts[i as usize].pos;
            (p(t.verts.y) - p(t.verts.x)).cross(&(p(t.verts.z) - p(t.verts.x))).norm() / 2.0
        }).sum()
    }

    fn flat(data: &str) -> Vec<u8> {
        StepFile::strip_flatten(format!("ISO-10303-21;\nHEADER;\nENDSEC;\nDATA;\n{}\nENDSEC;\n\
            END-ISO-10303-21;\n", data).as_bytes())
    }

    #[test]
    fn triangulated_face() {
        // Unit square as two triangles, plus a 'text;with;semicolons' name
        let f = flat("#1=COORDINATES_LIST('',4,((0.,0.,0.),(1.,0.,0.),(1.,1.,0.),(0.,1.,0.)));
#2=TRIANGULATED_FACE('text;with;semicolons',#1,4,((0.,0.,1.)),$,(),((1,2,3),(1,3,4)));
#3=TESSELLATED_SOLID('',(#2),$);");
        let mesh = mesh_from_flat(&f);
        assert_eq!(mesh.triangles.len(), 2);
        assert!((area(&mesh) - 1.0).abs() < 1e-12);
        assert!(mesh.verts.iter().all(|v| (v.norm.z - 1.0).abs() < 1e-12));
    }

    #[test]
    fn complex_face_strips_and_fans() {
        // A 2x1 rectangle as a 4-point strip, plus a unit square as a fan,
        // with pnindex remapping into the coordinates list
        let f = flat("#1=COORDINATES_LIST('',8,((0.,0.,0.),(2.,0.,0.),(0.,1.,0.),(2.,1.,0.),
(5.,0.,0.),(6.,0.,0.),(6.,1.,0.),(5.,1.,0.)));
#2=COMPLEX_TRIANGULATED_FACE('',#1,8,(),$,(1,2,3,4,5,6,7,8),((1,2,3,4)),((5,6,7,8)));
#3=TESSELLATED_SHELL('',(#2),$);");
        let mesh = mesh_from_flat(&f);
        assert_eq!(mesh.triangles.len(), 4);
        assert!((area(&mesh) - 3.0).abs() < 1e-12);
        // Computed normals are unit length
        assert!(mesh.verts.iter().all(|v| (v.norm.norm() - 1.0).abs() < 1e-9));
    }

    #[test]
    fn annotation_geometry_is_ignored() {
        let f = flat("#1=COORDINATES_LIST('',3,((0.,0.,0.),(1.,0.,0.),(0.,1.,0.)));
#2=TRIANGULATED_FACE('',#1,3,(),$,(),((1,2,3)));
#3=TESSELLATED_GEOMETRIC_SET('datum',(#2));");
        assert!(mesh_from_flat(&f).triangles.is_empty());
    }

    #[test]
    fn brep_files_have_no_tessellation() {
        let data = "#1=CARTESIAN_POINT('',(0.,0.,0.));";
        let f = flat(data);
        assert!(mesh_from_flat(&f).triangles.is_empty());
        let _ = triangulate(&StepFile::parse(&f));
    }
}
