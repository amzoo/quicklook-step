use std::ffi::CStr;
use std::os::raw::c_char;
use std::ptr;

use step::step_file::StepFile;
use triangulate::triangulate::triangulate;

#[repr(C)]
pub struct FFIVertex {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub nx: f32,
    pub ny: f32,
    pub nz: f32,
    pub r: f32,
    pub g: f32,
    pub b: f32,
}

#[repr(C)]
pub struct FFIMesh {
    pub vertices: *mut FFIVertex,
    pub vertex_count: u32,
    pub indices: *mut u32,
    pub index_count: u32,
}

/// Parse a STEP file and return a triangulated mesh.
/// Returns null on failure. Caller must free with `foxtrot_free_mesh`.
#[no_mangle]
pub extern "C" fn foxtrot_parse_step_file(path: *const c_char) -> *mut FFIMesh {
    if path.is_null() {
        return ptr::null_mut();
    }

    let c_str = unsafe { CStr::from_ptr(path) };
    let path_str = match c_str.to_str() {
        Ok(s) => s,
        Err(_) => return ptr::null_mut(),
    };

    let data = match std::fs::read(path_str) {
        Ok(d) => d,
        Err(_) => return ptr::null_mut(),
    };

    let flat = StepFile::strip_flatten(&data);
    let step = StepFile::parse(&flat);
    let (mesh, _stats) = triangulate(&step);

    if mesh.verts.is_empty() || mesh.triangles.is_empty() {
        return ptr::null_mut();
    }

    // Convert vertices
    let mut vertices: Vec<FFIVertex> = Vec::with_capacity(mesh.verts.len());
    for v in &mesh.verts {
        vertices.push(FFIVertex {
            x: v.pos.x as f32,
            y: v.pos.y as f32,
            z: v.pos.z as f32,
            nx: v.norm.x as f32,
            ny: v.norm.y as f32,
            nz: v.norm.z as f32,
            r: v.color.x as f32,
            g: v.color.y as f32,
            b: v.color.z as f32,
        });
    }

    // Convert triangle indices to flat array
    let mut indices: Vec<u32> = Vec::with_capacity(mesh.triangles.len() * 3);
    for t in &mesh.triangles {
        indices.push(t.verts.x);
        indices.push(t.verts.y);
        indices.push(t.verts.z);
    }

    let vertex_count = vertices.len() as u32;
    let index_count = indices.len() as u32;

    let vertices_ptr = vertices.as_mut_ptr();
    std::mem::forget(vertices);

    let indices_ptr = indices.as_mut_ptr();
    std::mem::forget(indices);

    let ffi_mesh = Box::new(FFIMesh {
        vertices: vertices_ptr,
        vertex_count,
        indices: indices_ptr,
        index_count,
    });

    Box::into_raw(ffi_mesh)
}

/// Free a mesh previously returned by `foxtrot_parse_step_file`.
#[no_mangle]
pub extern "C" fn foxtrot_free_mesh(mesh: *mut FFIMesh) {
    if mesh.is_null() {
        return;
    }
    unsafe {
        let m = Box::from_raw(mesh);
        if !m.vertices.is_null() {
            Vec::from_raw_parts(m.vertices, m.vertex_count as usize, m.vertex_count as usize);
        }
        if !m.indices.is_null() {
            Vec::from_raw_parts(m.indices, m.index_count as usize, m.index_count as usize);
        }
    }
}
