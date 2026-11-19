use std::ffi::CString;

fn parse(bytes: &[u8]) -> bool {
    let path = std::env::temp_dir().join(format!("ffi_test_{}.stp", bytes.len()));
    std::fs::write(&path, bytes).unwrap();
    let c = CString::new(path.to_str().unwrap()).unwrap();
    let m = foxtrot_ffi::foxtrot_parse_step_file(c.as_ptr());
    let ok = !m.is_null();
    foxtrot_ffi::foxtrot_free_mesh(m);
    ok
}

/// Malformed input must yield a null mesh, never abort the process.
#[test]
fn malformed_inputs_do_not_abort() {
    let cases: [&[u8]; 4] = [
        b"",
        b"ISO-10303-21;\nHEADER;\nENDSEC;\nDATA;\n#1=CARTESIAN_POINT('',(0.,0.,",
        b"ISO-10303-21;\nDATA;\n#1=ADVANCED_FACE('',(#2),#3,.T.);\n#2=FACE_BOUND('',#9,.T.);\nENDSEC;\nEND-ISO-10303-21;",
        &[0xff, 0xfe, 0x00, 0x01, 0x80, 0x23, 0x31, 0x3d],
    ];
    for c in cases {
        assert!(!parse(c));
    }
}
