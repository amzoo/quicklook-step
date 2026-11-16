# QuickLook STEP

A macOS QuickLook extension for previewing STEP/STP CAD files. Press Space on any `.step` or `.stp` file in Finder to get an interactive 3D preview.

Built with [foxtrot](https://github.com/Formlabs/foxtrot) (Rust STEP parser + triangulator) and SceneKit.

## Features

- **Quick Look preview** — interactive 3D view with rotation/zoom
- **Finder thumbnails** — rendered 3D thumbnails for STEP files
- Universal binary (Apple Silicon + Intel)

## Install

1. Download `QuickLookSTEP.zip` from the [latest release](https://github.com/amzoo/quicklook-step/releases/latest)
2. Unzip and move `QuickLookSTEP.app` to `/Applications`
3. Right-click the app → **Open**, then click **Open** in the dialog (required once since the app is not notarized)
4. Press Space on any `.step` or `.stp` file in Finder to preview it

## Building from Source

### Requirements

- macOS 13.0+
- Rust toolchain (`rustup`)
- Xcode 14+
- `cbindgen` (`cargo install cbindgen`)

### Steps

1. Build the Rust static library:

```bash
bash build-rust.sh
```

`build-rust.sh` automatically installs the `aarch64-apple-darwin` and `x86_64-apple-darwin` Rust targets via `rustup` (needed for the universal binary).

2. Build the app and extensions in Xcode:

```bash
xcodebuild -project QuickLookSTEP/QuickLookSTEP.xcodeproj -scheme QuickLookSTEP build
```

Or open `QuickLookSTEP/QuickLookSTEP.xcodeproj` in Xcode and build from there.

## Project Structure

```
foxtrot/          — STEP parser & triangulator (from Formlabs/foxtrot)
foxtrot-ffi/      — Rust FFI crate (C-compatible static library)
include/          — Auto-generated C header
QuickLookSTEP/
  QuickLookSTEP/      — Host app (SwiftUI)
  PreviewExtension/   — QLPreviewingController (interactive 3D preview)
  Shared/             — STEPSceneBuilder (FFI mesh → SceneKit)
```

## License

[CC BY-NC 4.0](LICENSE) — free to use and modify for non-commercial purposes with attribution.
