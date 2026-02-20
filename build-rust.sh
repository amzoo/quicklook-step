#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")/foxtrot-ffi"

echo "Building for aarch64-apple-darwin..."
cargo build --release --target aarch64-apple-darwin

echo "Building for x86_64-apple-darwin..."
cargo build --release --target x86_64-apple-darwin

echo "Creating universal binary..."
mkdir -p target/universal
lipo -create \
    target/aarch64-apple-darwin/release/libfoxtrot_ffi.a \
    target/x86_64-apple-darwin/release/libfoxtrot_ffi.a \
    -output target/universal/libfoxtrot_ffi.a

echo "Generating C header..."
mkdir -p ../include
cbindgen --config cbindgen.toml --crate foxtrot-ffi --output ../include/foxtrot_ffi.h

echo "Done!"
echo "  Library: foxtrot-ffi/target/universal/libfoxtrot_ffi.a"
echo "  Header:  include/foxtrot_ffi.h"
