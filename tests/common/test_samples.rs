//! Utilities for accessing test samples.

use std::path::PathBuf;

/// Returns the path to a test sample file.
pub fn get_sample_path(filename: &str) -> PathBuf {
    let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.push("tests");
    path.push("samples");
    path.push(filename);

    if !path.exists() {
        panic!("Test sample not found: {}", path.display());
    }

    path
}
