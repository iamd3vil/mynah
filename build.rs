fn main() {
    // transcribe-cpp-sys 0.1.3 compiles CBLAS calls into its static lib but
    // doesn't emit a BLAS link directive on Linux; link the system CBLAS.
    // (Worth reporting upstream — remove once the sys crate handles it.)
    println!("cargo:rustc-link-lib=dylib=cblas");
}
