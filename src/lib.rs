//! # MP4 Audio Extractor
//!
//! A library for extracting audio streams from MP4 container files.
//!
//! ## Basic Usage
//!
//! ```rust,no_run
//! use std::fs::File;
//! use std::io::BufReader;
//! use mp4_audio_extractor::Extractor;
//!
//! fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let file = File::open("video.mp4")?;
//!     let size = file.metadata().unwrap().len();
//!     let reader = BufReader::new(file);
//!     
//!     let extractor = Extractor::new();
//!     let audio_data = extractor.extract_audio(reader, size)?;
//!     
//!     // Do something with the audio data
//!     println!("Extracted {} bytes of audio", audio_data.len());
//!     
//!     Ok(())
//! }
//! ```

pub mod error;
pub mod extractor;

pub use error::Error;
pub use extractor::Extractor;

/// Re-export of the Result type specialized for this library
pub type Result<T> = std::result::Result<T, Error>;