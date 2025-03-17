//! Error handling for the MP4 audio extractor.

use std::fmt;
use std::io;
use std::error::Error as StdError;

/// Errors that can occur during MP4 audio extraction.
#[derive(Debug)]
pub enum Error {
    /// An I/O error occurred.
    Io(io::Error),
    
    /// The input file is not a valid MP4 file.
    InvalidMp4(String),
    
    /// No audio stream was found in the MP4 file.
    NoAudioStream,
    
    /// The audio stream uses an unsupported codec.
    UnsupportedCodec(String),
    
    /// An error occurred while parsing the MP4 file structure.
    ParsingError(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Io(err) => write!(f, "I/O error: {}", err),
            Error::InvalidMp4(msg) => write!(f, "Invalid MP4 file: {}", msg),
            Error::NoAudioStream => write!(f, "No audio stream found"),
            Error::UnsupportedCodec(codec) => write!(f, "Unsupported audio codec: {}", codec),
            Error::ParsingError(msg) => write!(f, "MP4 parsing error: {}", msg),
        }
    }
}

impl StdError for Error {
    fn source(&self) -> Option<&(dyn StdError + 'static)> {
        match self {
            Error::Io(err) => Some(err),
            _ => None,
        }
    }
}

impl From<io::Error> for Error {
    fn from(err: io::Error) -> Self {
        Error::Io(err)
    }
}