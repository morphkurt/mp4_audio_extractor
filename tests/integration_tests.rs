//! Integration tests for the MP4 audio extractor.

mod common;

use common::test_samples;
use mp4_audio_extractor::{Error, Extractor};
use std::fs::File;
use std::io::BufReader;

#[test]
fn test_extract_audio_from_valid_mp4() {
    let file_path = test_samples::get_sample_path("sample1.mp4");
    let file = File::open(file_path).unwrap();
    let size = file.metadata().unwrap().len();
    let reader = BufReader::new(file);

    let extractor = Extractor::new();
    let result: Result<Vec<u8>, Error> = extractor.extract_audio(reader, size);

    assert!(
        result.is_ok(),
        "Failed to extract audio: {:?}",
        result.err()
    );

    let audio_data = result.unwrap();

    assert!(!audio_data.is_empty(), "Extracted audio data is empty");
}

#[test]
fn test_extract_audio_from_opus_mp4() {
    env_logger::init();

    let file_path = test_samples::get_sample_path("sample2.mp4");
    let file = File::open(file_path).unwrap();
    let size = file.metadata().unwrap().len();
    let reader = BufReader::new(file);

    let extractor = Extractor::new();
    let result: Result<Vec<u8>, Error> = extractor.extract_audio(reader, size);

    assert!(
        result.is_ok(),
        "Failed to extract audio: {:?}",
        result.err()
    );

    let audio_data = result.unwrap();

    assert!(!audio_data.is_empty(), "Extracted audio data is empty");
}
