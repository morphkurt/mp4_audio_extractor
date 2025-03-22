//! Core functionality for extracting audio from MP4 files.

use mp4::{AacConfig, MediaConfig, MediaType, Mp4Config, Mp4Writer, TrackConfig};
use mp4::{AvcConfig, HevcConfig, OpusConfig, Vp9Config};
use std::collections::HashMap;
use std::io::{BufReader, Cursor, Read, Seek};
use std::vec;

use crate::Error;
use crate::Result;

/// Configuration options for the audio extraction process.
#[derive(Debug, Clone)]
pub struct ExtractorConfig {
    /// Whether to include metadata in the extracted audio
    pub include_metadata: bool,

    /// Maximum buffer size for processing (in bytes)
    pub buffer_size: usize,

    /// Output format (default is "mp4")
    pub output_format: String,
}

/// The main extractor struct responsible for extracting audio from MP4 files.
#[derive(Debug)]
pub struct Extractor {}

#[derive(Debug)]
struct SampleConfig {
    start_offset: u64,
}

impl Extractor {
    /// Creates a new extractor with default configuration.
    pub fn new() -> Self {
        Self {}
    }

    pub fn trim<R: Read + Seek>(
        &self,
        reader: BufReader<R>,
        size: u64,
        start_time_us: u64,
        end_time_us: u64,
    ) -> Result<Vec<u8>> {
        log::info!(
            "Starting audio extraction from MP4 file of size {} bytes",
            size
        );

        // Parse the original MP4 file
        let mut mp4: mp4::Mp4Reader<BufReader<R>> = mp4::Mp4Reader::read_header(reader, size)
            .map_err(|e| Error::ParsingError(format!("Failed to parse MP4 header: {}", e)))?;

        // Create output buffer with a reasonable initial capacity
        let estimated_size = (size).max(1024 * 1024);
        let mut output_buffer = Vec::with_capacity(estimated_size as usize);
        let output_cursor = Cursor::new(&mut output_buffer);

        // Create MP4 configuration
        let config = Mp4Config {
            major_brand: *mp4.major_brand(),
            minor_version: mp4.minor_version(),
            compatible_brands: mp4.compatible_brands().to_vec(),
            timescale: mp4.timescale(),
        };
        // Create MP4Writer
        let mut writer = Mp4Writer::write_start(output_cursor, &config)
            .map_err(|e| Error::ParsingError(format!("Failed to create MP4 writer: {}", e)))?;

        // Find and configure audio tracks
        let track_map: HashMap<u32, u32> = self.configure_tracks(&mut mp4, &mut writer, false)?;

        let sample_config = self.process_samples(
            start_time_us,
            end_time_us,
            &mut mp4,
            &mut writer,
            &track_map,
        )?;

        for (k, v) in sample_config {
            let index = track_map.get(&k).unwrap();
            let _ = writer.update_offset(*index - 1, v.start_offset, end_time_us - start_time_us);
        }

        writer
            .write_end()
            .map_err(|e| Error::ParsingError(format!("Failed to finalize MP4 file: {}", e)))?;

        log::info!(
            "Successfully created MP4 data in memory with extracted audio: {}",
            output_buffer.len(),
        );

        Ok(output_buffer)
    }

    /// Extracts audio data from an MP4 file using a BufReader.
    ///
    /// Returns the raw audio data as a Vec<u8>.
    ///
    /// # Arguments
    ///
    /// * `reader` - A BufReader containing the MP4 file data
    /// * `size` - The size of the MP4 file in bytes
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The MP4 file cannot be parsed
    /// - No audio stream is found
    /// - Sample reading or writing fails
    pub fn extract_audio<R: Read + Seek>(
        &self,
        reader: BufReader<R>,
        size: u64,
    ) -> Result<Vec<u8>> {
        log::info!(
            "Starting audio extraction from MP4 file of size {} bytes",
            size
        );

        // Parse the original MP4 file
        let mut mp4: mp4::Mp4Reader<BufReader<R>> = mp4::Mp4Reader::read_header(reader, size)
            .map_err(|e| Error::ParsingError(format!("Failed to parse MP4 header: {}", e)))?;

        // Find the first audio track
        let trak = mp4
            .moov
            .traks
            .iter()
            .find(|track| {
                if let Ok(handler_type) = str::parse::<mp4::FourCC>("soun") {
                    track.mdia.hdlr.handler_type == handler_type
                } else {
                    false
                }
            })
            .ok_or(Error::NoAudioStream)?;

        let track_id = trak.tkhd.track_id;
        log::debug!("Found audio track with ID {}", track_id);

        // Get total samples for this track
        let total_samples = mp4
            .sample_count(track_id)
            .map_err(|e| Error::ParsingError(format!("Failed to get sample count: {}", e)))?;

        log::debug!("Track has {} samples", total_samples);

        // Create output buffer with a reasonable initial capacity
        let estimated_size = (size / 4).max(1024 * 1024);
        let mut output_buffer = Vec::with_capacity(estimated_size as usize);
        let output_cursor = Cursor::new(&mut output_buffer);

        // Create MP4 configuration
        let config = self.create_mp4_config(&trak.mdia.mdhd.timescale);

        // Create MP4Writer
        let mut writer = Mp4Writer::write_start(output_cursor, &config)
            .map_err(|e| Error::ParsingError(format!("Failed to create MP4 writer: {}", e)))?;

        // Find and configure audio tracks
        _ = self.configure_tracks(&mut mp4, &mut writer, true)?;

        // Process all samples
        self.process_audio_samples(track_id, total_samples, &mut mp4, &mut writer)?;

        // Finalize the MP4 file
        writer
            .write_end()
            .map_err(|e| Error::ParsingError(format!("Failed to finalize MP4 file: {}", e)))?;

        log::info!(
            "Successfully created MP4 data in memory with extracted audio: {}",
            output_buffer.len(),
        );

        Ok(output_buffer)
    }

    /// Creates the MP4 configuration with appropriate brands and timescale.
    fn create_mp4_config(&self, timescale: &u32) -> Mp4Config {
        Mp4Config {
            major_brand: str::parse("isom").unwrap_or_default(),
            minor_version: 512,
            compatible_brands: vec![
                str::parse("isom").unwrap_or_default(),
                str::parse("iso2").unwrap_or_default(),
                str::parse("mp41").unwrap_or_default(),
            ],
            timescale: *timescale,
        }
    }

    /// Configures audio tracks in the MP4 writer.
    fn configure_tracks<R: Read + Seek>(
        &self,
        mp4: &mut mp4::Mp4Reader<R>,
        writer: &mut Mp4Writer<Cursor<&mut Vec<u8>>>,
        audio_only: bool,
    ) -> Result<HashMap<u32, u32>> {
        let mut map: HashMap<u32, u32> = HashMap::new();
        for (track_id, track) in mp4.tracks() {
            match track.media_type() {
                Ok(MediaType::AAC) => {
                    // Get audio profile or return error if not available
                    let audio_profile = match track.audio_profile() {
                        Ok(profile) => profile,
                        Err(_) => {
                            return Err(Error::ParsingError(
                                "Failed to get audio profile".to_string(),
                            ))
                        }
                    };

                    // Get frequency index or return error if not available
                    let freq_index = match track.sample_freq_index() {
                        Ok(index) => index,
                        Err(_) => {
                            return Err(Error::ParsingError(
                                "Failed to get frequency index".to_string(),
                            ))
                        }
                    };

                    // Get channel configuration or return error if not available
                    let chan_conf = match track.channel_config() {
                        Ok(conf) => conf,
                        Err(_) => {
                            return Err(Error::ParsingError(
                                "Failed to get channel config".to_string(),
                            ))
                        }
                    };

                    let media_conf = MediaConfig::AacConfig(AacConfig {
                        bitrate: track.bitrate(),
                        profile: audio_profile,
                        freq_index: freq_index,
                        chan_conf: chan_conf,
                    });

                    // Get track type or return error if not available
                    let track_type = match track.track_type() {
                        Ok(t_type) => t_type,
                        Err(_) => {
                            return Err(Error::ParsingError("Failed to get track type".to_string()))
                        }
                    };

                    let track_conf = TrackConfig {
                        track_type: track_type,
                        timescale: track.timescale(),
                        language: track.language().to_string(),
                        media_conf,
                    };

                    let id = writer.add_track(&track_conf).map_err(|e| {
                        Error::ParsingError(format!("Failed to add track {}: {}", track_id, e))
                    })?;

                    map.insert(*track_id, id);

                    log::debug!("Added AAC audio track {} to output", track_id);
                }
                Ok(MediaType::H264) => {
                    // Get frequency index or return error if not available
                    let seq_param_set = match track.sequence_parameter_set() {
                        Ok(seq_param_set) => seq_param_set,
                        Err(_) => {
                            return Err(Error::ParsingError(
                                "Failed to get frequency index".to_string(),
                            ))
                        }
                    };

                    // Get channel configuration or return error if not available
                    let pic_param_set = match track.picture_parameter_set() {
                        Ok(pic_param_set) => pic_param_set,
                        Err(_) => {
                            return Err(Error::ParsingError(
                                "Failed to get channel config".to_string(),
                            ))
                        }
                    };

                    let media_conf = MediaConfig::AvcConfig(AvcConfig {
                        width: track.width(),
                        height: track.height(),
                        seq_param_set: seq_param_set.to_vec(),
                        pic_param_set: pic_param_set.to_vec(),
                    });

                    // Get track type or return error if not available
                    let track_type = match track.track_type() {
                        Ok(t_type) => t_type,
                        Err(_) => {
                            return Err(Error::ParsingError("Failed to get track type".to_string()))
                        }
                    };

                    let track_conf = TrackConfig {
                        track_type: track_type,
                        timescale: track.timescale(),
                        language: track.language().to_string(),
                        media_conf,
                    };

                    if !audio_only {
                        let id = writer.add_track(&track_conf).map_err(|e| {
                            Error::ParsingError(format!("Failed to add track {}: {}", track_id, e))
                        })?;

                        map.insert(*track_id, id);
                        log::debug!("Added AVC audio track {} to output", track_id);
                    }
                }
                Ok(MediaType::H265) => {
                    let media_conf = MediaConfig::HevcConfig(HevcConfig {
                        width: track.width(),
                        height: track.height(),
                    });

                    // Get track type or return error if not available
                    let track_type = match track.track_type() {
                        Ok(t_type) => t_type,
                        Err(_) => {
                            return Err(Error::ParsingError("Failed to get track type".to_string()))
                        }
                    };

                    let track_conf = TrackConfig {
                        track_type: track_type,
                        timescale: track.timescale(),
                        language: track.language().to_string(),
                        media_conf,
                    };

                    if !audio_only {
                        let id = writer.add_track(&track_conf).map_err(|e| {
                            Error::ParsingError(format!("Failed to add track {}: {}", track_id, e))
                        })?;

                        map.insert(*track_id, id);
                        log::debug!("Added HEVC audio track {} to output", track_id);
                    }
                }
                Ok(MediaType::VP9) => {
                    let media_conf = MediaConfig::Vp9Config(Vp9Config {
                        width: track.width(),
                        height: track.height(),
                    });

                    // Get track type or return error if not available
                    let track_type = match track.track_type() {
                        Ok(t_type) => t_type,
                        Err(_) => {
                            return Err(Error::ParsingError("Failed to get track type".to_string()))
                        }
                    };

                    let track_conf = TrackConfig {
                        track_type: track_type,
                        timescale: track.timescale(),
                        language: track.language().to_string(),
                        media_conf,
                    };

                    if !audio_only {
                        let id = writer.add_track(&track_conf).map_err(|e| {
                            Error::ParsingError(format!("Failed to add track {}: {}", track_id, e))
                        })?;

                        map.insert(*track_id, id);
                        log::debug!("Added VP9 audio track {} to output", track_id);
                    }
                }
                Ok(MediaType::OPUS) => {
                    // Get audio profile or return error if not available

                    // Get frequency index or return error if not available
                    let freq_index = match track.sample_freq_index() {
                        Ok(index) => index,
                        Err(_) => {
                            return Err(Error::ParsingError(
                                "Failed to get frequency index".to_string(),
                            ))
                        }
                    };

                    // Get channel configuration or return error if not available
                    let chan_conf = match track.channel_config() {
                        Ok(conf) => conf,
                        Err(_) => {
                            return Err(Error::ParsingError(
                                "Failed to get channel config".to_string(),
                            ))
                        }
                    };

                    let media_conf = MediaConfig::OpusConfig(OpusConfig {
                        bitrate: track.bitrate(),
                        freq_index,
                        chan_conf,
                        pre_skip: 0,
                    });

                    // Get track type or return error if not available
                    let track_type = match track.track_type() {
                        Ok(t_type) => t_type,
                        Err(_) => {
                            return Err(Error::ParsingError("Failed to get track type".to_string()))
                        }
                    };

                    let track_conf = TrackConfig {
                        track_type: track_type,
                        timescale: track.timescale(),
                        language: track.language().to_string(),
                        media_conf,
                    };

                    let id = writer.add_track(&track_conf).map_err(|e| {
                        Error::ParsingError(format!("Failed to add track {}: {}", track_id, e))
                    })?;

                    map.insert(*track_id, id);
                    log::debug!("Added OPUS audio track {} to output", track_id);
                }
                _ => {
                    log::debug!("Skipping unknown track {}", track_id);
                    continue;
                }
            }
        }

        Ok(map)
    }

    /// Processes all samples from the input track to the output.
    fn process_samples<R: Read + Seek>(
        &self,
        start_time_us: u64,
        end_time_us: u64,
        mp4: &mut mp4::Mp4Reader<R>,
        writer: &mut Mp4Writer<Cursor<&mut Vec<u8>>>,
        map: &HashMap<u32, u32>,
    ) -> Result<HashMap<u32, SampleConfig>> {
        let mut sample_config_map: HashMap<u32, SampleConfig> = HashMap::new();
        let track_ids = mp4
            .moov
            .traks
            .iter()
            .map(|t| (t.tkhd.track_id, t.mdia.mdhd.timescale))
            .collect::<Vec<(u32, u32)>>();
        for (track_id, timescale) in track_ids {
            let start_time = (start_time_us * timescale as u64) / 1_000_000;
            let end_time = (end_time_us * timescale as u64) / 1_000_000;
            let mut curr_time: u64 = 0;
            let mut offset: u64 = 0;
            let mut buffered_samples: Vec<u32> = Vec::new();
            for sample_id in 1..=mp4.sample_count(track_id).unwrap_or(0) {
                match mp4.read_sample_metadata(track_id, sample_id) {
                    Ok(Some(sample)) => {
                        curr_time += sample.duration as u64;
                        if curr_time > end_time {
                            continue;
                        }
                        if curr_time < start_time {
                            if sample.is_sync {
                                offset = start_time.saturating_sub(curr_time);
                                buffered_samples.clear();
                            }
                            buffered_samples.push(sample_id);
                            continue;
                        }
                        buffered_samples.push(sample_id);
                    }
                    Ok(None) => {
                        log::warn!("Sample {} not found, stopping extraction", sample_id);
                        break;
                    }
                    Err(e) => {
                        return Err(Error::ParsingError(format!(
                            "Failed to read sample {}: {}",
                            sample_id, e
                        )));
                    }
                }
            }
            for buffred_id in &buffered_samples {
                let s = mp4.read_sample(track_id, *buffred_id).unwrap().unwrap();
                writer
                    .write_sample(*map.get(&track_id).unwrap(), &s)
                    .map_err(|e| {
                        Error::ParsingError(format!("Failed to write sample {}: {}", buffred_id, e))
                    })?;
                //     println!("writting buffered sample id {}, track id {}, is_key {}, render {}",buffred_id, track_id, s.is_sync, s.rendering_offset);
            }
            sample_config_map.insert(
                track_id,
                SampleConfig {
                    start_offset: offset,
                },
            );
        }
        Ok(sample_config_map)
    }

    /// Processes all samples from the input track to the output.
    fn process_audio_samples<R: Read + Seek>(
        &self,
        track_id: u32,
        total_samples: u32,
        mp4: &mut mp4::Mp4Reader<R>,
        writer: &mut Mp4Writer<Cursor<&mut Vec<u8>>>,
    ) -> Result<()> {
        let mut processed_samples = 0;
        let log_interval = (total_samples / 10).max(1); // Log progress every 10%
        for sample_id in 1..=total_samples {
            match mp4.read_sample(track_id, sample_id) {
                Ok(Some(sample)) => {
                    // Write the sample data to the output MP4
                    writer.write_sample(1, &sample).map_err(|e| {
                        Error::ParsingError(format!("Failed to write sample {}: {}", sample_id, e))
                    })?;

                    processed_samples += 1;

                    // Log progress periodically
                    if processed_samples % log_interval == 0 || processed_samples == total_samples {
                        let progress = (processed_samples as f64 / total_samples as f64) * 100.0;
                        log::info!(
                            "Processing progress: {:.1}% ({}/{})",
                            progress,
                            processed_samples,
                            total_samples
                        );
                    }
                }
                Ok(None) => {
                    log::warn!("Sample {} not found, stopping extraction", sample_id);
                    break;
                }
                Err(e) => {
                    return Err(Error::ParsingError(format!(
                        "Failed to read sample {}: {}",
                        sample_id, e
                    )));
                }
            }
        }

        log::info!(
            "Successfully processed {} out of {} samples",
            processed_samples,
            total_samples
        );
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::File;
    use std::io::BufReader;
    use std::path::Path;

    #[test]
    fn test_extract_audio_from_file() -> Result<()> {
        // Test setup
        let test_file_path = Path::new("test_data/sample.mp4");
        if !test_file_path.exists() {
            println!("Test file not found, skipping test");
            return Ok(());
        }

        // Get file metadata once
        let metadata = std::fs::metadata(test_file_path)?;
        let file_size = metadata.len();

        // Read the original MP4 file and extract its metadata
        let file = File::open(test_file_path)?;
        let reader = BufReader::new(file);
        let mut original_mp4 = mp4::Mp4Reader::read_header(reader, file_size)
            .map_err(|e| Error::ParsingError(format!("Failed to parse original MP4: {}", e)))?;
        let original_duration = original_mp4.duration();

        // Find the audio track in original file
        let original_audio_track = original_mp4
            .moov
            .traks
            .iter()
            .find(|track| {
                if let Ok(handler_type) = str::parse::<mp4::FourCC>("soun") {
                    track.mdia.hdlr.handler_type == handler_type
                } else {
                    false
                }
            })
            .ok_or(Error::NoAudioStream)?;

        let original_track_id = original_audio_track.tkhd.track_id;
        let original_sample_count = original_mp4.sample_count(original_track_id).map_err(|e| {
            Error::ParsingError(format!("Failed to get original sample count: {}", e))
        })?;

        // Get audio properties of original file
        let original_track =
            original_mp4
                .tracks()
                .get(&original_track_id)
                .ok_or(Error::ParsingError(format!(
                    "Original track {} not found",
                    original_track_id
                )))?;

        let original_bitrate = original_track.bitrate();
        let original_timescale = original_track.timescale();

        // Create a new file handle for extraction
        let file = File::open(test_file_path)?;
        let reader = BufReader::new(file);

        // Perform the extraction
        let extractor = Extractor::new();
        let extracted_data = extractor.extract_audio(reader, file_size)?;

        // Verify the extracted data
        let extracted_reader = BufReader::new(Cursor::new(&extracted_data));
        let mut extracted_mp4 =
            mp4::Mp4Reader::read_header(extracted_reader, extracted_data.len() as u64).map_err(
                |e| Error::ParsingError(format!("Failed to parse extracted MP4: {}", e)),
            )?;

        // Compare durations
        assert_eq!(
            original_duration,
            extracted_mp4.duration(),
            "Duration mismatch"
        );

        // Check for audio track presence
        let extracted_audio_track = extracted_mp4
            .moov
            .traks
            .iter()
            .find(|track| {
                if let Ok(handler_type) = str::parse::<mp4::FourCC>("soun") {
                    track.mdia.hdlr.handler_type == handler_type
                } else {
                    false
                }
            })
            .ok_or(Error::ParsingError(
                "No audio track found in extracted MP4".to_string(),
            ))?;

        // Track should be present
        assert!(
            extracted_audio_track.tkhd.track_id > 0,
            "Invalid track ID in extracted file"
        );

        // Verify sample counts
        let extracted_track_id = extracted_audio_track.tkhd.track_id;
        let extracted_sample_count =
            extracted_mp4
                .sample_count(extracted_track_id)
                .map_err(|e| {
                    Error::ParsingError(format!("Failed to get extracted sample count: {}", e))
                })?;

        assert_eq!(
            original_sample_count, extracted_sample_count,
            "Sample count mismatch: original={}, extracted={}",
            original_sample_count, extracted_sample_count
        );

        // Compare audio quality metrics
        let extracted_track =
            extracted_mp4
                .tracks()
                .get(&extracted_track_id)
                .ok_or(Error::ParsingError(format!(
                    "Extracted track {} not found",
                    extracted_track_id
                )))?;

        // Check timescale (should be preserved)
        assert_eq!(
            original_timescale,
            extracted_track.timescale(),
            "Timescale mismatch: original={}, extracted={}",
            original_timescale,
            extracted_track.timescale()
        );

        // Check media type
        assert_eq!(
            extracted_track.media_type().unwrap(),
            MediaType::AAC,
            "Extracted track should be AAC audio"
        );

        // Compare audio configuration if possible
        if let Ok(original_profile) = original_track.audio_profile() {
            if let Ok(extracted_profile) = extracted_track.audio_profile() {
                assert_eq!(
                    original_profile, extracted_profile,
                    "Audio profile mismatch: original={:?}, extracted={:?}",
                    original_profile, extracted_profile
                );
            }
        }

        if let Ok(original_freq) = original_track.sample_freq_index() {
            if let Ok(extracted_freq) = extracted_track.sample_freq_index() {
                assert_eq!(
                    original_freq,
                    extracted_freq,
                    "Sample frequency index mismatch: original={}, extracted={}",
                    original_freq.freq(),
                    extracted_freq.freq()
                );
            }
        }

        if let Ok(original_channels) = original_track.channel_config() {
            if let Ok(extracted_channels) = extracted_track.channel_config() {
                assert_eq!(
                    original_channels, extracted_channels,
                    "Channel configuration mismatch: original={}, extracted={}",
                    original_channels, extracted_channels
                );
            }
        }

        // Verify the bitrate is reasonable (might not be exactly the same)
        let extracted_bitrate = extracted_track.bitrate();
        assert!(
            extracted_bitrate > 0,
            "Extracted bitrate should be positive"
        );

        // Typical tolerance for bitrate (within 10% of original)
        let bitrate_tolerance = original_bitrate as f64 * 0.1;
        let bitrate_diff = (original_bitrate as f64 - extracted_bitrate as f64).abs();
        assert!(
            bitrate_diff <= bitrate_tolerance,
            "Bitrate differs too much: original={}, extracted={}, tolerance={}",
            original_bitrate,
            extracted_bitrate,
            bitrate_tolerance
        );

        // Optional: Sample a few frames and compare their sizes (to check for corruption)
        if original_sample_count > 10 {
            for i in 1..=10 {
                let sample_id = i * (original_sample_count / 10);

                let original_sample = original_mp4
                    .read_sample(original_track_id, sample_id)
                    .map_err(|e| {
                        Error::ParsingError(format!(
                            "Failed to read original sample {}: {}",
                            sample_id, e
                        ))
                    })?;

                let extracted_sample = extracted_mp4
                    .read_sample(extracted_track_id, sample_id)
                    .map_err(|e| {
                        Error::ParsingError(format!(
                            "Failed to read extracted sample {}: {}",
                            sample_id, e
                        ))
                    })?;

                if let (Some(orig), Some(extr)) = (original_sample, extracted_sample) {
                    // The samples might not be identical, but they should be close in size
                    let size_ratio = orig.bytes.len() as f64 / extr.bytes.len() as f64;
                    assert!(size_ratio > 0.5 && size_ratio < 2.0,
                            "Sample size ratio out of bounds: {} (original: {} bytes, extracted: {} bytes)",
                            size_ratio, orig.bytes.len(), extr.bytes.len());
                }
            }
        }

        Ok(())
    }
}
