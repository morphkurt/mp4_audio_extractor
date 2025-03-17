# MP4 Audio Extractor

## Overview
This library provides core functionality for extracting audio from MP4 files. It allows users to parse MP4 files, identify audio tracks, and extract raw audio data efficiently.

## Features
- Extracts AAC audio from MP4 files
- Configurable buffer size and output format
- Supports metadata inclusion
- Optimized for performance with efficient memory management

## Installation
To use this library, add the required dependencies in your Cargo.toml:

```toml
[dependencies]
mp4 = "*"
log = "*"
```

## Usage
```rust
use std::fs::File;
use std::io::{BufReader, Read, Seek};
use mp4_audio_extractor::Extractor;

fn main() {
    let file = File::open("sample.mp4").expect("Failed to open file");
    let metadata = file.metadata().expect("Failed to get metadata");
    let file_size = metadata.len();
    let reader = BufReader::new(file);

    let extractor = Extractor::new();
    let audio_data = extractor.extract_audio(reader, file_size).expect("Failed to extract audio");

    // Process extracted audio data
    println!("Extracted {} bytes of audio", audio_data.len());
}
```

## Error Handling
The library returns detailed errors if extraction fails, including:
- MP4 file parsing errors
- No audio stream found
- Sample reading or writing failures

## Testing
Run tests with:
```sh
cargo test
```

## License
This project is licensed under the MIT License.

