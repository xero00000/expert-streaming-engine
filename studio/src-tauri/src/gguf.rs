use std::{
    fs::File,
    io::{Read, Seek, SeekFrom},
    path::Path,
};

#[derive(Debug, Default, PartialEq, Eq)]
pub struct GgufMetadata {
    pub architecture: Option<String>,
    pub context_length: Option<u64>,
}

pub fn inspect(path: &Path) -> Result<GgufMetadata, String> {
    let mut file = File::open(path).map_err(|error| error.to_string())?;
    let mut magic = [0_u8; 4];
    file.read_exact(&mut magic)
        .map_err(|error| error.to_string())?;
    if &magic != b"GGUF" {
        return Err("not a GGUF file".into());
    }
    let version = read_u32(&mut file)?;
    if !(2..=3).contains(&version) {
        return Err(format!("unsupported GGUF version {version}"));
    }
    let _tensor_count = read_u64(&mut file)?;
    let metadata_count = read_u64(&mut file)?;
    if metadata_count > 1_000_000 {
        return Err("GGUF metadata count is unreasonable".into());
    }

    let mut metadata = GgufMetadata::default();
    for _ in 0..metadata_count {
        let key = read_string(&mut file)?;
        let value_type = read_u32(&mut file)?;
        if key == "general.architecture" && value_type == 8 {
            metadata.architecture = Some(read_string(&mut file)?);
        } else if key.ends_with(".context_length") {
            metadata.context_length = read_integer(&mut file, value_type)?;
        } else {
            skip_value(&mut file, value_type)?;
        }
        if metadata.architecture.is_some() && metadata.context_length.is_some() {
            break;
        }
    }
    Ok(metadata)
}

fn read_u32(reader: &mut impl Read) -> Result<u32, String> {
    let mut bytes = [0_u8; 4];
    reader
        .read_exact(&mut bytes)
        .map_err(|error| error.to_string())?;
    Ok(u32::from_le_bytes(bytes))
}

fn read_u64(reader: &mut impl Read) -> Result<u64, String> {
    let mut bytes = [0_u8; 8];
    reader
        .read_exact(&mut bytes)
        .map_err(|error| error.to_string())?;
    Ok(u64::from_le_bytes(bytes))
}

fn read_string(reader: &mut impl Read) -> Result<String, String> {
    let length = read_u64(reader)?;
    if length > 16 * 1024 * 1024 {
        return Err("GGUF string is too large".into());
    }
    let mut bytes = vec![0_u8; length as usize];
    reader
        .read_exact(&mut bytes)
        .map_err(|error| error.to_string())?;
    String::from_utf8(bytes).map_err(|error| error.to_string())
}

fn read_integer(reader: &mut (impl Read + Seek), value_type: u32) -> Result<Option<u64>, String> {
    let value = match value_type {
        0 => read_fixed::<1>(reader)
            .map(u8::from_le_bytes)
            .map(u64::from)?,
        2 => read_fixed::<2>(reader)
            .map(u16::from_le_bytes)
            .map(u64::from)?,
        4 => read_fixed::<4>(reader)
            .map(u32::from_le_bytes)
            .map(u64::from)?,
        10 => read_u64(reader)?,
        _ => {
            skip_value(reader, value_type)?;
            return Ok(None);
        }
    };
    Ok(Some(value))
}

fn read_fixed<const N: usize>(reader: &mut impl Read) -> Result<[u8; N], String> {
    let mut bytes = [0_u8; N];
    reader
        .read_exact(&mut bytes)
        .map_err(|error| error.to_string())?;
    Ok(bytes)
}

fn skip_value(reader: &mut (impl Read + Seek), value_type: u32) -> Result<(), String> {
    let fixed_size = match value_type {
        0 | 1 | 7 => Some(1),
        2 | 3 => Some(2),
        4..=6 => Some(4),
        10..=12 => Some(8),
        _ => None,
    };
    if let Some(size) = fixed_size {
        reader
            .seek(SeekFrom::Current(size))
            .map_err(|error| error.to_string())?;
        return Ok(());
    }
    match value_type {
        8 => {
            let length = read_u64(reader)?;
            if length > i64::MAX as u64 {
                return Err("GGUF string length overflows seek".into());
            }
            reader
                .seek(SeekFrom::Current(length as i64))
                .map_err(|error| error.to_string())?;
        }
        9 => {
            let element_type = read_u32(reader)?;
            let count = read_u64(reader)?;
            if count > 100_000_000 {
                return Err("GGUF array count is unreasonable".into());
            }
            for _ in 0..count {
                skip_value(reader, element_type)?;
            }
        }
        _ => return Err(format!("unknown GGUF metadata type {value_type}")),
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_real_fixture_when_configured() {
        let Ok(path) = std::env::var("ESE_STUDIO_E2E_MODEL") else {
            return;
        };
        let metadata = inspect(Path::new(&path)).expect("fixture metadata should parse");
        assert!(metadata.architecture.is_some());
        assert!(metadata.context_length.is_some_and(|context| context > 0));
    }
}
