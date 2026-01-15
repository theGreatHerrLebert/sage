//! Fragment ion intensity prediction support
//!
//! This module provides data structures and traits for integrating predicted
//! fragment ion intensities into the database search scoring.
//!
//! ## Workflow
//!
//! 1. Python (via sagepy) accesses `IndexedDatabase.peptides` to get sequences
//! 2. Python generates intensity predictions for all peptides
//! 3. Python writes predictions to a binary file (in peptide index order)
//! 4. Sage loads the file via `PredictedIntensityStore` for use during scoring

use crate::ion_series::Kind;
use crate::peptide::Peptide;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{self, BufReader, Read, Write};
use std::path::Path;

/// Predicted intensities for all fragments of a peptide.
///
/// Stored as a flattened tensor with dimensions `[ion_type][position][charge]`.
/// Intensities are normalized values typically in `[0.0, 1.0]`.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct PeptideFragmentIntensities {
    /// Peptide length (determines position dimension: peptide_len - 1 fragments)
    pub peptide_len: usize,
    /// Maximum charge state stored (1-indexed)
    pub max_charge: u8,
    /// Ion types included (typically B, Y)
    pub ion_kinds: Vec<Kind>,
    /// Flattened intensity array.
    /// Index = kind_idx * (peptide_len - 1) * max_charge + position * max_charge + (charge - 1)
    pub intensities: Vec<f32>,
}

impl PeptideFragmentIntensities {
    /// Create a new instance for a peptide with uniform intensities (all 1.0)
    pub fn uniform(peptide_len: usize, max_charge: u8, ion_kinds: Vec<Kind>) -> Self {
        let num_positions = peptide_len.saturating_sub(1);
        let total_elements = ion_kinds.len() * num_positions * max_charge as usize;
        Self {
            peptide_len,
            max_charge,
            ion_kinds,
            intensities: vec![1.0; total_elements],
        }
    }

    /// Create from a raw 3D tensor `[ion_type][position][charge]`
    ///
    /// The tensor should have dimensions:
    /// - First: ion_kinds.len()
    /// - Second: peptide_len - 1 (number of fragment positions)
    /// - Third: max_charge
    pub fn from_tensor(
        tensor: &[Vec<Vec<f32>>],
        peptide_len: usize,
        max_charge: u8,
        ion_kinds: Vec<Kind>,
    ) -> Self {
        let num_positions = peptide_len.saturating_sub(1);
        let mut intensities = Vec::with_capacity(ion_kinds.len() * num_positions * max_charge as usize);

        for kind_data in tensor.iter().take(ion_kinds.len()) {
            for position_data in kind_data.iter().take(num_positions) {
                for charge_idx in 0..max_charge as usize {
                    let intensity = position_data.get(charge_idx).copied().unwrap_or(0.0);
                    intensities.push(intensity);
                }
            }
        }

        Self {
            peptide_len,
            max_charge,
            ion_kinds,
            intensities,
        }
    }

    /// Get predicted intensity for a specific fragment.
    ///
    /// # Arguments
    /// * `kind` - Ion type (B, Y, etc.)
    /// * `position` - Fragment position index (0-indexed, from IonSeries enumeration)
    /// * `charge` - Charge state (1-indexed)
    ///
    /// # Returns
    /// The predicted intensity, or `None` if the indices are out of bounds
    /// or the ion kind is not supported.
    pub fn get(&self, kind: Kind, position: usize, charge: u8) -> Option<f32> {
        let kind_idx = self.ion_kinds.iter().position(|k| *k == kind)?;
        let num_positions = self.peptide_len.saturating_sub(1);

        if position >= num_positions || charge == 0 || charge > self.max_charge {
            return None;
        }

        let idx = kind_idx * num_positions * self.max_charge as usize
            + position * self.max_charge as usize
            + (charge - 1) as usize;

        self.intensities.get(idx).copied()
    }

    /// Get predicted intensity, returning 1.0 if not found (neutral weight)
    pub fn get_or_default(&self, kind: Kind, position: usize, charge: u8) -> f32 {
        self.get(kind, position, charge).unwrap_or(1.0)
    }
}

/// Trait for fragment intensity prediction.
///
/// Allows different prediction backends (neural network, lookup table, etc.)
/// to be used interchangeably.
pub trait FragmentIntensityPredictor: Send + Sync {
    /// Predict intensities for a single peptide.
    ///
    /// Returns a tensor of predicted intensities for all fragment ions
    /// across all supported ion types and charge states.
    fn predict(&self, peptide: &Peptide) -> PeptideFragmentIntensities;

    /// Batch prediction for multiple peptides (optional optimization).
    ///
    /// Default implementation calls `predict` for each peptide.
    fn predict_batch(&self, peptides: &[&Peptide]) -> Vec<PeptideFragmentIntensities> {
        peptides.iter().map(|p| self.predict(p)).collect()
    }

    /// Get the maximum charge state this predictor supports.
    fn max_charge(&self) -> u8;

    /// Get the ion kinds this predictor supports.
    fn supported_ion_kinds(&self) -> &[Kind];
}

/// Uniform predictor that returns 1.0 for all fragments.
///
/// This is the default predictor that maintains backwards compatibility
/// with the original scoring behavior (no intensity weighting).
pub struct UniformPredictor {
    max_charge: u8,
    ion_kinds: Vec<Kind>,
}

impl UniformPredictor {
    /// Create a new uniform predictor with specified charge and ion kinds.
    pub fn new(max_charge: u8, ion_kinds: Vec<Kind>) -> Self {
        Self { max_charge, ion_kinds }
    }

    /// Create with default settings (B and Y ions, charge 1-3).
    pub fn default_settings() -> Self {
        Self {
            max_charge: 3,
            ion_kinds: vec![Kind::B, Kind::Y],
        }
    }
}

impl FragmentIntensityPredictor for UniformPredictor {
    fn predict(&self, peptide: &Peptide) -> PeptideFragmentIntensities {
        PeptideFragmentIntensities::uniform(
            peptide.sequence.len(),
            self.max_charge,
            self.ion_kinds.clone(),
        )
    }

    fn max_charge(&self) -> u8 {
        self.max_charge
    }

    fn supported_ion_kinds(&self) -> &[Kind] {
        &self.ion_kinds
    }
}

// ============================================================================
// Binary File Format for Pre-computed Predictions
// ============================================================================

/// Magic number for the binary file format: "SAGI" (Sage Intensities)
pub const INTENSITY_FILE_MAGIC: u32 = 0x49474153; // "SAGI" in little-endian
/// Current file format version
pub const INTENSITY_FILE_VERSION: u32 = 1;

/// Header for the predicted intensities binary file.
///
/// ## Binary Layout (little-endian)
/// ```text
/// Offset  Size  Field
/// 0       4     magic (0x49474153 = "SAGI")
/// 4       4     version (currently 1)
/// 8       8     peptide_count (u64)
/// 16      1     max_charge (u8)
/// 17      1     ion_kind_count (u8)
/// 18      N     ion_kinds (N bytes, where N = ion_kind_count)
/// ```
#[derive(Clone, Debug)]
pub struct IntensityFileHeader {
    pub peptide_count: u64,
    pub max_charge: u8,
    pub ion_kinds: Vec<Kind>,
}

impl IntensityFileHeader {
    /// Size of the fixed part of the header (before ion_kinds array)
    pub const FIXED_SIZE: usize = 4 + 4 + 8 + 1 + 1; // 18 bytes

    /// Write header to a writer
    pub fn write<W: Write>(&self, writer: &mut W) -> io::Result<()> {
        writer.write_all(&INTENSITY_FILE_MAGIC.to_le_bytes())?;
        writer.write_all(&INTENSITY_FILE_VERSION.to_le_bytes())?;
        writer.write_all(&self.peptide_count.to_le_bytes())?;
        writer.write_all(&[self.max_charge])?;
        writer.write_all(&[self.ion_kinds.len() as u8])?;
        for kind in &self.ion_kinds {
            writer.write_all(&[*kind as u8])?;
        }
        Ok(())
    }

    /// Read header from a reader
    pub fn read<R: Read>(reader: &mut R) -> io::Result<Self> {
        let mut buf4 = [0u8; 4];
        let mut buf8 = [0u8; 8];
        let mut buf1 = [0u8; 1];

        reader.read_exact(&mut buf4)?;
        let magic = u32::from_le_bytes(buf4);
        if magic != INTENSITY_FILE_MAGIC {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("Invalid magic number: expected 0x{:08X}, got 0x{:08X}", INTENSITY_FILE_MAGIC, magic),
            ));
        }

        reader.read_exact(&mut buf4)?;
        let version = u32::from_le_bytes(buf4);
        if version != INTENSITY_FILE_VERSION {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("Unsupported version: expected {}, got {}", INTENSITY_FILE_VERSION, version),
            ));
        }

        reader.read_exact(&mut buf8)?;
        let peptide_count = u64::from_le_bytes(buf8);

        reader.read_exact(&mut buf1)?;
        let max_charge = buf1[0];

        reader.read_exact(&mut buf1)?;
        let ion_kind_count = buf1[0] as usize;

        let mut ion_kinds = Vec::with_capacity(ion_kind_count);
        for _ in 0..ion_kind_count {
            reader.read_exact(&mut buf1)?;
            let kind = match buf1[0] {
                0 => Kind::A,
                1 => Kind::B,
                2 => Kind::C,
                3 => Kind::X,
                4 => Kind::Y,
                5 => Kind::Z,
                n => return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("Invalid ion kind: {}", n),
                )),
            };
            ion_kinds.push(kind);
        }

        Ok(Self {
            peptide_count,
            max_charge,
            ion_kinds,
        })
    }

    /// Total header size including ion_kinds
    pub fn total_size(&self) -> usize {
        Self::FIXED_SIZE + self.ion_kinds.len()
    }
}

/// Store for pre-computed predicted intensities loaded from a binary file.
///
/// ## File Format
///
/// The binary file has three sections:
///
/// 1. **Header** (variable size, see `IntensityFileHeader`)
/// 2. **Offsets** (peptide_count * 8 bytes): byte offset into data section for each peptide
/// 3. **Data**: concatenated f32 arrays for each peptide's intensities
///
/// ### Data Layout per Peptide
///
/// For a peptide of length L with K ion kinds and max charge C:
/// - Number of positions: L - 1
/// - Number of f32 values: K * (L - 1) * C
/// - Layout: `[kind_0_pos_0_charge_1, kind_0_pos_0_charge_2, ..., kind_0_pos_0_charge_C,
///            kind_0_pos_1_charge_1, ..., kind_K_pos_(L-2)_charge_C]`
///
/// ## Python Writer Example
///
/// ```python
/// import struct
/// import numpy as np
///
/// def write_intensity_file(path, peptide_lengths, predictions, max_charge, ion_kinds):
///     """
///     predictions: list of numpy arrays, one per peptide
///                  each array has shape [ion_kinds, peptide_len-1, max_charge]
///     ion_kinds: list of int (0=A, 1=B, 2=C, 3=X, 4=Y, 5=Z)
///     """
///     with open(path, 'wb') as f:
///         # Header
///         f.write(struct.pack('<I', 0x49474153))  # magic "SAGI"
///         f.write(struct.pack('<I', 1))           # version
///         f.write(struct.pack('<Q', len(predictions)))  # peptide_count
///         f.write(struct.pack('<B', max_charge))
///         f.write(struct.pack('<B', len(ion_kinds)))
///         for k in ion_kinds:
///             f.write(struct.pack('<B', k))
///
///         # Calculate offsets
///         offsets = []
///         current_offset = 0
///         for pred in predictions:
///             offsets.append(current_offset)
///             current_offset += pred.size * 4  # f32 = 4 bytes
///
///         # Write offsets
///         for off in offsets:
///             f.write(struct.pack('<Q', off))
///
///         # Write data
///         for pred in predictions:
///             f.write(pred.astype('<f4').tobytes())
/// ```
pub struct PredictedIntensityStore {
    header: IntensityFileHeader,
    /// Byte offsets into the data section for each peptide
    offsets: Vec<u64>,
    /// All intensity data concatenated
    data: Vec<f32>,
}

impl PredictedIntensityStore {
    /// Load predicted intensities from a binary file.
    pub fn load<P: AsRef<Path>>(path: P) -> io::Result<Self> {
        let file = File::open(path)?;
        let mut reader = BufReader::new(file);

        let header = IntensityFileHeader::read(&mut reader)?;

        // Read offsets
        let mut offsets = Vec::with_capacity(header.peptide_count as usize);
        let mut buf8 = [0u8; 8];
        for _ in 0..header.peptide_count {
            reader.read_exact(&mut buf8)?;
            offsets.push(u64::from_le_bytes(buf8));
        }

        // Read all remaining data as f32
        let mut data_bytes = Vec::new();
        reader.read_to_end(&mut data_bytes)?;

        if data_bytes.len() % 4 != 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Data section size is not a multiple of 4 bytes",
            ));
        }

        let data: Vec<f32> = data_bytes
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();

        Ok(Self { header, offsets, data })
    }

    /// Get predicted intensities for a peptide by index.
    ///
    /// Returns `None` if the peptide index is out of bounds.
    pub fn get(&self, peptide_idx: usize, peptide_len: usize) -> Option<PeptideFragmentIntensities> {
        if peptide_idx >= self.offsets.len() {
            return None;
        }

        let num_positions = peptide_len.saturating_sub(1);
        let elements_per_peptide = self.header.ion_kinds.len() * num_positions * self.header.max_charge as usize;

        let start = (self.offsets[peptide_idx] / 4) as usize; // Convert byte offset to f32 index
        let end = start + elements_per_peptide;

        if end > self.data.len() {
            return None;
        }

        Some(PeptideFragmentIntensities {
            peptide_len,
            max_charge: self.header.max_charge,
            ion_kinds: self.header.ion_kinds.clone(),
            intensities: self.data[start..end].to_vec(),
        })
    }

    /// Get a single intensity value directly without constructing PeptideFragmentIntensities.
    ///
    /// This is more efficient for scoring when you only need individual lookups.
    pub fn get_intensity(
        &self,
        peptide_idx: usize,
        peptide_len: usize,
        kind: Kind,
        position: usize,
        charge: u8,
    ) -> Option<f32> {
        if peptide_idx >= self.offsets.len() {
            return None;
        }

        let kind_idx = self.header.ion_kinds.iter().position(|k| *k == kind)?;
        let num_positions = peptide_len.saturating_sub(1);

        if position >= num_positions || charge == 0 || charge > self.header.max_charge {
            return None;
        }

        let start = (self.offsets[peptide_idx] / 4) as usize;
        let local_idx = kind_idx * num_positions * self.header.max_charge as usize
            + position * self.header.max_charge as usize
            + (charge - 1) as usize;

        self.data.get(start + local_idx).copied()
    }

    /// Get intensity with fallback to 1.0 if not found.
    pub fn get_intensity_or_default(
        &self,
        peptide_idx: usize,
        peptide_len: usize,
        kind: Kind,
        position: usize,
        charge: u8,
    ) -> f32 {
        self.get_intensity(peptide_idx, peptide_len, kind, position, charge)
            .unwrap_or(1.0)
    }

    /// Number of peptides in the store.
    pub fn peptide_count(&self) -> usize {
        self.offsets.len()
    }

    /// Maximum charge state stored.
    pub fn max_charge(&self) -> u8 {
        self.header.max_charge
    }

    /// Ion kinds stored.
    pub fn ion_kinds(&self) -> &[Kind] {
        &self.header.ion_kinds
    }

    /// Create a new store from a list of PeptideFragmentIntensities.
    ///
    /// This is useful for testing or for creating a store programmatically.
    pub fn from_predictions(
        predictions: Vec<PeptideFragmentIntensities>,
        max_charge: u8,
        ion_kinds: Vec<Kind>,
    ) -> Self {
        let mut offsets = Vec::with_capacity(predictions.len());
        let mut data = Vec::new();

        for pred in &predictions {
            offsets.push((data.len() * 4) as u64); // byte offset
            data.extend_from_slice(&pred.intensities);
        }

        Self {
            header: IntensityFileHeader {
                peptide_count: predictions.len() as u64,
                max_charge,
                ion_kinds,
            },
            offsets,
            data,
        }
    }

    /// Create a uniform store where all intensities are 1.0.
    ///
    /// This is useful for testing the weighted scoring code path without
    /// actual predictions. With uniform intensities, weighted scores should
    /// be identical to unweighted scores.
    ///
    /// # Arguments
    /// * `peptide_lengths` - Length of each peptide in the database
    /// * `max_charge` - Maximum fragment charge state
    /// * `ion_kinds` - Ion types to include (typically B and Y)
    pub fn uniform(peptide_lengths: &[usize], max_charge: u8, ion_kinds: Vec<Kind>) -> Self {
        let predictions: Vec<PeptideFragmentIntensities> = peptide_lengths
            .iter()
            .map(|&len| PeptideFragmentIntensities::uniform(len, max_charge, ion_kinds.clone()))
            .collect();

        Self::from_predictions(predictions, max_charge, ion_kinds)
    }

    /// Write the store to a binary file.
    pub fn write<P: AsRef<Path>>(&self, path: P) -> io::Result<()> {
        use std::io::BufWriter;
        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);

        self.header.write(&mut writer)?;

        // Write offsets
        for offset in &self.offsets {
            writer.write_all(&offset.to_le_bytes())?;
        }

        // Write data
        for &value in &self.data {
            writer.write_all(&value.to_le_bytes())?;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_uniform_intensities() {
        let intensities = PeptideFragmentIntensities::uniform(10, 3, vec![Kind::B, Kind::Y]);

        // peptide_len = 10, so 9 fragment positions
        // 2 ion types * 9 positions * 3 charges = 54 elements
        assert_eq!(intensities.intensities.len(), 54);

        // All should be 1.0
        for &i in &intensities.intensities {
            assert_eq!(i, 1.0);
        }
    }

    #[test]
    fn test_get_intensity() {
        let intensities = PeptideFragmentIntensities::uniform(5, 2, vec![Kind::B, Kind::Y]);

        // Valid lookups should return 1.0
        assert_eq!(intensities.get(Kind::B, 0, 1), Some(1.0));
        assert_eq!(intensities.get(Kind::B, 3, 2), Some(1.0)); // position 3 is valid (0-3 for len 5)
        assert_eq!(intensities.get(Kind::Y, 2, 1), Some(1.0));

        // Out of bounds
        assert_eq!(intensities.get(Kind::B, 4, 1), None); // position 4 invalid for len 5
        assert_eq!(intensities.get(Kind::B, 0, 3), None); // charge 3 > max_charge 2
        assert_eq!(intensities.get(Kind::B, 0, 0), None); // charge 0 invalid
        assert_eq!(intensities.get(Kind::A, 0, 1), None); // Kind::A not in ion_kinds
    }

    #[test]
    fn test_from_tensor() {
        // Create a simple tensor: 2 ion types, 3 positions, 2 charges
        let tensor = vec![
            vec![
                vec![0.1, 0.2], // B ion, position 0, charges 1,2
                vec![0.3, 0.4], // B ion, position 1
                vec![0.5, 0.6], // B ion, position 2
            ],
            vec![
                vec![0.7, 0.8], // Y ion, position 0
                vec![0.9, 1.0], // Y ion, position 1
                vec![1.1, 1.2], // Y ion, position 2
            ],
        ];

        let intensities = PeptideFragmentIntensities::from_tensor(
            &tensor,
            4, // peptide_len = 4 means 3 positions
            2,
            vec![Kind::B, Kind::Y],
        );

        assert_eq!(intensities.get(Kind::B, 0, 1), Some(0.1));
        assert_eq!(intensities.get(Kind::B, 0, 2), Some(0.2));
        assert_eq!(intensities.get(Kind::B, 2, 2), Some(0.6));
        assert_eq!(intensities.get(Kind::Y, 0, 1), Some(0.7));
        assert_eq!(intensities.get(Kind::Y, 2, 2), Some(1.2));
    }

    #[test]
    fn test_get_or_default() {
        let intensities = PeptideFragmentIntensities::uniform(5, 2, vec![Kind::B, Kind::Y]);

        // Valid lookup
        assert_eq!(intensities.get_or_default(Kind::B, 0, 1), 1.0);

        // Invalid lookup returns default 1.0
        assert_eq!(intensities.get_or_default(Kind::A, 0, 1), 1.0);
        assert_eq!(intensities.get_or_default(Kind::B, 100, 1), 1.0);
    }

    #[test]
    fn test_intensity_store_roundtrip() {
        // Create predictions for peptides of different lengths
        let pred1 = PeptideFragmentIntensities::from_tensor(
            &vec![
                vec![vec![0.1, 0.2], vec![0.3, 0.4], vec![0.5, 0.6]], // B: 3 positions, 2 charges
                vec![vec![0.7, 0.8], vec![0.9, 1.0], vec![1.1, 1.2]], // Y: 3 positions, 2 charges
            ],
            4, // peptide_len = 4
            2,
            vec![Kind::B, Kind::Y],
        );

        let pred2 = PeptideFragmentIntensities::from_tensor(
            &vec![
                vec![vec![2.1, 2.2], vec![2.3, 2.4]], // B: 2 positions
                vec![vec![2.5, 2.6], vec![2.7, 2.8]], // Y: 2 positions
            ],
            3, // peptide_len = 3
            2,
            vec![Kind::B, Kind::Y],
        );

        let store = PredictedIntensityStore::from_predictions(
            vec![pred1.clone(), pred2.clone()],
            2,
            vec![Kind::B, Kind::Y],
        );

        // Write to temp file and read back
        let temp_path = std::env::temp_dir().join("test_intensities.bin");
        store.write(&temp_path).unwrap();
        let loaded = PredictedIntensityStore::load(&temp_path).unwrap();

        // Verify header
        assert_eq!(loaded.peptide_count(), 2);
        assert_eq!(loaded.max_charge(), 2);
        assert_eq!(loaded.ion_kinds(), &[Kind::B, Kind::Y]);

        // Verify peptide 0 (length 4)
        assert_eq!(loaded.get_intensity(0, 4, Kind::B, 0, 1), Some(0.1));
        assert_eq!(loaded.get_intensity(0, 4, Kind::B, 0, 2), Some(0.2));
        assert_eq!(loaded.get_intensity(0, 4, Kind::Y, 2, 2), Some(1.2));

        // Verify peptide 1 (length 3)
        assert_eq!(loaded.get_intensity(1, 3, Kind::B, 0, 1), Some(2.1));
        assert_eq!(loaded.get_intensity(1, 3, Kind::Y, 1, 2), Some(2.8));

        // Verify full PeptideFragmentIntensities retrieval
        let retrieved = loaded.get(0, 4).unwrap();
        assert_eq!(retrieved.get(Kind::B, 0, 1), Some(0.1));
        assert_eq!(retrieved.get(Kind::Y, 2, 2), Some(1.2));

        // Clean up
        std::fs::remove_file(&temp_path).ok();
    }
}
