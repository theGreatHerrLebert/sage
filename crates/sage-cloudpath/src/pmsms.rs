//! Sage reader for IonMaiden pseudo-MS/MS (`.pmsms` bundles).
//!
//! Drop-in sibling of `tdf.rs`: turns a *deconvolved* diaPASEF run into a
//! `Vec<RawSpectrum>` the search core consumes unchanged — one `RawSpectrum`
//! per precursor, each with a single real precursor m/z. Run Sage NARROW (not
//! `wide_window`): the upstream deconvolution already demultiplexed the windows,
//! so this reader only *parses* a bundle; it does no deconvolution itself.
//!
//! This module is self-contained — it vendors a tiny read-only reader for the
//! `mmappet` struct-of-arrays format (a directory of `schema.txt` + flat
//! little-endian `<idx>.bin` columns) so it carries no dependency on the
//! (private) producer pipeline.
//!
//! Two on-disk layouts are auto-detected:
//!
//! 1. **C++ `mkpmsms` bundle** (`precursors.parquet` present) — the format the
//!    production pipeline emits:
//!    ```text
//!    run.pmsms/
//!      pmsms.mmappet/         # FLAT peaks: tof,intensity (u32)
//!      precursors.parquet     # precursor_idx,mz,rt,inv_ion_mobility,charges,
//!                             #   fragment_spectrum_start,fragment_event_cnt  (CSR ranges)
//!      tof2mz.mmappet/        # f32[tof] calibration (column "x")
//!    ```
//!    Needs the `parquet` feature (sage-cli enables it).
//!
//! 2. **Ragged mmappet bundle** (`precursors.mmappet` present) — peaks
//!    `tof,intensity,score` + a nested `dataindex.mmappet` (`precursor_idx,size,
//!    idx`) + a typed `precursors.mmappet` + `tof2mz.bin` (f64).
//!
//! `SAGE_PMSMS_LIMIT=N` searches only the first N precursors. MS2 only.

use std::path::Path;

use sage_core::spectrum::{Precursor, RawSpectrum, Representation};

mod mmappet {
    //! Minimal read-only reader for the mmappet column format.
    use std::path::Path;

    use memmap2::Mmap;

    /// One memory-mapped flat column, reinterpretable as a typed LE slice.
    pub struct Column {
        map: Mmap,
        dtype: String,
    }

    impl Column {
        fn cast<T: Copy>(&self, np: &str) -> Result<&[T], String> {
            if self.dtype != np {
                return Err(format!("column dtype is {}, expected {np}", self.dtype));
            }
            let bytes = &self.map[..];
            let sz = std::mem::size_of::<T>();
            if bytes.len() % sz != 0 {
                return Err(format!("column length {} not a multiple of {sz}", bytes.len()));
            }
            if bytes.is_empty() {
                return Ok(&[]);
            }
            // mmap base pointers are page-aligned, hence aligned for any scalar.
            debug_assert_eq!(bytes.as_ptr() as usize % std::mem::align_of::<T>(), 0);
            // SAFETY: length is an exact multiple of size_of::<T>, the pointer is
            // aligned, and the requested scalar types have no invalid bit patterns.
            Ok(unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const T, bytes.len() / sz) })
        }
        pub fn u8(&self) -> Result<&[u8], String> { self.cast("uint8") }
        pub fn u32(&self) -> Result<&[u32], String> { self.cast("uint32") }
        pub fn u64(&self) -> Result<&[u64], String> { self.cast("uint64") }
        pub fn f32(&self) -> Result<&[f32], String> { self.cast("float32") }
        pub fn f64(&self) -> Result<&[f64], String> { self.cast("float64") }
    }

    /// Open column `name` from a mmappet dataset directory (mmaps `<idx>.bin`).
    pub fn column(dir: &Path, name: &str) -> Result<Column, String> {
        let schema_path = dir.join("schema.txt");
        let schema = std::fs::read_to_string(&schema_path)
            .map_err(|e| format!("{}: {e}", schema_path.display()))?;
        let mut found = None;
        for (i, line) in schema.lines().filter(|l| !l.trim().is_empty()).enumerate() {
            // each line is "<numpy-dtype> <name>"
            let trimmed = line.trim_start();
            let split = trimmed.find(char::is_whitespace).unwrap_or(trimmed.len());
            let dtype = &trimmed[..split];
            let col = trimmed[split..].trim_start();
            if col == name {
                found = Some((i, dtype.to_string()));
                break;
            }
        }
        let (idx, dtype) = found.ok_or_else(|| format!("column {name:?} not in {}", schema_path.display()))?;
        let bin = dir.join(format!("{idx}.bin"));
        let file = std::fs::File::open(&bin).map_err(|e| format!("{}: {e}", bin.display()))?;
        // SAFETY: read-only map of a file we just opened.
        let map = unsafe { Mmap::map(&file) }.map_err(|e| format!("mmap {}: {e}", bin.display()))?;
        Ok(Column { map, dtype })
    }
}

#[derive(Debug, thiserror::Error)]
pub enum PmsmsError {
    #[error("pmsms reader: {0}")]
    Read(String),
    #[error(transparent)]
    Io(#[from] std::io::Error),
}

fn err(ctx: impl std::fmt::Display) -> PmsmsError {
    PmsmsError::Read(ctx.to_string())
}

// Read an integer parquet cell as u64/i64 regardless of signed/unsigned logical
// type (the two bundle variants differ: unsigned int64 vs signed int64).
#[cfg(feature = "parquet")]
fn row_u64(row: &parquet::record::Row, i: usize) -> Result<u64, PmsmsError> {
    use parquet::record::RowAccessor;
    row.get_ulong(i).or_else(|_| row.get_long(i).map(|v| v as u64)).map_err(err)
}
#[cfg(feature = "parquet")]
fn row_i64(row: &parquet::record::Row, i: usize) -> Result<i64, PmsmsError> {
    use parquet::record::RowAccessor;
    row.get_long(i).or_else(|_| row.get_ulong(i).map(|v| v as i64)).map_err(err)
}

fn limit() -> usize {
    std::env::var("SAGE_PMSMS_LIMIT").ok().and_then(|v| v.parse().ok()).unwrap_or(0)
}

pub struct PseudoMsMsReader;

impl PseudoMsMsReader {
    pub fn parse(&self, dir: impl AsRef<Path>, file_id: usize) -> Result<Vec<RawSpectrum>, PmsmsError> {
        let dir = dir.as_ref();
        if dir.join("precursors.parquet").exists() {
            self.parse_parquet_bundle(dir, file_id)
        } else {
            self.parse_mmappet_bundle(dir, file_id)
        }
    }

    // ---- C++ mkpmsms bundle: flat peaks + parquet CSR ranges ------------------
    #[cfg(feature = "parquet")]
    fn parse_parquet_bundle(&self, dir: &Path, file_id: usize) -> Result<Vec<RawSpectrum>, PmsmsError> {
        use parquet::file::reader::{FileReader, SerializedFileReader};
        use parquet::record::RowAccessor;

        let tof2mz = load_tof2mz(dir)?;
        let n_tof = tof2mz.len();

        let tof_col = mmappet::column(&dir.join("pmsms.mmappet"), "tof").map_err(err)?;
        let int_col = mmappet::column(&dir.join("pmsms.mmappet"), "intensity").map_err(err)?;
        let tof = tof_col.u32().map_err(err)?;
        let inten = int_col.u32().map_err(err)?;
        let n_peaks = tof.len();

        let file = std::fs::File::open(dir.join("precursors.parquet"))?;
        let reader = SerializedFileReader::new(file).map_err(err)?;

        // Map column NAME -> field index. The two bundle variants order columns
        // differently (and the reference carries ~40 columns), so never read by position.
        let schema = reader.metadata().file_metadata().schema_descr();
        let mut col = std::collections::HashMap::new();
        for i in 0..schema.num_columns() {
            col.insert(schema.column(i).name().to_string(), i);
        }
        let idx = |n: &str| col.get(n).copied().ok_or_else(|| err(format!("precursors.parquet missing column '{n}'")));
        let (i_pidx, i_mz, i_rt, i_iim, i_ch, i_start, i_cnt) = (
            idx("precursor_idx")?, idx("mz")?, idx("rt")?, idx("inv_ion_mobility")?,
            idx("charges")?, idx("fragment_spectrum_start")?, idx("fragment_event_cnt")?,
        );
        let lim = limit();

        let mut out = Vec::new();
        for row in reader.get_row_iter(None).map_err(err)? {
            let row = row.map_err(err)?;
            let pidx = row_u64(&row, i_pidx)?;
            let mz = row.get_double(i_mz).map_err(err)?;
            let rt = row.get_double(i_rt).map_err(err)?;
            let iim = row.get_double(i_iim).map_err(err)?;
            let charge = row_i64(&row, i_ch)?;
            let start = row_u64(&row, i_start)? as usize;
            let cnt = row_u64(&row, i_cnt)? as usize;

            let end = start.checked_add(cnt).filter(|&e| e <= n_peaks).ok_or_else(|| {
                err(format!("precursor {pidx}: peak span {start}..+{cnt} exceeds {n_peaks}"))
            })?;
            out.push(build_spectrum(
                file_id, pidx, mz, rt, iim, charge, &tof[start..end], &inten[start..end], &tof2mz, n_tof,
            )?);
            if lim != 0 && out.len() >= lim {
                break;
            }
        }
        log::info!("pmsms(parquet): {} spectra from {}", out.len(), dir.display());
        Ok(out)
    }

    #[cfg(not(feature = "parquet"))]
    fn parse_parquet_bundle(&self, _dir: &Path, _file_id: usize) -> Result<Vec<RawSpectrum>, PmsmsError> {
        Err(err("precursors.parquet bundle needs the `parquet` feature"))
    }

    // ---- Ragged mmappet bundle: pmsms+dataindex + precursors.mmappet ----------
    fn parse_mmappet_bundle(&self, dir: &Path, file_id: usize) -> Result<Vec<RawSpectrum>, PmsmsError> {
        let tof2mz = load_tof2mz(dir)?;
        let n_tof = tof2mz.len();

        let pm = dir.join("pmsms.mmappet");
        let tof_col = mmappet::column(&pm, "tof").map_err(err)?;
        let int_col = mmappet::column(&pm, "intensity").map_err(err)?;
        let tof = tof_col.u32().map_err(err)?;
        let inten = int_col.u32().map_err(err)?;

        // ragged index (CSR): precursor_idx, size, idx (start offset)
        let di = pm.join("dataindex.mmappet");
        let di_pidx = mmappet::column(&di, "precursor_idx").map_err(err)?;
        let di_size = mmappet::column(&di, "size").map_err(err)?;
        let di_idx = mmappet::column(&di, "idx").map_err(err)?;
        let (di_pidx, di_size, di_idx) = (di_pidx.u64().map_err(err)?, di_size.u64().map_err(err)?, di_idx.u64().map_err(err)?);

        // precursor metadata, joined by precursor_idx
        let pd = dir.join("precursors.mmappet");
        let (c_idx, c_mz, c_rt, c_iim, c_ch) = (
            mmappet::column(&pd, "precursor_idx").map_err(err)?,
            mmappet::column(&pd, "mz").map_err(err)?,
            mmappet::column(&pd, "rt").map_err(err)?,
            mmappet::column(&pd, "inv_ion_mobility").map_err(err)?,
            mmappet::column(&pd, "charge").map_err(err)?,
        );
        let (p_idx, p_mz, p_rt, p_iim, p_ch) =
            (c_idx.u64().map_err(err)?, c_mz.f64().map_err(err)?, c_rt.f64().map_err(err)?, c_iim.f64().map_err(err)?, c_ch.u8().map_err(err)?);
        let mut row_of = std::collections::HashMap::with_capacity(p_idx.len());
        for (r, &pid) in p_idx.iter().enumerate() {
            row_of.insert(pid, r);
        }

        let n_peaks = tof.len();
        let lim = limit();
        let mut out = Vec::with_capacity(di_pidx.len());
        for i in 0..di_pidx.len() {
            let pidx = di_pidx[i];
            let start = di_idx[i] as usize;
            let end = start.checked_add(di_size[i] as usize).filter(|&e| e <= n_peaks).ok_or_else(|| {
                err(format!("spectrum {i}: peak span exceeds {n_peaks}"))
            })?;
            let &r = row_of.get(&pidx).ok_or_else(|| err(format!("precursor_idx {pidx} not in precursors.mmappet")))?;
            out.push(build_spectrum(
                file_id, pidx, p_mz[r], p_rt[r], p_iim[r], p_ch[r] as i64, &tof[start..end], &inten[start..end], &tof2mz, n_tof,
            )?);
            if lim != 0 && out.len() >= lim {
                break;
            }
        }
        Ok(out)
    }
}

#[allow(clippy::too_many_arguments)]
fn build_spectrum(
    file_id: usize,
    pidx: u64,
    prec_mz: f64,
    rt: f64,
    iim: f64,
    charge: i64,
    tof: &[u32],
    inten: &[u32],
    tof2mz: &[f64],
    n_tof: usize,
) -> Result<RawSpectrum, PmsmsError> {
    // Drop zero-intensity peaks: they carry no signal, and a peptide matching
    // only such peaks gives summed_intensity 0 -> average_ppm = 0/0 = NaN, which
    // poisons Sage's LDA scatter matrix (-> heuristic fallback).
    let mut mz = Vec::with_capacity(tof.len());
    let mut intensity: Vec<f32> = Vec::with_capacity(tof.len());
    for (&t, &iv) in tof.iter().zip(inten) {
        if iv == 0 {
            continue;
        }
        let ti = t as usize;
        if ti >= n_tof {
            return Err(err(format!("fragment tof {t} out of range for tof2mz (len {n_tof})")));
        }
        mz.push(tof2mz[ti] as f32);
        intensity.push(iv as f32);
    }
    let total_ion_current = intensity.iter().sum();

    let mut precursor = Precursor::default();
    precursor.mz = prec_mz as f32;
    // sagepy-parity: leave charge unset so Sage searches the configured
    // precursor_charge range, rather than locking to the deconvolved charge.
    let _ = charge;
    precursor.charge = None;
    precursor.inverse_ion_mobility = Some(iim as f32);
    // isolation_window left None: deconvolution assigned a single precursor m/z (narrow mode).

    Ok(RawSpectrum {
        file_id,
        ms_level: 2,
        id: pidx.to_string(),
        representation: Representation::Centroid, // REQUIRED: process_ms2 panics on Profile
        scan_start_time: (rt / 60.0) as f32,      // rt seconds -> minutes
        ion_injection_time: rt as f32,
        total_ion_current,
        mz,
        intensity,
        precursors: vec![precursor],
        mobility: None,
        ..Default::default()
    })
}

/// Load tof2mz as f64 from `tof2mz.mmappet` (f32/f64 column "x") or `tof2mz.bin` (f64).
fn load_tof2mz(dir: &Path) -> Result<Vec<f64>, PmsmsError> {
    let mm = dir.join("tof2mz.mmappet");
    if mm.exists() {
        let col = mmappet::column(&mm, "x").map_err(err)?;
        return col
            .f32()
            .map(|s| s.iter().map(|&v| v as f64).collect())
            .or_else(|_| col.f64().map(|s| s.to_vec()))
            .map_err(err);
    }
    let bin = dir.join("tof2mz.bin");
    let bytes = std::fs::read(&bin)?;
    if bytes.len() % 8 != 0 {
        return Err(err(format!("{}: not a whole number of f64", bin.display())));
    }
    Ok(bytes.chunks_exact(8).map(|c| f64::from_le_bytes(c.try_into().unwrap())).collect())
}
