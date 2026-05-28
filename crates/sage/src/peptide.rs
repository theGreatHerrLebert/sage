use std::cmp::Ordering;
use std::{collections::HashMap, fmt::Debug, sync::Arc};

use crate::modification::ModificationSpecificity;
use crate::{
    enzyme::{Digest, DigestGroup, Position},
    mass::{monoisotopic, H2O},
};
use fnv::FnvHashSet;
use itertools::Itertools;
use rand::prelude::SliceRandom;
use rand::thread_rng;
use smallvec::SmallVec;

/// Per-residue modification masses, stored **sparsely**: only the residue
/// positions carrying a nonzero mass-delta, sorted ascending by index, with no
/// duplicate indices. An unmodified peptide stores an empty list.
///
/// This canonical form (sorted / unique / nonzero-only) makes the derived
/// `PartialEq` correct for de-duplication (two peptides have equal residue mods
/// iff their sparse lists are equal), and [`Mods::cmp_dense`] reproduces the
/// historical dense `Vec<f32>` ordering used by [`Peptide::initial_sort`].
///
/// Memory: replaces the eager `vec![0.0; sequence.len()]` (one heap allocation
/// per peptide, almost always all zeros for typical searches) with inline
/// storage for up to 2 modified sites — the common case allocates nothing.
///
/// All construction / mutation goes through [`Mods::set_if_unmodified`] (or
/// [`Mods::from_dense`] for the reverse/shuffle remap), which preserve the
/// canonical invariant. Mod masses are assumed finite (NaN rejected at config).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Mods(SmallVec<[(u16, f32); 2]>);

impl Mods {
    /// Mass-delta at residue index `i` (0.0 if that residue is unmodified).
    #[inline]
    pub fn mass_at(&self, i: usize) -> f32 {
        let key = i as u16;
        for &(idx, mass) in &self.0 {
            if idx == key {
                return mass;
            }
            if idx > key {
                break; // sorted — no entry for `i`
            }
        }
        0.0
    }

    /// Set residue `i` to `mass` **iff** it is currently unmodified. Mirrors the
    /// historical `if modifications[i] == 0.0 { modifications[i] = mass }` guard.
    ///
    /// Self-enforces the canonical invariant: a zero `mass` is a no-op (never
    /// stored), so `Mods` can never hold an explicit-zero entry that would make
    /// derived `PartialEq` diverge from dense equality during dedup. (Config
    /// validation also rejects zero-mass mods; this guard is belt-and-suspenders.)
    pub fn set_if_unmodified(&mut self, i: usize, mass: f32) {
        if mass == 0.0 {
            return;
        }
        debug_assert!(
            i <= u16::MAX as usize,
            "residue index {i} exceeds u16; max peptide length must be <= 65535"
        );
        let key = i as u16;
        match self.0.binary_search_by(|(idx, _)| idx.cmp(&key)) {
            Ok(_) => {} // already modified — no-op
            Err(pos) => self.0.insert(pos, (key, mass)),
        }
    }

    /// Sum of all residue mod masses (N/C-term masses are tracked separately).
    #[inline]
    pub fn total(&self) -> f32 {
        self.0.iter().map(|&(_, m)| m).sum()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Iterate the modified `(residue_index, mass)` pairs, sorted by index.
    pub fn iter(&self) -> impl Iterator<Item = (usize, f32)> + '_ {
        self.0.iter().map(|&(i, m)| (i as usize, m))
    }

    /// Materialise the dense per-residue vector of length `len` (0.0 at
    /// unmodified positions). Used at the reverse/shuffle remap sites and for
    /// the FFI/Display dense view.
    pub fn to_dense(&self, len: usize) -> Vec<f32> {
        let mut dense = vec![0.0f32; len];
        for &(i, m) in &self.0 {
            if (i as usize) < len {
                dense[i as usize] = m;
            }
        }
        dense
    }

    /// Build the canonical sparse form from a dense vector (drops zeros, which
    /// also drops `-0.0` since `-0.0 != 0.0` is false).
    pub fn from_dense(dense: &[f32]) -> Self {
        debug_assert!(
            dense.len() <= u16::MAX as usize + 1,
            "dense mod length {} exceeds u16 index range",
            dense.len()
        );
        let mut v: SmallVec<[(u16, f32); 2]> = SmallVec::new();
        for (i, &m) in dense.iter().enumerate() {
            if m != 0.0 {
                v.push((i as u16, m));
            }
        }
        Mods(v)
    }

    /// Reproduce `Vec<f32>::partial_cmp(...).unwrap_or(Equal)` over the dense
    /// view. Only called when comparing peptides of identical sequence (hence
    /// identical length) — see [`Peptide::initial_sort`] — so the dense
    /// lengths match and this is a pure lexicographic comparison with 0.0 at
    /// gaps. Walks both sorted sparse lists by ascending index.
    pub fn cmp_dense(&self, other: &Self) -> Ordering {
        let (a, b) = (&self.0, &other.0);
        let (mut i, mut j) = (0usize, 0usize);
        while i < a.len() && j < b.len() {
            let (ai, am) = a[i];
            let (bj, bm) = b[j];
            let ord = if ai == bj {
                let o = am.partial_cmp(&bm).unwrap_or(Ordering::Equal);
                i += 1;
                j += 1;
                o
            } else if ai < bj {
                let o = am.partial_cmp(&0.0).unwrap_or(Ordering::Equal);
                i += 1;
                o
            } else {
                let o = 0.0f32.partial_cmp(&bm).unwrap_or(Ordering::Equal);
                j += 1;
                o
            };
            if ord != Ordering::Equal {
                return ord;
            }
        }
        while i < a.len() {
            let o = a[i].1.partial_cmp(&0.0).unwrap_or(Ordering::Equal);
            if o != Ordering::Equal {
                return o;
            }
            i += 1;
        }
        while j < b.len() {
            let o = 0.0f32.partial_cmp(&b[j].1).unwrap_or(Ordering::Equal);
            if o != Ordering::Equal {
                return o;
            }
            j += 1;
        }
        Ordering::Equal
    }
}

#[derive(Clone, PartialEq, Default)]
pub struct Peptide {
    pub decoy: bool,
    pub sequence: Arc<[u8]>,
    pub modifications: Mods,
    /// Modification on peptide C-terminus
    pub nterm: Option<f32>,
    /// Modification on peptide C-terminus
    pub cterm: Option<f32>,
    /// Monoisotopic mass, inclusive of N/C-terminal mods
    pub monoisotopic: f32,
    /// Number of missed cleavages for this sequence
    pub missed_cleavages: u8,
    /// Is this a semi-enzymatic peptide?
    pub semi_enzymatic: bool,
    /// Where is this peptide located in the protein?
    pub position: Position,

    pub proteins: Vec<Arc<str>>,
}

impl Peptide {
    pub fn initial_sort(&self, other: &Self) -> std::cmp::Ordering {
        self.sequence
            .cmp(&other.sequence)
            .then_with(|| self.modifications.cmp_dense(&other.modifications))
            .then_with(|| {
                self.nterm
                    .partial_cmp(&other.nterm)
                    .unwrap_or(Ordering::Equal)
            })
            .then_with(|| {
                self.cterm
                    .partial_cmp(&other.cterm)
                    .unwrap_or(Ordering::Equal)
            })
    }
}

impl Debug for Peptide {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Peptide")
            .field("proteins", &self.proteins)
            .field("decoy", &self.decoy)
            .field(
                "sequence",
                &std::str::from_utf8(&self.sequence).unwrap_or("error"),
            )
            .field("nterm", &self.nterm)
            .field("cterm", &self.cterm)
            .field("monoisotopic", &self.monoisotopic)
            .field("missed_cleavages", &self.missed_cleavages)
            .field("position", &self.position)
            .finish()
    }
}

impl Peptide {
    pub fn label(&self) -> i32 {
        match self.decoy {
            true => -1,
            false => 1,
        }
    }

    pub fn proteins(&self, decoy_tag: &str, generate_decoys: bool) -> String {
        if self.decoy {
            self.proteins
                .iter()
                .map(|s| {
                    if generate_decoys {
                        format!("{}{}", decoy_tag, s)
                    } else {
                        s.to_string()
                    }
                })
                .join(";")
        } else {
            self.proteins.iter().join(";")
        }
    }

    pub fn modification_count(&self, target: ModificationSpecificity, mass: f32) -> usize {
        match target {
            ModificationSpecificity::PeptideN(r) | ModificationSpecificity::ProteinN(r) => {
                if r.map(|resi| resi == *self.sequence.first().unwrap_or(&0))
                    .unwrap_or(true)
                    && self.nterm.unwrap_or_default() == mass
                {
                    1
                } else {
                    0
                }
            }
            ModificationSpecificity::PeptideC(r) | ModificationSpecificity::ProteinC(r) => {
                if r.map(|resi| resi == *self.sequence.last().unwrap_or(&0))
                    .unwrap_or(true)
                    && self.cterm.unwrap_or_default() == mass
                {
                    1
                } else {
                    0
                }
            }
            ModificationSpecificity::Residue(resi) => self
                .modifications
                .iter()
                .filter(|&(idx, m)| mass == m && self.sequence.get(idx) == Some(&resi))
                .count(),
        }
    }

    fn modification_mass(&self) -> f32 {
        self.modifications.total() + self.nterm.unwrap_or(0.0) + self.cterm.unwrap_or(0.0)
    }

    /// Apply all variable mods in `sites` to self
    fn apply_site(&mut self, site: Site, mass: f32) {
        match site {
            Site::Nterm => {
                if self.nterm.is_none() {
                    self.nterm = self.nterm.or(Some(0.0)).map(|x| x + mass);
                }
            }
            Site::Cterm => {
                if self.cterm.is_none() {
                    self.cterm = self.cterm.or(Some(0.0)).map(|x| x + mass);
                }
            }
            Site::Sequence(index) => {
                self.modifications.set_if_unmodified(index as usize, mass);
            }
        }
    }

    fn push_resi(&self, acc: &mut Vec<(Site, f32)>, target: ModificationSpecificity, mass: f32) {
        match (target, self.position) {
            (ModificationSpecificity::PeptideN(None), _) => acc.push((Site::Nterm, mass)),
            (ModificationSpecificity::PeptideN(Some(resi)), _)
            if resi == *self.sequence.first().unwrap_or(&0) =>
                {
                    acc.push((Site::Sequence(0), mass))
                }
            (ModificationSpecificity::PeptideC(None), _) => acc.push((Site::Cterm, mass)),
            (ModificationSpecificity::PeptideC(Some(resi)), _)
            if resi == *self.sequence.last().unwrap_or(&0) =>
                {
                    acc.push((
                        Site::Sequence(self.sequence.len().saturating_sub(1) as u32),
                        mass,
                    ))
                }
            (ModificationSpecificity::ProteinN(None), Position::Nterm | Position::Full) => {
                acc.push((Site::Nterm, mass))
            }
            (ModificationSpecificity::ProteinN(Some(resi)), Position::Nterm | Position::Full)
            if resi == *self.sequence.first().unwrap_or(&0) =>
                {
                    acc.push((Site::Sequence(0), mass))
                }
            (ModificationSpecificity::ProteinC(None), Position::Cterm | Position::Full) => {
                acc.push((Site::Cterm, mass))
            }
            (ModificationSpecificity::ProteinC(Some(resi)), Position::Cterm | Position::Full)
            if resi == *self.sequence.last().unwrap_or(&0) =>
                {
                    acc.push((
                        Site::Sequence(self.sequence.len().saturating_sub(1) as u32),
                        mass,
                    ))
                }
            (ModificationSpecificity::Residue(resi), _) => {
                acc.extend(
                    self.sequence
                        .iter()
                        .enumerate()
                        .filter_map(|(idx, residue)| {
                            if resi == *residue {
                                Some((Site::Sequence(idx as u32), mass))
                            } else {
                                None
                            }
                        }),
                );
            }
            _ => {}
        }
    }

    fn static_mods(&mut self, target: ModificationSpecificity, mass: f32) {
        match (target, self.position) {
            (ModificationSpecificity::PeptideN(None), _) => self.apply_site(Site::Nterm, mass),
            (ModificationSpecificity::PeptideN(Some(resi)), _)
            if resi == *self.sequence.first().unwrap_or(&0) =>
                {
                    self.apply_site(Site::Sequence(0), mass)
                }
            (ModificationSpecificity::PeptideC(None), _) => self.apply_site(Site::Cterm, mass),
            (ModificationSpecificity::PeptideC(Some(resi)), _)
            if resi == *self.sequence.last().unwrap_or(&0) =>
                {
                    self.apply_site(
                        Site::Sequence(self.sequence.len().saturating_sub(1) as u32),
                        mass,
                    )
                }
            (ModificationSpecificity::ProteinN(None), Position::Nterm | Position::Full) => {
                self.apply_site(Site::Nterm, mass)
            }
            (ModificationSpecificity::ProteinN(Some(resi)), Position::Nterm | Position::Full)
            if resi == *self.sequence.first().unwrap_or(&0) =>
                {
                    self.apply_site(Site::Sequence(0), mass)
                }
            (ModificationSpecificity::ProteinC(None), Position::Cterm | Position::Full) => {
                self.apply_site(Site::Cterm, mass)
            }
            (ModificationSpecificity::ProteinC(Some(resi)), Position::Cterm | Position::Full)
            if resi == *self.sequence.last().unwrap_or(&0) =>
                {
                    self.apply_site(
                        Site::Sequence(self.sequence.len().saturating_sub(1) as u32),
                        mass,
                    )
                }
            (ModificationSpecificity::Residue(resi), _) => {
                // Collect first to avoid borrowing `self.sequence` while mutating
                // `self.modifications`; set_if_unmodified preserves the guard.
                let positions: SmallVec<[usize; 8]> = self
                    .sequence
                    .iter()
                    .enumerate()
                    .filter_map(|(idx, residue)| (resi == *residue).then_some(idx))
                    .collect();
                for idx in positions {
                    self.modifications.set_if_unmodified(idx, mass);
                }
            }
            _ => {}
        }
    }

    /// Apply variable modifications, then static modifications to a peptide
    pub fn apply(
        mut self,
        variable_mods: &[(ModificationSpecificity, f32)],
        static_mods: &HashMap<ModificationSpecificity, f32>,
        combinations: usize,
    ) -> Vec<Peptide> {
        if variable_mods.is_empty() {
            for (target, mass) in static_mods {
                self.static_mods(*target, *mass);
            }
            self.monoisotopic += self.modification_mass();
            vec![self]
        } else {
            let mut mods = Vec::new();
            for (residue, mass) in variable_mods.iter() {
                self.push_resi(&mut mods, *residue, *mass);
            }

            let mut modified = Vec::new();
            modified.push(self.clone());

            for n in 1..=combinations {
                'next: for combination in mods.iter().combinations(n).filter(no_duplicates) {
                    let mut set = FnvHashSet::default();
                    for (site, _) in &combination {
                        if !set.insert(*site) {
                            continue 'next;
                        }
                    }
                    let mut peptide = self.clone();
                    for (site, mass) in combination {
                        peptide.apply_site(*site, *mass);
                    }
                    modified.push(peptide);
                }
            }

            // Apply static mods to all peptides
            for peptide in modified.iter_mut() {
                for (target, mass) in static_mods {
                    peptide.static_mods(*target, *mass);
                }
                peptide.monoisotopic += peptide.modification_mass();
            }

            modified
        }
    }

    pub fn reverse(&self, keep_ends: bool) -> Peptide {
        let mut pep = self.clone();
        pep.decoy = !self.decoy;
        let n = pep.sequence.len();
        if n > 1 {
            let mut s = Vec::from(pep.sequence.as_ref());
            // Materialise dense, reuse the proven dense remap, re-canonicalise.
            let mut m = pep.modifications.to_dense(n);

            if keep_ends {
                let n_sub_1 = n.saturating_sub(1);
                if n_sub_1 > 1 {
                    // only reverse the internal sequence, tryptic cleavage motive stays preserved
                    s[1..n_sub_1].reverse();
                    m[1..n_sub_1].reverse();
                }
            } else {
                // reverse the entire sequence (HLA?)
                s.reverse();
                m.reverse();
            }

            pep.sequence = Arc::from(s.into_boxed_slice());
            pep.modifications = Mods::from_dense(&m);
        }
        pep
    }

    pub fn shuffle(&self, keep_ends: bool) -> Peptide {
        let mut pep = self.clone();
        pep.decoy = !pep.decoy;
        let n = pep.sequence.len();
        if n > 1 {
            let mut s = Vec::from(pep.sequence.as_ref());
            // Materialise dense, reuse the proven dense remap, re-canonicalise.
            let mut m = pep.modifications.to_dense(n);
            let mut rng = thread_rng();

            if keep_ends {
                if n > 2 {
                    let mut indices: Vec<usize> = (1..n-1).collect();
                    indices.shuffle(&mut rng);

                    let mut s_shuffled = s.clone();
                    let mut m_shuffled = m.clone();
                    for (i, &idx) in indices.iter().enumerate() {
                        s_shuffled[i + 1] = s[idx];
                        m_shuffled[i + 1] = m[idx];
                    }
                    s[1..n-1].copy_from_slice(&s_shuffled[1..n-1]);
                    m[1..n-1].copy_from_slice(&m_shuffled[1..n-1]);
                }
            } else {
                let mut indices: Vec<usize> = (0..n).collect();
                indices.shuffle(&mut thread_rng());

                let mut s_shuffled = s.clone();
                let mut m_shuffled = m.clone();
                for (i, &idx) in indices.iter().enumerate() {
                    s_shuffled[i] = s[idx];
                    m_shuffled[i] = m[idx];
                }
                s = s_shuffled;
                m = m_shuffled;
            }

            pep.sequence = Arc::from(s.into_boxed_slice());
            pep.modifications = Mods::from_dense(&m);
        }
        pep
    }
}

fn no_duplicates(combination: &Vec<&(Site, f32)>) -> bool {
    let mut n = 0;
    let mut c = 0;
    for (site, _) in combination {
        match site {
            Site::Nterm => n += 1,
            Site::Cterm => c += 1,
            _ => {}
        }
    }

    n <= 1 && c <= 1
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
enum Site {
    Nterm,
    Cterm,
    Sequence(u32),
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum PeptideError {
    InvalidSequence(String),
}

impl TryFrom<DigestGroup> for Peptide {
    type Error = PeptideError;

    fn try_from(value: DigestGroup) -> Result<Self, Self::Error> {
        let mut pep = Peptide::try_from(value.reference)?;
        pep.proteins = value.proteins;
        Ok(pep)
    }
}

impl TryFrom<Digest> for Peptide {
    type Error = PeptideError;

    fn try_from(value: Digest) -> Result<Self, Self::Error> {
        let mut mass = H2O;
        // This is an important invariant to enforce, that ensures safety
        // while reversing peptide sequences
        if !value.sequence.is_ascii() {
            return Err(PeptideError::InvalidSequence(value.sequence));
        }
        for c in value.sequence.as_bytes() {
            let mono = monoisotopic(*c);
            if mono == 0.0 {
                return Err(PeptideError::InvalidSequence(value.sequence));
            }
            mass += mono;
        }

        Ok(Peptide {
            decoy: value.decoy,
            position: value.position,
            modifications: Mods::default(),
            sequence: Arc::from(value.sequence.into_bytes().into_boxed_slice()),
            monoisotopic: mass,
            nterm: None,
            cterm: None,
            missed_cleavages: value.missed_cleavages,
            semi_enzymatic: value.semi_enzymatic,
            proteins: vec![value.protein],
        })
    }
}

impl std::fmt::Display for Peptide {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if let Some(m) = self.nterm {
            write!(f, "[{:+}]-", m)?;
        }
        for (i, c) in self.sequence.iter().enumerate() {
            let m = self.modifications.mass_at(i);
            if m != 0.0 {
                write!(f, "{}[{:+}]", *c as char, m)?;
            } else {
                write!(f, "{}", *c as char)?;
            }
        }
        if let Some(m) = self.cterm {
            write!(f, "-[{:+}]", m)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod test {
    use crate::enzyme::{Enzyme, EnzymeParameters};

    use super::*;

    // ---- Mods sparse-representation correctness gate (Codex review) ----
    // The sparse Mods must reproduce the historical dense Vec<f32> semantics
    // exactly: equality (for dedup) and ordering (for initial_sort -> PeptideIx).

    fn dense_cmp(a: &[f32], b: &[f32]) -> Ordering {
        a.partial_cmp(b).unwrap_or(Ordering::Equal)
    }

    #[test]
    fn mods_roundtrip_and_mass_at() {
        let dense = vec![0.0, 79.96633, 0.0, 0.0, 15.994915, 0.0];
        let m = Mods::from_dense(&dense);
        // mass_at reproduces dense indexing
        for (i, &v) in dense.iter().enumerate() {
            assert_eq!(m.mass_at(i), v, "mass_at({i})");
        }
        // to_dense round-trips
        assert_eq!(m.to_dense(dense.len()), dense);
        // total matches dense sum
        assert!((m.total() - dense.iter().sum::<f32>()).abs() < 1e-6);
    }

    #[test]
    fn mods_empty_equals_all_zeros() {
        // canonical invariant: unmodified (empty) compares equal to a dense all-zero
        let empty = Mods::default();
        let from_zeros = Mods::from_dense(&[0.0, 0.0, 0.0]);
        assert_eq!(empty, from_zeros, "all-zero dense must canonicalize to empty");
        assert_eq!(empty.cmp_dense(&from_zeros), Ordering::Equal);
    }

    #[test]
    fn mods_set_if_unmodified_guard() {
        let mut m = Mods::default();
        m.set_if_unmodified(2, 10.0);
        m.set_if_unmodified(2, 99.0); // already modified -> no-op (mirrors `== 0.0` guard)
        assert_eq!(m.mass_at(2), 10.0);
        m.set_if_unmodified(0, -17.0); // negative mass (pyro-glu) must be supported + sorted
        assert_eq!(m.to_dense(3), vec![-17.0, 0.0, 10.0]);
    }

    #[test]
    fn mods_zero_mass_never_stored() {
        // Self-enforcing canonical invariant: a 0.0 set is a no-op, so it can
        // never break derived-PartialEq dedup against an unmodified peptide.
        let mut m = Mods::default();
        m.set_if_unmodified(1, 0.0);
        assert!(m.is_empty(), "zero-mass set must not create an entry");
        assert_eq!(m, Mods::default());
    }

    #[quickcheck_macros::quickcheck]
    fn mods_cmp_matches_dense(a: Vec<u8>, b: Vec<u8>) -> bool {
        // Build equal-length dense vectors from small byte payloads mapped to a
        // few realistic mod masses incl. 0.0 and a negative; compare sparse vs
        // dense ordering AND equality. Only the equal-length case is meaningful
        // (initial_sort/dedup compare same-sequence => same-length peptides).
        let masses = [0.0f32, 15.994915, 79.96633, -17.026549, 42.010565];
        let n = a.len().min(b.len());
        let da: Vec<f32> = a[..n].iter().map(|&x| masses[(x as usize) % masses.len()]).collect();
        let db: Vec<f32> = b[..n].iter().map(|&x| masses[(x as usize) % masses.len()]).collect();
        let ma = Mods::from_dense(&da);
        let mb = Mods::from_dense(&db);
        // ordering fidelity
        if ma.cmp_dense(&mb) != dense_cmp(&da, &db) {
            return false;
        }
        // equality fidelity (drives dedup correctness)
        if (ma == mb) != (da == db) {
            return false;
        }
        true
    }

    fn var_mod_sequence(
        peptide: &Peptide,
        mods: &[(ModificationSpecificity, f32)],
        combo: usize,
    ) -> Vec<String> {
        let static_mods = HashMap::default();
        peptide
            .clone()
            .apply(&mods, &static_mods, combo)
            .into_iter()
            .map(|p| p.to_string())
            .collect::<Vec<_>>()
    }

    #[test]
    fn full() {
        let sequence = "MPEPTIDEKMSAGEKEND";
        let tryp = EnzymeParameters {
            min_len: 0,
            max_len: 50,
            missed_cleavages: 0,
            enzyme: Enzyme::new("KR", "P", true, false),
        };

        let peptides = tryp
            .digest(sequence, Default::default())
            .into_iter()
            .map(Peptide::try_from)
            .collect::<Result<Vec<_>, _>>()
            .unwrap();

        assert_eq!(peptides.len(), 3);
        assert_eq!(peptides[0].to_string(), "MPEPTIDEK");
        assert_eq!(peptides[0].position, Position::Nterm);
        assert_eq!(peptides[1].to_string(), "MSAGEK");
        assert_eq!(peptides[1].position, Position::Internal);
        assert_eq!(peptides[2].to_string(), "END");
        assert_eq!(peptides[2].position, Position::Cterm);

        use ModificationSpecificity::*;

        let mods = [
            (ProteinN(None), 42.0),
            (ProteinC(None), 11.0),
            (PeptideN(None), 12.0),
            (PeptideC(None), 19.0),
        ];
        let a = var_mod_sequence(&peptides[0], &mods, 2);
        let b = var_mod_sequence(&peptides[1], &mods, 2);
        let c = var_mod_sequence(&peptides[2], &mods, 2);

        // Make sure no duplicates exist
        assert_eq!(
            a,
            vec![
                "MPEPTIDEK",
                "[+42]-MPEPTIDEK",
                "[+12]-MPEPTIDEK",
                "MPEPTIDEK-[+19]",
                "[+42]-MPEPTIDEK-[+19]",
                "[+12]-MPEPTIDEK-[+19]",
            ]
        );

        assert_eq!(
            b,
            vec![
                "MSAGEK",
                "[+12]-MSAGEK",
                "MSAGEK-[+19]",
                "[+12]-MSAGEK-[+19]",
            ]
        );

        assert_eq!(
            c,
            vec![
                "END",
                "END-[+11]",
                "[+12]-END",
                "END-[+19]",
                "[+12]-END-[+11]",
                "[+12]-END-[+19]",
            ]
        );
    }

    #[test]
    fn test_variable_mods() {
        use ModificationSpecificity::*;
        let variable_mods = [(Residue(b'M'), 16.0f32), (Residue(b'C'), 57.)];
        let peptide = Peptide::try_from(Digest {
            sequence: "GCMGCMG".into(),
            ..Default::default()
        })
        .unwrap();

        let expected = vec![
            "GCMGCMG",
            "GCM[+16]GCMG",
            "GCMGCM[+16]G",
            "GC[+57]MGCMG",
            "GCMGC[+57]MG",
            "GCM[+16]GCM[+16]G",
            "GC[+57]M[+16]GCMG",
            "GCM[+16]GC[+57]MG",
            "GC[+57]MGCM[+16]G",
            "GCMGC[+57]M[+16]G",
            "GC[+57]MGC[+57]MG",
        ];

        let peptides = var_mod_sequence(&peptide, &variable_mods, 2);
        assert_eq!(peptides, expected);
    }

    #[test]
    fn test_variable_mods_no_effeect() {
        use ModificationSpecificity::*;
        let variable_mods = [(Residue(b'M'), 16.0f32), (Residue(b'C'), 57.)];
        let peptide = Peptide::try_from(Digest {
            sequence: "AAAAAAAA".into(),
            ..Default::default()
        })
        .unwrap();

        let expected = vec!["AAAAAAAA"];
        let peptides = var_mod_sequence(&peptide, &variable_mods, 2);
        assert_eq!(peptides, expected);
    }

    #[test]
    fn test_variable_mods_nterm() {
        use ModificationSpecificity::*;
        let variable_mods = [(PeptideN(None), 42.), (Residue(b'M'), 16.)];
        let peptide = Peptide::try_from(Digest {
            sequence: "GCMGCMG".into(),
            ..Default::default()
        })
        .unwrap();

        let expected = vec![
            "GCMGCMG",
            "[+42]-GCMGCMG",
            "GCM[+16]GCMG",
            "GCMGCM[+16]G",
            "[+42]-GCM[+16]GCMG",
            "[+42]-GCMGCM[+16]G",
            "GCM[+16]GCM[+16]G",
            "[+42]-GCM[+16]GCM[+16]G",
        ];

        let peptides = var_mod_sequence(&peptide, &variable_mods, 3);
        assert_eq!(peptides, expected);
    }

    #[test]
    fn test_variable_mods_cterm() {
        use ModificationSpecificity::*;
        let variable_mods = [(PeptideC(None), 42.), (Residue(b'M'), 16.)];
        let peptide = Peptide::try_from(Digest {
            sequence: "GCMGCMG".into(),
            ..Default::default()
        })
        .unwrap();

        let expected = vec![
            "GCMGCMG",
            "GCMGCMG-[+42]",
            "GCM[+16]GCMG",
            "GCMGCM[+16]G",
            "GCM[+16]GCMG-[+42]",
            "GCMGCM[+16]G-[+42]",
            "GCM[+16]GCM[+16]G",
            "GCM[+16]GCM[+16]G-[+42]",
        ];

        let peptides = var_mod_sequence(&peptide, &variable_mods, 3);
        assert_eq!(peptides, expected);
    }

    #[test]
    fn test_variable_mods_multi() {
        use ModificationSpecificity::*;
        let variable_mods = [(Residue(b'S'), 79.), (Residue(b'S'), 541.)];
        let peptide = Peptide::try_from(Digest {
            sequence: "GGGSGGGS".into(),
            ..Default::default()
        })
        .unwrap();

        let expected = vec![
            "GGGSGGGS",
            "GGGS[+79]GGGS",
            "GGGSGGGS[+79]",
            "GGGS[+541]GGGS",
            "GGGSGGGS[+541]",
            "GGGS[+79]GGGS[+79]",
            "GGGS[+79]GGGS[+541]",
            "GGGS[+541]GGGS[+79]",
            "GGGS[+541]GGGS[+541]",
        ];

        let peptides = var_mod_sequence(&peptide, &variable_mods, 2);
        assert_eq!(peptides, expected);
    }

    /// Check that picked-peptide approach will match forward and reverse peptides
    #[test]
    fn test_psuedo_forward() {
        let trypsin = crate::enzyme::EnzymeParameters {
            missed_cleavages: 0,
            min_len: 3,
            max_len: 30,
            enzyme: Enzyme::new("KR", "P", true, false),
        };

        let fwd = "MADEEKLPPGWEKRMSRSSGRVYYFNHITNASQWERPSGN";
        for digest in trypsin.digest(fwd, Default::default()) {
            let fwd = Peptide::try_from(digest.clone()).unwrap();
            let rev = Peptide::try_from(digest.reverse()).unwrap();

            assert_eq!(fwd.decoy, false);
            assert_eq!(rev.decoy, true);
            assert!(
                fwd.sequence.len() < 4 || fwd.sequence != rev.sequence,
                "{} {}",
                fwd,
                rev
            );
            assert_eq!(rev.reverse(true).to_string(), fwd.to_string());
        }
    }

    #[test]
    fn apply_mods() {
        use ModificationSpecificity::*;
        let peptide = Peptide::try_from(Digest {
            sequence: "AACAACAA".into(),
            ..Default::default()
        })
        .unwrap();

        let expected = vec![
            "AAC[+57]AAC[+57]AA",
            "AAC[+30]AAC[+57]AA",
            "AAC[+57]AAC[+30]AA",
            "AAC[+30]AAC[+30]AA",
        ];

        let mut static_mods = HashMap::new();
        static_mods.insert(Residue(b'C'), 57.0);

        let variable_mods = [(Residue(b'C'), 30.0)];

        let peptides = peptide
            .apply(&variable_mods, &static_mods, 2)
            .into_iter()
            .map(|p| p.to_string())
            .collect::<Vec<_>>();

        assert_eq!(peptides, expected);
    }

    #[test]
    fn modification_sites() {
        use Site::*;
        let peptide = Peptide::try_from(Digest {
            sequence: "AACAACAA".into(),
            ..Default::default()
        })
        .unwrap();

        let mut mods = vec![];
        peptide.push_resi(&mut mods, ModificationSpecificity::Residue(b'C'), 16.0);
        assert_eq!(mods, vec![(Sequence(2), 16.0), (Sequence(5), 16.0)]);
        mods.clear();

        peptide.push_resi(&mut mods, ModificationSpecificity::PeptideC(None), 16.0);
        assert_eq!(mods, vec![(Cterm, 16.0)]);
        mods.clear();

        peptide.push_resi(&mut mods, ModificationSpecificity::PeptideN(None), 16.0);
        assert_eq!(mods, vec![(Nterm, 16.0)]);
        mods.clear();

        let mut mods = vec![];
        for (residue, mass) in [("^", 12.0), ("$", 200.0), ("C", 57.0), ("A", 43.0)] {
            peptide.push_resi(&mut mods, residue.parse().unwrap(), mass);
        }

        assert_eq!(
            mods,
            vec![
                (Nterm, 12.0),
                (Cterm, 200.0),
                (Sequence(2), 57.0),
                (Sequence(5), 57.0),
                (Sequence(0), 43.0),
                (Sequence(1), 43.0),
                (Sequence(3), 43.0),
                (Sequence(4), 43.0),
                (Sequence(6), 43.0),
                (Sequence(7), 43.0),
            ]
        );
    }
}
