use pyo3::prelude::*;
use pyo3::types::PyDict;
use numpy::{PyArray1, PyReadonlyArray2};

/// Clamped differences a - b, shifted from [-T, T] to [0, 2T].
fn diffs(out: &mut [u8], a: &[u8], b: &[u8], t: i16) {
    for ((o, &p), &q) in out.iter_mut().zip(a).zip(b) {
        *o = ((p as i16 - q as i16).clamp(-t, t) + t) as u8;
    }
}

/// Differences of neighboring pixels, in a 2D grid with rows of the given width.
struct Plane {
    q: Vec<u8>,
    width: usize,
}

impl Plane {
    /// Plane of shape [height, width], row i holds the differences a - b of (a, b) = f(i).
    fn from_rows<'a>(height: usize, width: usize, f: impl Fn(usize) -> (&'a [u8], &'a [u8]), t: i16) -> Self {
        let mut q = vec![0u8; height * width];
        for (i, out) in q.chunks_exact_mut(width).enumerate() {
            let (a, b) = f(i);
            diffs(out, a, b, t);
        }
        Plane { q, width }
    }

    fn row(&self, i: usize) -> &[u8] {
        &self.q[i * self.width..(i + 1) * self.width]
    }
}

/// Co-occurrence counts of triplets of differences, with values in [0, D).
///
/// Counted in four copies, triplet k into copy k mod 4,
/// so that consecutive triplets falling into the same bin do not wait for each other.
struct Cooc {
    d: usize,
    copies: [Vec<u32>; 4],
}

impl Cooc {
    fn new(d: usize) -> Self {
        Cooc { d, copies: std::array::from_fn(|_| vec![0; d * d * d]) }
    }

    /// Counts the triplets (prev[k], cur[k], next[k]).
    fn add(&mut self, prev: &[u8], cur: &[u8], next: &[u8]) {
        let d = self.d;
        let n = cur.len();
        let (prev, next) = (&prev[..n], &next[..n]);
        let index = |k: usize| (prev[k] as usize * d + cur[k] as usize) * d + next[k] as usize;
        // four triplets per step, written out, so that they are counted independently
        let [c0, c1, c2, c3] = &mut self.copies;
        let mut k = 0;
        while k + 4 <= n {
            c0[index(k)] += 1;
            c1[index(k + 1)] += 1;
            c2[index(k + 2)] += 1;
            c3[index(k + 3)] += 1;
            k += 4;
        }
        for k in k..n {
            c0[index(k)] += 1;
        }
    }

    /// Normalized counts, symmetrized with the opposite direction,
    /// i.e., triplet (r, c, l) merged with (2T - l, 2T - c, 2T - r).
    fn features(&self) -> Vec<f64> {
        let d = self.d;
        let count = |r: usize, c: usize, l: usize| -> u32 {
            let index = (r * d + c) * d + l;
            self.copies.iter().map(|copy| copy[index]).sum()
        };
        let mut f = Vec::with_capacity(d * d * d);
        for r in 0..d {
            for c in 0..d {
                for l in 0..d {
                    f.push((count(r, c, l) + count(d - 1 - l, d - 1 - c, d - 1 - r)) as f64);
                }
            }
        }
        let total: f64 = f.iter().sum();
        f.iter().map(|v| v / total).collect()
    }
}

/// Normalized, symmetrized straight and diagonal co-occurrences of a row-major image of shape [h, w].
fn compute(x: &[u8], h: usize, w: usize, t: i16) -> (Vec<f64>, Vec<f64>) {
    let d = (2 * t + 1) as usize;
    let row = |i: usize| &x[i * w..(i + 1) * w];
    let mut straight = Cooc::new(d);
    let mut diagonal = Cooc::new(d);

    // horizontal, q[i, j] = x[i, j] - x[i, j + 1]
    if w >= 3 {
        let q = Plane::from_rows(h, w - 1, |i| (&row(i)[..w - 1], &row(i)[1..]), t);
        for i in 0..h {
            straight.add(&q.row(i)[..w - 3], &q.row(i)[1..w - 2], &q.row(i)[2..]);
        }
    }
    // vertical, q[i, j] = x[i, j] - x[i + 1, j]
    if h >= 3 {
        let q = Plane::from_rows(h - 1, w, |i| (row(i), row(i + 1)), t);
        for i in 1..h - 2 {
            straight.add(q.row(i - 1), q.row(i), q.row(i + 1));
        }
    }
    if h >= 3 && w >= 3 {
        // diagonal, q[i, j] = x[i, j] - x[i + 1, j + 1]
        let q = Plane::from_rows(h - 1, w - 1, |i| (&row(i)[..w - 1], &row(i + 1)[1..]), t);
        for i in 1..h - 2 {
            diagonal.add(&q.row(i - 1)[..w - 3], &q.row(i)[1..w - 2], &q.row(i + 1)[2..]);
        }
        // minor diagonal, q[i, j] = x[i + 1, j] - x[i, j + 1]
        let q = Plane::from_rows(h - 1, w - 1, |i| (&row(i + 1)[..w - 1], &row(i)[1..]), t);
        for i in 1..h - 2 {
            diagonal.add(&q.row(i + 1)[..w - 3], &q.row(i)[1..w - 2], &q.row(i - 1)[2..]);
        }
    }
    (straight.features(), diagonal.features())
}

/// Extract co-occurrence features from a 2D array of u8 values.
///
/// Parameters
/// ----------
/// x1 : np.ndarray
///     A 2D NumPy array of unsigned 8-bit integers representing the image.
/// T : int, optional
///     Truncation threshold. Defaults to 3.
///
/// Returns
/// -------
/// dict
///     Dictionary with normalized co-occurrence features:
///     - "straight": 1D NumPy array of straight co-occurrences
///     - "diagonal": 1D NumPy array of diagonal co-occurrences
#[pyfunction]
#[pyo3(signature = (x1, T = 3, rounded = false))]
#[allow(non_snake_case, unused_variables)]
fn extract<'py>(py: Python<'py>, x1: PyReadonlyArray2<'py, u8>, T: Option<i16>, rounded: bool) -> PyResult<Bound<'py, PyDict>> {
    let t = T.unwrap_or(3);
    // row-major image
    let view = x1.as_array();
    let (h, w) = (view.shape()[0], view.shape()[1]);
    let image = view.as_standard_layout();
    let x = image.as_slice().unwrap();
    // compute without holding the GIL, so that threads can extract features in parallel
    let (straight, diagonal) = py.detach(|| compute(x, h, w, t));

    // Create a dictionary
    let features = PyDict::new(py);
    features.set_item("straight", PyArray1::from_vec(py, straight))?;
    features.set_item("diagonal", PyArray1::from_vec(py, diagonal))?;
    Ok(features)
}

#[pymodule]
pub fn spam(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(extract, m)?)?;
    Ok(())
}
