use crate::prelude_dev::*;

pub enum TensorDotAxes {
    /// The axes are specified as a pair of vectors, one for each tensor.
    /// Each vector contains the indices of the axes to sum over for that tensor.
    /// The two vectors must have the same length, and the i-th element of the first vector
    /// corresponds to the i-th element of the second vector.
    PairOfVecs(Vec<isize>, Vec<isize>),

    /// The axes are specified as a single integer `n`, which means to sum over the last `n` axes
    /// of the first tensor and the first `n` axes of the second tensor.
    Int(usize),
}

impl<X1, X2> TryFrom<(X1, X2)> for TensorDotAxes
where
    X1: TryInto<AxesIndex<isize>, Error: Into<Error>>,
    X2: TryInto<AxesIndex<isize>, Error: Into<Error>>,
{
    type Error = Error;

    fn try_from(value: (X1, X2)) -> Result<Self> {
        let axes_a = value.0.try_into().map_err(Into::into)?.as_ref().to_vec();
        let axes_b = value.1.try_into().map_err(Into::into)?.as_ref().to_vec();
        Ok(TensorDotAxes::PairOfVecs(axes_a, axes_b))
    }
}

impl From<i32> for TensorDotAxes {
    fn from(n: i32) -> Self {
        TensorDotAxes::Int(n as usize)
    }
}

impl From<isize> for TensorDotAxes {
    fn from(n: isize) -> Self {
        TensorDotAxes::Int(n as usize)
    }
}

/// Converts `tensordot` parameters to an equivalent `einsum` subscript string.
///
/// # Arguments
/// * `dim_a` – number of dimensions of the first tensor.
/// * `dim_b` – number of dimensions of the second tensor.
/// * `axes` – a pair of vectors specifying which axes to sum over. The first vector contains axes
///   of the first tensor, the second vector contains corresponding axes of the second tensor.
///
/// # Example
/// ```
/// let s = tensordot_to_einsum_str(2, 2, (vec![1], vec![0]));
/// assert_eq!(s, "ba,ac->bc");  // equivalent to "ik,kj->ij"
/// ```
#[allow(clippy::needless_range_loop)]
pub fn tensordot_to_einsum_str(
    dim_a: usize,
    dim_b: usize,
    axes: impl TryInto<TensorDotAxes, Error: Into<Error>>,
) -> Result<String> {
    let axes = axes.try_into().map_err(Into::into)?;

    let (axes_a, axes_b): (Vec<usize>, Vec<usize>) = match axes {
        TensorDotAxes::PairOfVecs(axes_a, axes_b) => (
            normalize_axes_index(axes_a.into(), dim_a, false, false)?.into_iter().map(|x| x as usize).collect(),
            normalize_axes_index(axes_b.into(), dim_b, false, false)?.into_iter().map(|x| x as usize).collect(),
        ),
        TensorDotAxes::Int(n) => {
            if n > dim_a || n > dim_b {
                return rstsr_raise!(
                    InvalidLayout,
                    "n must be less than or equal to the number of dimensions of both tensors"
                )?;
            }
            let axes_a = (dim_a - n..dim_a).collect();
            let axes_b = (0..n).collect();
            (axes_a, axes_b)
        },
    };

    assert_eq!(axes_a.len(), axes_b.len(), "axes must have same length");

    // Validate axis indices
    for &i in &axes_a {
        assert!(i < dim_a, "axis {} out of bounds for tensor A", i);
    }
    for &j in &axes_b {
        assert!(j < dim_b, "axis {} out of bounds for tensor B", j);
    }

    // Label generator: a..z then A..Z (enough for most practical uses)
    let mut label_gen = (b'a'..=b'z').chain(b'A'..=b'Z').map(|c| c as char);

    // Storage for labels assigned to each dimension (None = not yet assigned)
    let mut a_labels = vec![None; dim_a];
    let mut b_labels = vec![None; dim_b];

    // 1. Assign the same label to each summed pair of axes
    for (&i, &j) in axes_a.iter().zip(axes_b.iter()) {
        let label = label_gen.next().ok_or_else(|| rstsr_error!(InvalidValue, "ran out of unique labels"))?;
        a_labels[i] = Some(label);
        b_labels[j] = Some(label);
    }

    // 2. Assign labels to the remaining (non‑summed) axes of A
    for i in 0..dim_a {
        if a_labels[i].is_none() {
            let label = label_gen.next().ok_or_else(|| rstsr_error!(InvalidValue, "ran out of unique labels"))?;
            a_labels[i] = Some(label);
        }
    }

    // 3. Assign labels to the remaining axes of B
    for j in 0..dim_b {
        if b_labels[j].is_none() {
            let label = label_gen.next().ok_or_else(|| rstsr_error!(InvalidValue, "ran out of unique labels"))?;
            b_labels[j] = Some(label);
        }
    }

    // Build the subscript string for A (all labels in order)
    let a_str: String = a_labels.iter().map(|opt| opt.unwrap()).collect();

    // Build the subscript string for B (all labels in order)
    let b_str: String = b_labels.iter().map(|opt| opt.unwrap()).collect();

    // Output labels: first all non‑summed axes of A, then all non‑summed axes of B
    let mut out_labels = Vec::new();
    for i in 0..dim_a {
        if !axes_a.contains(&i) {
            out_labels.push(a_labels[i].unwrap());
        }
    }
    for j in 0..dim_b {
        if !axes_b.contains(&j) {
            out_labels.push(b_labels[j].unwrap());
        }
    }
    let out_str: String = out_labels.into_iter().collect();

    Ok(format!("{}, {} -> {}", a_str, b_str, out_str))
}
