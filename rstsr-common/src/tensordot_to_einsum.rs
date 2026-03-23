use crate::prelude_dev::*;

/// Converts `tensordot` parameters to an equivalent `einsum` subscript string.
///
/// # Arguments
/// * `dim_a` – number of dimensions of the first tensor.
/// * `dim_b` – number of dimensions of the second tensor.
/// * `axes` – a pair of vectors specifying which axes to sum over. The first vector contains axes
///   of the first tensor, the second vector contains corresponding axes of the second tensor.
///   Default value is `2`.
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
    axes: impl TryInto<AxesPairIndex<isize>, Error: Into<Error>>,
) -> Result<String> {
    let mut axes = axes.try_into().map_err(Into::into)?;
    if axes == AxesPairIndex::None {
        axes = AxesPairIndex::Val(2);
    }

    let (axes_a, axes_b): (Vec<usize>, Vec<usize>) = match axes {
        AxesPairIndex::None => unreachable!("this have been handled above"),
        AxesPairIndex::Pair(axes_a, axes_b) => (
            normalize_axes_index(axes_a, dim_a, false, false)?.into_iter().map(|x| x as usize).collect(),
            normalize_axes_index(axes_b, dim_b, false, false)?.into_iter().map(|x| x as usize).collect(),
        ),
        AxesPairIndex::Val(n) => {
            if n < 0 {
                return rstsr_raise!(InvalidValue, "n must be non-negative")?;
            }
            let n = n as usize;
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
