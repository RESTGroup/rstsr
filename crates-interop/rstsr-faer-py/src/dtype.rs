//! Dtype singletons: the 13 canonical array-API dtypes as identity objects.

use pyo3::prelude::*;

/// Identity-based dtype object; exactly one instance per canonical dtype,
/// created once at module init.
#[pyclass(frozen, module = "rstsr_faer.rstsr_faer")]
pub struct Dtype {
    #[pyo3(get)]
    pub name: &'static str,
}

#[pymethods]
impl Dtype {
    pub fn __repr__(&self) -> String {
        self.name.to_string()
    }
}

macro_rules! for_each_dtype {
    ($mac:ident) => {
        $mac!(bool, "bool");
        $mac!(int8, "int8");
        $mac!(int16, "int16");
        $mac!(int32, "int32");
        $mac!(int64, "int64");
        $mac!(uint8, "uint8");
        $mac!(uint16, "uint16");
        $mac!(uint32, "uint32");
        $mac!(uint64, "uint64");
        $mac!(float32, "float32");
        $mac!(float64, "float64");
        $mac!(complex64, "complex64");
        $mac!(complex128, "complex128");
    };
}


/// Register all 13 dtype singletons as module attributes.
pub fn add_dtype_objects(m: &Bound<'_, PyModule>) -> PyResult<()> {
    macro_rules! add {
        ($attr:ident, $name:literal) => {{
            let obj = Py::new(m.py(), Dtype { name: $name })?;
            m.add(stringify!($attr), obj)?;
        }};
    }
    for_each_dtype!(add);
    Ok(())
}
