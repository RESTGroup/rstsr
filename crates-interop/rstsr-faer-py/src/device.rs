//! Device object: a single `Device("cpu")` singleton backing onto DeviceFaer.

use pyo3::prelude::*;

#[pyclass(frozen, module = "rstsr_faer.rstsr_faer")]
pub struct Device;

#[pymethods]
impl Device {
    pub fn __repr__(&self) -> &'static str {
        "Device(\"cpu\")"
    }
}

/// Register the device singleton as module attribute `device_cpu`.
pub fn add_device_object(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let obj = Py::new(m.py(), Device)?;
    m.add("device_cpu", obj)?;
    Ok(())
}
