#[cfg(test)]
mod test {
    use rstsr_core::prelude::*;
    use rstsr_openblas::device::DeviceOpenBLAS;

    #[test]
    fn workable() {
        let device = DeviceOpenBLAS::new(16);
        let a = rt::arange((10usize, &device));
        println!("{:?}", a);
    }
}
