use crate::prelude_dev::*;
use rstsr_openblas_ffi::{cblas, ffi};

pub fn gemm_blas_no_conj_f32(
    c: &mut [f32],
    lc: &Layout<Ix2>,
    a: &[f32],
    la: &Layout<Ix2>,
    b: &[f32],
    lb: &Layout<Ix2>,
    alpha: f32,
    beta: f32,
    nthreads: usize,
) -> Result<()> {
    // nthreads is only used for `assign_arbitary_cpu_rayon`.
    // the threading of openblas should be handled outside this function.

    // we assume output layout is c-prefer or f-prefer
    debug_assert!(lc.c_prefer() || lc.f_prefer());

    // change to f-contig anyway
    // we do not handle conj, so this can be done easily
    if !lc.f_prefer() {
        return gemm_blas_no_conj_f32(
            c,
            &lc.reverse_axes(),
            b,
            &lb.reverse_axes(),
            a,
            &la.reverse_axes(),
            alpha,
            beta,
            nthreads,
        );
    }

    // we assume that the layout is correct
    let sc = lc.shape();
    let sa = la.shape();
    let sb = lb.shape();
    debug_assert_eq!(sc[0], sa[0]);
    debug_assert_eq!(sa[1], sb[0]);
    debug_assert_eq!(sc[1], sb[1]);

    let m = sc[0];
    let n = sc[1];
    let k = sa[1];

    // determine trans/layout and clone data if necessary
    let mut a_data: Option<Vec<f32>> = None;
    let mut b_data: Option<Vec<f32>> = None;
    let (a_trans, la) = if la.f_prefer() {
        (cblas::NoTrans, la.clone())
    } else if la.c_prefer() {
        (cblas::Trans, la.reverse_axes())
    } else {
        let len = la.size();
        a_data = unsafe {
            let mut a_vec = Vec::with_capacity(len);
            a_vec.set_len(len);
            Some(a_vec)
        };
        let la_data = la.shape().new_f_contig(None);
        assign_arbitary_cpu_rayon(a_data.as_mut().unwrap(), &la_data, a, &la, nthreads)?;
        (cblas::NoTrans, la_data)
    };
    let (b_trans, lb) = if lb.f_prefer() {
        (cblas::NoTrans, lb.clone())
    } else if lb.c_prefer() {
        (cblas::Trans, lb.reverse_axes())
    } else {
        let len = lb.size();
        b_data = unsafe {
            let mut b_vec = Vec::with_capacity(len);
            b_vec.set_len(len);
            Some(b_vec)
        };
        let lb_data = lb.shape().new_f_contig(None);
        assign_arbitary_cpu_rayon(b_data.as_mut().unwrap(), &lb_data, b, &lb, nthreads)?;
        (cblas::NoTrans, lb_data)
    };

    // final configuration
    let lda = la.stride()[1];
    let ldb = lb.stride()[1];
    let ldc = lc.stride()[1];
    println!("lda: {}, ldb: {}, ldc: {}", lda, ldb, ldc);
    let ptr_c = unsafe { c.as_mut_ptr().add(lc.offset()) };
    let ptr_a = if let Some(a_data) = a_data.as_ref() {
        a_data.as_ptr()
    } else {
        unsafe { a.as_ptr().add(la.offset()) }
    };
    let ptr_b = if let Some(b_data) = b_data.as_ref() {
        b_data.as_ptr()
    } else {
        unsafe { b.as_ptr().add(lb.offset()) }
    };

    // actual computation
    unsafe {
        ffi::cblas::cblas_sgemm(
            cblas::ColMajor as ffi::cblas::CBLAS_ORDER,
            a_trans as ffi::cblas::CBLAS_TRANSPOSE,
            b_trans as ffi::cblas::CBLAS_TRANSPOSE,
            m as ffi::cblas::blasint,
            n as ffi::cblas::blasint,
            k as ffi::cblas::blasint,
            alpha,
            ptr_a,
            lda as ffi::cblas::blasint,
            ptr_b,
            ldb as ffi::cblas::blasint,
            beta,
            ptr_c,
            ldc as ffi::cblas::blasint,
        );
    }
    Ok(())
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn playground() {
        let a = vec![1., 2., 3., 4., 5., 6.];
        let b = vec![1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.];
        let mut c = vec![0.0; 16];

        let la = [2, 3].c();
        let lb = [3, 4].c();
        let lc = [2, 4].c();
        gemm_blas_no_conj_f32(&mut c, &lc, &a, &la, &b, &lb, 1.0, 0.0, 16).unwrap();
        let c_tsr = TensorView::new(asarray(&c).into_raw_parts().0, lc);
        println!("{:}", c_tsr);
        println!("{:}", c_tsr.reshape([8]));
        let c_ref = asarray(vec![38., 44., 50., 56., 83., 98., 113., 128.]);
        assert!(allclose_f64(&c_tsr.map(|v| *v as f64), &c_ref));

        let la = [2, 3].c();
        let lb = [3, 4].c();
        let lc = [2, 4].f();
        gemm_blas_no_conj_f32(&mut c, &lc, &a, &la, &b, &lb, 1.0, 0.0, 16).unwrap();
        let c_tsr = TensorView::new(asarray(&c).into_raw_parts().0, lc);
        println!("{:}", c_tsr);
        println!("{:}", c_tsr.reshape([8]));
        let c_ref = asarray(vec![38., 44., 50., 56., 83., 98., 113., 128.]);
        assert!(allclose_f64(&c_tsr.map(|v| *v as f64), &c_ref));

        let la = [2, 3].f();
        let lb = [3, 4].c();
        let lc = [2, 4].c();
        gemm_blas_no_conj_f32(&mut c, &lc, &a, &la, &b, &lb, 1.0, 0.0, 16).unwrap();
        let c_tsr = TensorView::new(asarray(&c).into_raw_parts().0, lc);
        println!("{:}", c_tsr);
        println!("{:}", c_tsr.reshape([8]));
        let c_ref = asarray(vec![61., 70., 79., 88., 76., 88., 100., 112.]);
        assert!(allclose_f64(&c_tsr.map(|v| *v as f64), &c_ref));

        let la = [2, 3].f();
        let lb = [3, 4].c();
        let lc = [2, 4].f();
        gemm_blas_no_conj_f32(&mut c, &lc, &a, &la, &b, &lb, 2.0, 0.0, 16).unwrap();
        let c_tsr = TensorView::new(asarray(&c).into_raw_parts().0, lc);
        println!("{:}", c_tsr);
        println!("{:}", c_tsr.reshape([8]));
        let c_ref = 2 * asarray(vec![61., 70., 79., 88., 76., 88., 100., 112.]);
        assert!(allclose_f64(&c_tsr.map(|v| *v as f64), &c_ref));
    }
}
