use num::Complex;

pub fn get_resources_dir() -> std::path::PathBuf {
    let mut d = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    d.push("resources");
    d
}

pub trait ResourceVecAPI<T> {
    fn get_vec(c: char) -> Vec<T>;
}

pub fn get_vec<T>(c: char) -> Vec<T>
where
    (): ResourceVecAPI<T>,
{
    <() as ResourceVecAPI<T>>::get_vec(c)
}

impl ResourceVecAPI<f32> for () {
    fn get_vec(c: char) -> Vec<f32> {
        let mut npy_path = get_resources_dir();

        match c {
            'a' => npy_path.push("a-f32.npy"),
            'b' => npy_path.push("b-f32.npy"),
            'c' => npy_path.push("c-f32.npy"),
            _ => panic!("Invalid character: {}", c),
        };

        let bytes = std::fs::read(npy_path).unwrap();
        let reader = npyz::NpyFile::new(&bytes[..]).unwrap();
        reader.into_vec::<f32>().unwrap()
    }
}

impl ResourceVecAPI<f64> for () {
    fn get_vec(c: char) -> Vec<f64> {
        let mut npy_path = get_resources_dir();

        match c {
            'a' => npy_path.push("a-f64.npy"),
            'b' => npy_path.push("b-f64.npy"),
            'c' => npy_path.push("c-f64.npy"),
            _ => panic!("Invalid character: {}", c),
        };

        let bytes = std::fs::read(npy_path).unwrap();
        let reader = npyz::NpyFile::new(&bytes[..]).unwrap();
        reader.into_vec::<f64>().unwrap()
    }
}

impl ResourceVecAPI<Complex<f32>> for () {
    fn get_vec(c: char) -> Vec<Complex<f32>> {
        let mut npy_path = get_resources_dir();

        match c {
            'a' => npy_path.push("a-c32.npy"),
            'b' => npy_path.push("b-c32.npy"),
            'c' => npy_path.push("c-c32.npy"),
            _ => panic!("Invalid character: {}", c),
        };

        let bytes = std::fs::read(npy_path).unwrap();
        let reader = npyz::NpyFile::new(&bytes[..]).unwrap();
        reader.into_vec::<Complex<f32>>().unwrap()
    }
}

impl ResourceVecAPI<Complex<f64>> for () {
    fn get_vec(c: char) -> Vec<Complex<f64>> {
        let mut npy_path = get_resources_dir();

        match c {
            'a' => npy_path.push("a-c64.npy"),
            'b' => npy_path.push("b-c64.npy"),
            'c' => npy_path.push("c-c64.npy"),
            _ => panic!("Invalid character: {}", c),
        };

        let bytes = std::fs::read(npy_path).unwrap();
        let reader = npyz::NpyFile::new(&bytes[..]).unwrap();
        reader.into_vec::<Complex<f64>>().unwrap()
    }
}

#[test]
fn playground() {
    use std::path::PathBuf;

    let mut d = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    d.push("resources");

    println!("{}", d.display());

    let v = get_vec::<Complex<f32>>('a');
    println!("{:?}", v[1]);
}
