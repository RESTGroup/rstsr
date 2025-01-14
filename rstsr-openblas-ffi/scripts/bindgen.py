# # Bindgen of `cblas.h`

import subprocess
import os

# ## Copy OpenBLAS header files

os.system("cp $HOME/Software/OpenBLAS-0.3.28/include/openblas/* .")

# ## Change OpenBLAS header files

for token in [
    "s/#define lapack_int        int64_t/typedef int64_t lapack_int;/g",
    "s/#define lapack_int        int32_t/typedef int32_t lapack_int;/g",
]:
    subprocess.run(["sed", "-i", token, "lapack.h"])

for token in [
    "s/#define LAPACK_ROW_MAJOR               101/const int LAPACK_ROW_MAJOR = 101;/g",
    "s/#define LAPACK_COL_MAJOR               102/const int LAPACK_COL_MAJOR = 102;/g",
]:
    subprocess.run(["sed", "-i", token, "lapacke.h"])

# ## Bindgen

subprocess.run([
    "bindgen", "cblas.h",
    "-o", "cblas.rs",
    "--allowlist-file", "cblas.h",
    "--no-layout-tests",
    "--use-core",
    "--merge-extern-blocks",
])

subprocess.run([
    "bindgen", "lapacke.h",
    "-o", "lapacke.rs",
    "--allowlist-file", "lapacke.h",
    "--no-layout-tests",
    "--use-core",
    "--merge-extern-blocks",
])

# ## Post-process

# ### cblas.rs

with open("cblas.rs", "r") as f:
    token = f.read()

token = token.replace("pub type blasint = ::core::ffi::c_int;", "")
token = token.replace("::core::ffi::c_char", "c_char")
token = token.replace("::core::ffi::c_void", "c_void")
token = token.replace("::core::ffi::c_int", "c_int")
token = token.replace("::core::ffi::c_uint", "c_uint")
token = token.replace("::core::ffi::c_ulong", "c_ulong")
token = token.replace("::core::option::Option", "Option")

token = """
#![allow(non_camel_case_types)]
#![allow(non_upper_case_globals)]

use core::ffi::{c_char, c_void, c_int, c_uint, c_ulong};

#[cfg(not(feature = "ilp64"))]
pub type blasint = i32;
#[cfg(feature = "ilp64")]
pub type blasint = i64;
""".strip() + "\n\n" + token

with open("cblas.rs", "w") as f:
    f.write(token)

subprocess.run(["rustfmt", "cblas.rs"])

# ### lapacke.rs

with open("lapacke.rs", "r") as f:
    token = f.read()

token = token.replace("pub type lapack_int = i32;", "")
token = token.replace("::core::ffi::c_char", "c_char")
token = token.replace("::core::ffi::c_void", "c_void")
token = token.replace("::core::ffi::c_int", "c_int")
token = token.replace("::core::option::Option", "Option")

token = """
#![allow(non_camel_case_types)]
#![allow(non_upper_case_globals)]

use core::ffi::{c_char, c_int};

#[cfg(not(feature = "ilp64"))]
pub type lapack_int = i32;
#[cfg(feature = "ilp64")]
pub type lapack_int = i64;
""".strip() + "\n\n" + token

with open("lapacke.rs", "w") as f:
    f.write(token)

subprocess.run(["rustfmt", "lapacke.rs"])

# ## Move file to src

for f in ["cblas.rs", "lapacke.rs"]:
    subprocess.run(["mv", f, "../src/ffi"])
