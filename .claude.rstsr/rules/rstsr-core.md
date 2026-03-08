---
paths:
  - "rstsr-core/**"
description: rstsr-core function impl/doc/test instructions
---

# What User Is Suggested to Do

- If new function to be implemented, user should provide the category of the implemented function (though this can also be inferred by AI agent if that's simple)
  - tensor function (`slice`, `reshape`, `transpose`, etc, using existed device trait operations);
  - operation function (`add`, `sin`, `isclose`, etc, also need to implement device trait operations);

# General Workflow

1. Enhance yourself (AI agent) for understanding of the function to be implemented.
  - especially numpy (and scipy if the function is not in NumPy); for API documentation, you are also referred to array-api.
  - External repositories (NumPy, SciPy, array-api, etc) should be at relative path to rstsr's working directory `../other-repos/<repo-name>`.

2. Tensor Function Implementation
  - Most implementations should be at `rstsr-core/src/tensor/`.
  - Generate directory for category (manuplication, indexing, etc). Currently only `manuplication` followed this convention, and other functions will be refactorized later. Then add function-specific code file (`reshape.rs`, `transpose.rs`, etc).
  - Implement function `into_<func>_f`, `to_<func>_f`/`<func>_f`, `into_<func>`, `to_<func>`/`<func>` if appropriate. There may have lots of naming exceptions (for example of basic indexing: `slice`, `into_slice`, `i`, etc.; `reshape` equilvant to `to_shape`), user may provide suggestions, and you can also ask user for suggestions.
  - **If function to be implemented does not exist, stop and ask user after you proposed a function name and its signature.**
  - Minimal testing (you can write temporary unittest in code, but will be moved to integration tests `rstsr-core/tests` and unittests discarded if everything works well).
  - Also implement associated function to `TensorAny` (usually, may have exceptions that should be implemented in owned `Tensor`, viewed `TensorView` or base class `TensorBase`).

3. Operation Function Implementation
  - (to be added later)

4. Integration Tests update
  - Add integration tests at `rstsr-core/tests/` for the new function.
  - You should refer to existing tests as example.
  - You need to prepare some tests for API documentation demonstration.
  - You can check NumPy's test cases for the similar function if exist.
    - You need to add the comment line like the following:
      ```
      // NumPy v2.4.2, lib/tests/test_shape_base.py, TestExpandDims::test_functionality (line 310)
      ```
    - If multiple cases in a single class, you may add comment line like
      ```
      // NumPy v2.4.2, _core/tests/test_regression.py, TestRegression::test_reshape*
      ...
      // CASE test_reshape_order (line 635)
      ...
      // CASE test_reshape_zero_strides (line 643)
      ```
    - NumPy repository should be git (version) tagged. You should use git tools to verify the exact version of the NumPy repository. Similar for other repositories (SciPy).
    - Code of python should also be added as comment for reference.
  - **Note it is possible that RSTSR not correctly, or behaves differently, to NumPy. If that happens, report to the maintainer. You must not cheat by changing the test target to make the test pass.** (dev note: this happened once by qwen3.5-plus).
  - If `rand` occurs but only testing arithmetic/shape-manuplication that does not involve real random number testing, you can use arange/ones/linspace to generate test data for simplicity. RSTSR currently does not support random number generation.
  - Run tests (only the newly added tests for the new function) to check if everything works well. Follow rules in `.claude/rules/code-related.md`.

5. Integration Tests for Documentation
  - Those tests are used for API documentation, but located and tested as integration tests. Usually named `test_<func>.rs`.
  - You can check NumPy's API docstrings for the similar function if exist.
  - The integration tests initialize device with `RowMajor` order, and use the device specified in `TESTCFG`.
  - You are suggested to `println!` the result for demonstration, and then use `assert!`/`assert_eq!` to check the result.
    - For tensor output, you are preferred to use display (`{:}`) instead of debug (`{:?}`) for better readability.
    - **You should at least run cargo test for a first time, use --nocapture to get the real output, and then insert the output into the comment.**
    - You are preferred to call functions as associated functions `tensor.func()`, instead of usual functions `rt::func(tensor)`, unless calling convention of usual function and associated function is different.
    - For tensor equivalence, you are suggested to use `assert!(rt::allclose(&result, &expected, None))`.
    - Example of this:
      1. You should first write the test code like this:
      ```rust
      let a = rt::tensor_from_nested!([[1, 2], [3, 4]], &device);
      let result = a.transpose(None);
      println!("{result}");
      let target = rt::tensor_from_nested!([[1, 3], [2, 4]], &device);
      assert!(rt::allclose(&result, &target, None));
      ```
      2. Then test code with `--nocapture` to get the real output, and verify the output is correct.
      3. Finally, insert the output into the comment:
      ```rust
      let a = rt::tensor_from_nested!([[1, 2], [3, 4]], &device);
      let result = a.transpose(None);
      println!("{result}");
      // [[ 1 3]
      //  [ 2 4]]
      let target = rt::tensor_from_nested!([[1, 3], [2, 4]], &device);
      assert!(rt::allclose(&result, &target, None));
      ```
      And this is directly translated from NumPy docstring (you can omit or retain the input print at your preference):
      ```python
      >>> import numpy as np
      >>> a = np.array([[1, 2], [3, 4]])
      >>> a
      array([[1, 2],
             [3, 4]])
      >>> np.transpose(a)
      array([[1, 3],
             [2, 4]])
      ```

6. API Documentation update
  - For this part, currently manuplication functions is documented, and can be referred as examples.
  - Only fully document `to_<func>` or `<func>`. Other functions (also all associated functions, functions that is fillable with suffix `_f`) just have the same title description, but use `See Also` to refer to the fully documented one. Function `reshape` is special that it follows different convention.
  - You should include the integration tests that delibrately written for API documentation in `Examples` section.
    - **You should first write tests in integration test file `test_<func>.rs`, test them, write back the output there-in, as point 5 specified. Then copy the code into the documentation following the convention listed below. Don't write from scratch and give the doctests without first writing to integration test file `test_<func>.rs`.**
    - You should always include the following first three lines at start.
    - You can discard the assertion lines, but keep the `println!` lines for demonstration.
    ```rust
    /// # use rstsr::prelude::*;
    /// # let mut device = DeviceCpu::default();
    /// # device.set_default_order(RowMajor);
    /// let a = rt::tensor_from_nested!([[1, 2], [3, 4]], &device);
    /// let result = a.transpose(None);
    /// println!("{result}");
    /// // [[ 1 3]
    /// //  [ 2 4]]
    ```
  - If some functions behaves very like NumPy or array-api functions, add `Notes of API accordance` section. For example of reshape:
    ```md
    # Notes of API accordance
    
    - Array-API: `reshape(x, /, shape, *, copy=None)` ([`reshape`](https://data-apis.org/array-api/latest/API_specification/generated/array_api.reshape.html))
    - NumPy: `reshape(a, /, shape, order='C', *, copy=False)` ([`numpy.reshape`](https://numpy.org/doc/stable/reference/generated/numpy.reshape.html)):
    - RSTSR: `rt::reshape_with_args(tensor, shape, (order, copy))`
    - RSTSR: `rt::reshape(tensor, shape)`
    
    <Include notes if you feel necessary.>
    ```
  - Run `cargo doc --no-deps` to check if doc build is successful.
  - Upgrade the description in `rstsr/src/docs/api_specification.md` if necessary.
  - If function that is in `rstsr/src/docs/array_api_standard.md`, but not tagged as implemented, modify the tag therin. If function is not in `array_api_standard.md`, ignore this step.

6. User Documentation update
   - (to be added later)

