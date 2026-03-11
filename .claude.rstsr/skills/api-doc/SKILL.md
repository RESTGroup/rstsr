---
description: Core function API documentation and testing guide.
---

Refer to skill `cargo-inst` for compilation instructions.

## Must Know

- **Note it is possible that RSTSR not correctly, or behaves differently, to NumPy. If that happens, report to the maintainer. You must not cheat by changing the test target to make the test pass.** (dev note: this happened once by qwen3.5-plus).

## General Workflow

1. Generate API test code, and test them.
  1.1. Find function in NumPy or SciPy.
  1.2. If found, translate test code into rust by guidelines defined in this file.
  1.3. If not found, write test code by yourself (but not that overwhelmed).
2. Generate doc test code, and test them.
  2.1. Find function in NumPy or SciPy or array-api.
  2.2. If found, get inspiration from their docstring, and write doc test code by guidelines defined in this file.
  2.3. If not found, write doc test code by yourself.
3. Create API documentation to *core function* (instead of *variant functions*), step by step.
  The following guidelines are for *core functions* only.
  3.1. Give a title. If NumPy/SciPy/array-api has the same function, directly use the same title.
  3.2. Write explanation and description if necessary.
  3.3. If the behavior of row/col-major is different, add warning message.
  3.4. Add `Parameters` and `Returns` section.
  3.5. Add `Examples` section. This **must follow doc test code**.
  3.6. Add `Notes of API accordance` section.
  3.7. Add `See Also` section, first refer to relavent functions if necessary, then variants of functions, then variants of associated functions.
  3.8 Add `Panics` section if necessary, only in.

Examples of API document (in docstring) should be tested, but they should first be tested in integration test (in folder `tests`).

## Detailed Guidelines for API Documentation

0. General workflow
  - You may find directories `numpy`, `scipy`, `array-api` at `<workspace>/../other-repos/<repo-name>`. NumPy and SciPy are are git repos with tags.
1. Generate API test code, and test them.
  1.0. Style of test code.
    - Example refer to `rstsr-core/tests/core_func/manuplication/test_transpose.rs`.
    - You should first include the following code:
      ```rust
      use crate::test_utils::*;
      use rstsr::prelude::*;

      use super::CATEGORY;
      use crate::TESTCFG;

      #[cfg(test)]
      mod <test_source>_<func> {
          use super::*;
          static FUNC: &str = "<test_source>_<func>";
      ```
      Then, for each test function, you should follow the following style (if it is NumPy's test's translation, please also include test function name, NumPy version, test file, test case and its class, line of NumPy's file):
      ```rust
      #[test]
      fn test_multiarray() {
          // NumPy v2.4.2, _core/tests/test_multiarray.py, TestMethods::test_transpose (line 2221)
          crate::specify_test!("test_multiarray");

          let mut device = TESTCFG.device.clone();
          device.set_default_order(RowMajor);
          <......>
      }
      ```
    - For array-allclose-equilvance, you can use function `assert_equal` for this:
      ```rust
      assert_equal(&result, &target, None);  // None means default rtol and atol tolerance.
      ```
      For other cases, such as comparing two `Vec<usize>`, you may use usual `assert!` or `assert_eq!`.
    - Tensor creation:
      - For nested tensors, such as `np.asarray([[1, 2], [3, 4]])`, you should use `rt::tensor_from_nested!([[1, 2], [3, 4]], &device)` to create tensor from nested array.
      - You can use `rt::arange((n, &device)).into_shape(shape)`, `rt::zeros((shape, &device))`, `rt::ones((shape, &device))`. Please explicitly specify the type of `let t: Tensor<some_type, _> = rt::zero((shape, &device))` when calling zero or ones functions. Replace `np.empty` as `rt::zeros`. Try to replace `np.random.randn` as `np.arange`.
  1.1. Find function in NumPy or SciPy.
    - For example of NumPy, the tests should be at `numpy/numpy/_core/tests` or `numpy/numpy/lib/tests`.
  1.2. If found, translate test code into rust by guidelines defined in this file.
    - Please note some tests may not be directly applicable to RSTSR, due to some functions are not found in RSTSR; if that happens, and does not found equilvants (for example `flatten()` is equilvant to `reshape(-1)`), just ignore those tests. For core functions, refer to `rstsr-core/src/prelude.dev` for most implemented functions (as usual function). There are also some functions implemented as associated or trait functions.
    - Not only translate, but directly copy codes from python and comment them. For example, the following commented lines are directly from python file:
      ```rust
      // arr = [[1, 2], [3, 4], [5, 6]]
      // tgt = [[1, 3, 5], [2, 4, 6]]
      // assert_equal(np.transpose(arr, (1, 0)), tgt)
      let arr = rt::tensor_from_nested!([[1, 2], [3, 4], [5, 6]], &device);
      let tgt = rt::tensor_from_nested!([[1, 3, 5], [2, 4, 6]], &device);
      assert_equal(rt::transpose(&arr, [1, 0]), &tgt, None);

      // assert_equal(np.transpose(arr, (-1, -2)), tgt)
      assert_equal(rt::transpose(&arr, [-1, -2]), &tgt, None);
      ```
    - Space a line between different test cases.
    - If encountered list of test cases, `test_broadcast_shapes_succeeds` in `test_broadcast_shapes.rs` can be example.
    - If expect error, first try `_f` suffix functions (for example `assert!(tensor.transpose_f(-2, -5).is_err())`). If that also panics, example can be `std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| a.reshape_f(new_shape)))` and then check if it is error.
    - If there are some intense tests in NumPy's class, and each case is very small, you can use the following comment code (which is also in `test_reshape.rs` case `numpy_reshape::regression`):
      ```rust
      // NumPy v2.4.2, _core/tests/test_regression.py, TestRegression::test_reshape*
      <......>
      // CASE test_reshape_order (line 635)

      // a = np.arange(6).reshape(2, 3, order='F')
      // assert_equal(a, [[0, 2, 4], [1, 3, 5]])
      let a = rt::arange((6, &device)).into_shape_with_args([2, 3], ColMajor);
      let tgt = rt::tensor_from_nested!([[0, 2, 4], [1, 3, 5]], &device);
      assert_equal(&a, &tgt, None);
      ```
  1.3. If not found, write test code by yourself (but not that overwhelmed).
    - Please use the name `mod custom_<func>` for the test module, and `FUNC` should be `custom_<func>`.
    - Try to write at least 2 cases, but no more than 10 cases. Test the some 1-3 edge cases that you feel very important, but not that overwhelmed.
2. Generate doc test code, and test them.
  2.0. Style of doc test code.
    - Code start from
      ```rust
      #[cfg(test)]
      mod doc_<func> {
          use super::*;
          static FUNC: &str = "doc_<func>";

          #[test]
          fn test_<case>() {
              crate::specify_test!("test_<case>");

              let mut device = TESTCFG.device.clone();
              device.set_default_order(RowMajor);
      ```
    - First create test case, and make the test passes. For example,
      ```rust
      // 3-D array: swapping axes 0 and 2
      let x = rt::tensor_from_nested!([[[0, 1], [2, 3]], [[4, 5], [6, 7]]], &device);
      let result = x.swapaxes(0, 2);
      println!("{result}");
      let target = rt::tensor_from_nested!([[[0, 4], [2, 6]], [[1, 5], [3, 7]]], &device);
      assert!(rt::allclose(&result, &target, None));
      ```
      Please note that tensor can be printed as display (`{result}`) instead of debug (`{:?}`), but layout/shape/strides and vectors should be printed as debug (`println!("{:?}", tensor.layout())`).
    - You should then run the test, capture the output, and add them as comments. **Please not add the output before running the test, because the output may be different from your expectation, and you should first make sure the test is correct.** You should use cargo test with `--nocapture` to get the stdout. For the above case, the finished code should be like:
      ```rust
      // 3-D array: swapping axes 0 and 2
      let x = rt::tensor_from_nested!([[[0, 1], [2, 3]], [[4, 5], [6, 7]]], &device);
      let result = x.swapaxes(0, 2);
      println!("{result}");
      // [[[ 0 4]
      //   [ 2 6]]
      //
      //  [[ 1 5]
      //   [ 3 7]]]
      let target = rt::tensor_from_nested!([[[0, 4], [2, 6]], [[1, 5], [3, 7]]], &device);
      assert!(rt::allclose(&result, &target, None));
      ```
3. Create API documentation to *core function* (instead of *variant functions*), step by step.
  3.0. Distinguish *core function* and *variant functions*.
    - For example, `transpose` (panic version that returns view) is a core function, while `transpose_f` (fallible version that returns result of view) and `into_transpose` (panic version that takes ownership and returns the same ownership) and `into_transpose_f` (fallible ownership-token) and `TensorAny::t()` (alias of `transpose` with argument `None`) and `permute_dims` (alias of `transpose`) are all variant functions.
    - All associated functions (includes `TensorAny::transpose()`) are all variant functions.
    - Except `reshape`, only fully document core functions, and use minimized documentation for variant functions.
    - The minimized documentation style is, for example,
      ```rust
      /// Permutes the axes (dimensions) of an array.
      ///
      /// See also [`transpose`].
      ```
      or exactly,
      ```rust
      /// <title of core_func>
      ///
      /// See also [`<core_func>`].
      ```
      The variant functions should have the same title as the core function.

  The following guidelines are for *core functions* only.

  3.1. Give a title. If NumPy/SciPy/array-api has the same function, directly use the same title.
    - First find NumPy/SciPy's title. If not, find array-api. Then if not, write on your own.
  3.2. Write explanation and description if necessary.
  3.3. If the behavior of row/col-major is different, add warning message.
    - Actually, not many functions have different behavior of row/col-major. However, some important functions like `reshape`, `asarray` and broadcasting have different behavior of row/col-major (NumPy vs Fortran/Julia/Matlab).
    - Add warning message like
      ```rust
      /// <div class="warning">
      ///
      /// **Row/Column Major Notice**
      ///
      /// This function behaves differently on default orders ([`RowMajor`] and [`ColMajor`]) of device.
      ///
      /// </div>
      ```
    - For these cases, you should be better add a special section/subsection for describing different behavior of row/col-major, and better add code examples for this (you will probably return to step 2).
  3.4. Add `Parameters` and `Returns` section.
    - The first parameter is usually `&TensorAny<R, T, B, D>`. The example can be
      ```rust
      /// - `tensor`: [`&TensorAny<R, T, B, D>`](TensorAny)
      ///   <empty line>
      ///   - The input tensor.
      ///   - Note on variant [`into_<func>`]: This takes ownership [`Tensor<R, T, B, D>`] of input tensor, and will not perform change to underlying data, only layout changes.
      ///   <empty line>
      ```
    - Some parameters may have overloads. If you feel this is not a trivial overloading, you can also add notes for this. For example of `reshape_with_args`, the note is
      ```rust
      /// - `args`: Into [`ReshapeArgs`]
      ///
      ///   - `order`: <......>
      ///   - `copy`: <......>
      ///     - True: <...>
      ///     - False: <...>
      ///     - None (default): <...>
      ///   
      ///   - Overloads:
      ///
      ///     - copy: [`bool`]
      ///     - copy: [`Option<bool>`] (None means default behavior)
      ///     - order: [`TensorOrder`]
      ///     - (order: [`TensorOrder`], copy: [`bool`])
      ///     - (order: [`TensorOrder`], copy: [`Option<bool>`])
      ```
  3.5. Add `Examples` section. This **must follow doc test code**.
    - Each example code should be summarized by a simple description before code block.
    - You should **directly copy** part of the code `mod test_<func>` in integration test file `test_<func>.rs` to example code block.
    - In most cases, you should start with import as
      ```rust
      /// # use rstsr::prelude::*;
      /// # let mut device = DeviceCpu::default();
      /// # device.set_default_order(RowMajor);
      ```
      No new-line after the last line.
      In a few cases, if you need to show behavior of col-major, then you should remove the `#` at the third line, and change the order to `ColMajor`.
    - You can ignore the assertion code (mainly for testing the correctness of example), and the definition of target variable related to that assertion. However, if the assertion is very important for understanding the function, you may also leave the assertion code.
    - A full example is
      ```rust
      /// # use rstsr::prelude::*;
      /// # let mut device = DeviceCpu::default();
      /// # device.set_default_order(RowMajor);
      // 3-D array: swapping axes 0 and 2
      let x = rt::tensor_from_nested!([[[0, 1], [2, 3]], [[4, 5], [6, 7]]], &device);
      let result = x.swapaxes(0, 2);
      println!("{result}");
      // [[[ 0 4]
      //   [ 2 6]]
      //
      //  [[ 1 5]
      //   [ 3 7]]]
      ```
  3.6. Add `Notes of API accordance` section.
    - You should refer to array-api first, then NumPy/SciPy, then RSTSR function itself. For example of `reshape_with_args`:
      ```rust
      /// - Array-API: `reshape(x, /, shape, *, copy=None)` ([`reshape`](https://data-apis.org/array-api/latest/API_specification/generated/array_api.reshape.html))
      /// - NumPy: `reshape(a, /, shape, order='C', *, copy=False)` ([`numpy.reshape`](https://numpy.org/doc/stable/reference/generated/numpy.reshape.html)):
      /// - RSTSR: `rt::reshape_with_args(tensor, shape, (order, copy))`
      ```
      The signature of python's function should directly copied from the soruce code in git repo at local directory.
    - You can check if there is any difference of behavior between NumPy/SciPy and RSTSR. If there is, you should also add notes for that.
  3.7. Add `See Also` section, first refer to relavent functions if necessary, then variants of functions, then variants of associated functions.
    - Example of this can be, for `transpose`:
      ```rust
      /// # See also
      /// <empty line>
      /// ## Related functions in RSTSR
      /// <empty line>
      /// - [`permute_dims`] - Alias for this function
      /// - [`reverse_axes`] - Reverse all axes order
      /// - [`swapaxes`] - Swap two specific axes
      /// <empty line>
      /// ## Variants of this function
      /// <empty line>
      /// - [`transpose`] / [`transpose_f`]: Returning a view.
      /// - [`into_transpose`] / [`into_transpose_f`]: Consuming version.
      /// <empty line>
      /// - Associated methods on `TensorAny`:
      ///   <empty line>
      ///   - [`TensorAny::transpose`] / [`TensorAny::transpose_f`]
      ///   - [`TensorAny::into_transpose`] / [`TensorAny::into_transpose_f`]
      ///   - [`TensorAny::t`] as shorthand for [`reverse_axes`]
      ```
  3.8 Add `Panics` section if necessary.
    - You can gracefully add 
      ```rust
      /// For a fallible version, use [`<func>_f`].
      ```

- A special case is that, if the function in integration test should fail but uses error propragation:
  ```rust
  <in test_func.rs>
  let result = rt::broadcast_arrays_f(vec![a, b]);  <this function should return error, but not panic>
  assert!(result.is_err());                         <this tests if the result is error>
  ```
  Then in API document, you can add `Panics` section like
  ```rust,should_panic
  let result = rt::broadcast_arrays(vec![a, b]);
  ```
  Please note that without `_f` suffix, the function should panic instead of returning error.
- Alias functions (declared by `pub use <func> as <alias>;`) do not need API docstring.
