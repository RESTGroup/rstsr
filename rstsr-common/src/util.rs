/* #region trait for split_at */

pub trait IterSplitAtAPI: Sized {
    // Function that split the iterator at the given index.
    // This is used for parallel iterator.
    fn split_at(self, index: usize) -> (Self, Self);
}

/* #endregion */
