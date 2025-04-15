# Layout API document

Layout defines how elements will be accessed alongwith vector of data.

## Definition

We use notation $n$ as number of dimension, and $0 \leqslant k < n$ as a specific dimension.

Layout $L(\bm{d}, \bm{t}, s)$ contains three components:
- shape $\bm{d} = (d_0, d_1, \cdots, d_{n - 1})$, where $d_k \geqslant 0$, getter function [`Layout::shape`].
- stride $\bm{t} = (t_0, t_1, \cdots, t_{n - 1})$, where $t_k \neq 0$, getter function [`Layout::stride`].
- offset $s$, where $s \geqslant 0$, getter function [`Layout::offset`].

For index computation, indices will also be involved:
- indices $\bm{i} = (i_0, i_1, \cdots, i_{n - 1})$, where $0 \leqslant i_k < d_k$

For a tensor, we generally use an array ($m_z$, $z \geqslant 0$) to store data in memory. With layout $L(\bm{d}, \bm{t}, s)$, the memory pointer for element at index $\bm{i}$ should be
$$
z(\bm{i}) = s + \bm{i} \cdot \bm{t} = s + \sum_{k = 0}^{n - 1} i_k t_k
$$

Note that in actual program, it is allowed to have negative indices $\tilde i_k \in [-d_k, d_k)$:
$$
i_k = \begin{cases}
\tilde i_k & i_k \in [0, d_k) \\\\
\tilde i_k + d_k & i_k \in [-d_k, 0)
\end{cases}
$$
