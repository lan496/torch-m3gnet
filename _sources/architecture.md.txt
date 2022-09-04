# Network architecture

## Three-body computation

`triplet_edge_index`
```{math}
    \mathcal{M}_{i} = \left\{ (j, k) \mid j, k \in \mathcal{N}_{i}, j \neq k \right\}
```

```{math}
    \cos \theta_{jik}
        = \frac{ \mathbf{r}_{ij} \cdot \mathbf{r}_{ik} }{ |\mathbf{r}_{ij} | | \mathbf{r}_{ik}| }
        \quad ((j, k) \in \mathcal{M}_{i})
```

## Bond featurizer

{cite}`doi:10.1063/1.5086167`

```{math}
e_{m} &= \frac{ m^{2}(m+2)^{2} }{ 4(m+1)^{2} + 1 } \\
d_{0} &= 1 \\
d_{m} &= 1 - \frac{ e_{m} }{ d_{m-1} } \quad (m \geq 1) \\
f_{m}(r)
    &= (-)^{m} \frac{ \sqrt{2}\pi }{ r_{c}^{3/2} }
        \frac{ (m+1)(m+2) }{ \sqrt{ (m+1)^{2} + (m+2)^{2} } }
        \left(
            \mathrm{sinc} \left( (m+1)\pi \frac{r}{r_{c}} \right)
            + \mathrm{sinc} \left( (m+2)\pi \frac{r}{r_{c}} \right)
        \right) \\
h_{0}(r) &= f_{0}(r) \\
h_{m}(r)
    &= \frac{1}{\sqrt{d_{m}}} \left( f_{m}(r) + \sqrt{\frac{ e_{m} }{ d_{m-1} }} h_{m-1}(r) \right)
    \quad (m \geq 1) \\
```

{math}`m = n_{\max} l_{\max}`

## References
```{bibliography}
:filter: docname in docnames
```
