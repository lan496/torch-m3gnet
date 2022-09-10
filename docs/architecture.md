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
e_{m} &= \frac{ m^{2}(m+2)^{2} }{ 4(m+1)^{4} + 1 } \\
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

## Atom featurizer

Embedding layer

## Three-body to bond

```{math}
  \tilde{\mathbf{v}_{i}}
    &= \mathcal{L}_{\sigma}(\mathbf{v}_{i}) \\
  \tilde{e}_{ij}
    &= \sum_{k \in \mathcal{N}_{i} \backslash \{ j \} }
          \left(
              j_{l}(z_{ln}\frac{r_{ik}}{r_{c}}) Y_{l}^{0}(\theta_{jik}) f_{c}(r_{ij}) f_{c}(r_{ik})
          \right)_{l=0, \dots, l_{\max}-1, n=0, \dots, n_{\max}-1}
          \odot
          \tilde{\mathbf{v}}_{k} \\
  \mathbf{e}_{ij}
      &\leftarrow \mathbf{e}_{ij}
          + \mathcal{L}_{\mathrm{swish}}(\tilde{\mathbf{e}}_{ij}) \odot \mathcal{L}_{\mathrm{sigmoid}}(\tilde{\mathbf{e}}_{ij}) \\
```
{math}`j_{l}` is the {math}`l`th spherical Bessel function with roots at {math}`z_{ln} \, (n=0, \dots, n_{\max}-1)`.
{math}`Y_{l}^{0}` is the spherical harmonics with {math}`m=0`.
{math}`\mathcal{L}_{\sigma}` is a one-layer perceptron with activation function {math}`\sigma`.

Cutoff function
```{math}
    f_{c}(r)
    = 1
        - 6 \left( \frac{r}{r_{c}} \right)^{5}
        + 15 \left( \frac{r}{r_{c}} \right)^{4}
        - 10 \left( \frac{r}{r_{c}} \right)^{3}
```

### Spherical Bessel function

The spherical Bessel function of the first kind
```{math}
  % DLMF 10.49.3
  j_{0}(z) &= \frac{\sin z}{z} \\
  j_{1}(z) &= \frac{\sin z}{z^{2}} - \frac{\cos z}{z} \\
  j_{l+1}(z)
    &= \frac{2l+1}{z} j_{l}(z) - j_{l-1}(z)
    \quad (l \geq 1)
    \quad (\mbox{DLMF 10.51.1}) \\
  j_{n}(z) &\approx \frac{z^{n}}{(2n+1)!!} \quad (z \to 0) \\
```

The derivative of spherical Bessel function of the first kind
```{math}
  j_{0}'(z) &= -j_{1}(z) \\
  j_{l}'(z)
    &= j_{l-1}(z) - \frac{l+1}{z} j_{l}(z)
    \quad (l \geq 1)
    \quad (\mbox{DLMF 10.51.2}) \\
```

### Spherical harmonics with {math}`m=0`

```{math}
  Y_{l}^{0}(\theta)
    &= \sqrt{ \frac{2l+1}{4\pi} } P_{l}(\cos \theta) \\
  \frac{d }{d \theta} Y_{l}^{0}(\theta)
    &= \sqrt{ \frac{2l+1}{4\pi} } \frac{d}{d \theta}  P_{l}(\cos \theta) \\
```
This definition adopts Condon-Shortley phase.

Legendre polynomial (DLMF 14.10.3)
```{math}
  P_{0}(x) &= 1 \\
  P_{1}(x) &= x \\
  (n + 1) P_{n+1}(x) - (2n+1) x P_{n}(x) + n P_{n-1}(x) &= 0 \quad (n \geq 1)
```

Derivative of Legendre polynomial
```{math}
  \frac{d}{d x} P_{n}(x) &= n P_{n-1}(x) + x \frac{d}{dx} P_{n-1}(x) \quad (n \geq 1) \\
  \frac{d}{d \theta} P_{n}^{(m)}(\cos \theta)
    &= -\frac{1}{\sin \theta} \left( (n+m) P_{n-1}^{(m)}(\cos \theta) - n \cos \theta P_{n}^{(m)}(\cos \theta) \right) \\
```

## References
```{bibliography}
:filter: docname in docnames
```
