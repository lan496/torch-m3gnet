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
    \quad (m \geq 1),
```
where {math}`1 \leq m < n_{\max}`

## Atom featurizer

Embedding layer

## Spherical Bessel function

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

The spherical Bessel functions {math}`\{ j_{l}(z_{ln} \frac{r}{r_{c}}) \}_{n=1,\dots}` form orthogonal basis on {math}`r \in [0, r_{c}]`,
```{math}
  \int_{0}^{r_{c}}
    j_{l}(z_{ln} \frac{r}{r_{c}}) j_{l}(z_{ln'} \frac{r}{r_{c}}) r^{2} dr
  = \frac{r_{c}^{3}}{2} \left( j_{l+1}(z_{ln}) \right)^{2} \delta_{nn'}.
```
Here {math}`z_{ln}` is the {math}`n`th root of {math}`j_{l}`.
We use 
uses normalized spherical Bessel functions [^vasp]
```{math}
  \chi_{ln}(r)
    &= \sqrt{ \frac{2}{r_{c}^{3}} } \frac{ j_{l}(z_{ln}\frac{r}{r_{c}}) }{ |j_{l+1}(z_{ln})| } \\
  \int_{0}^{r_{c}} \chi_{ln}(r) \chi_{ln'}(r) r^{2} dr
    &= \delta_{nn'}.
```

[^vasp]: The normalization constant is slightly different from that of VASP 6.0 {cite}`PhysRevB.100.014105`.

## Spherical harmonics with {math}`m=0`

```{math}
  Y_{l}^{0}(\theta)
    &= \sqrt{ \frac{2l+1}{4\pi} } P_{l}(\cos \theta) \\
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

## Three-body to bond

```{math}
  \tilde{\mathbf{v}_{i}}
    &= \mathcal{L}_{\sigma}(\mathbf{v}_{i}) \\
  \tilde{e}_{ij}
    &= \sum_{k \in \mathcal{N}_{i} \backslash \{ j \} }
          \left(
              \chi_{ln}(r_{ik}) Y_{l}^{0}(\theta_{jik}) f_{3c}(r_{ij}) f_{3c}(r_{ik})
          \right)_{l=0, \dots, l_{\max}-1, n=0, \dots, n_{\max}-1}
          \odot
          \tilde{\mathbf{v}}_{k} \\
  \mathbf{e}_{ij}
      &\leftarrow \mathbf{e}_{ij}
          + \mathcal{L}_{\mathrm{swish}}(\tilde{\mathbf{e}}_{ij}) \odot \mathcal{L}_{\mathrm{sigmoid}}(\tilde{\mathbf{e}}_{ij}) \\
```
{math}`\mathcal{L}_{\sigma}` is a one-layer perceptron with activation function {math}`\sigma`.

Cutoff function
```{math}
    f_{3c}(r)
    = 1
        - 6 \left( \frac{r}{r_{3c}} \right)^{5}
        + 15 \left( \frac{r}{r_{3c}} \right)^{4}
        - 10 \left( \frac{r}{r_{3c}} \right)^{3}
```


## Graph Convolution

```{math}
\mathbf{e}_{ij}'
  &= \mathbf{e}_{ij} + \mathbf{\phi}_{e}(\mathbf{v}_{i} \oplus \mathbf{v}_{j} \oplus \mathbf{e}_{ij}) \odot \mathbf{W}_{e}^{0} \mathbf{e}_{ij}^{0} \\
\mathbf{v}_{i}'
  &= \mathbf{v}_{i} + \sum_{j \in \mathcal{N}_{i}} \mathbf{\phi}_{e}'(\mathbf{v}_{i} \oplus \mathbf{v}_{j} \oplus \mathbf{e}_{ij}') \odot \mathbf{W}_{e}'^{0} \mathbf{e}_{ij}^{0}
```

## Readout

```{math}
  \tilde{E}_{\mathrm{M3GNet}} = \sum_{i} \phi_{3}(\mathbf{v}_{i})
```

## Elemental reference energy

We first fit total energies in a training dataset from atom types {math}`\{ t_{i} \}`.
Then, we normalize the residual on the training dataset as
```{math}
  E(\mathbf{A}, \{ \mathbf{r}_{i} \}_{i=1}^{N}, \{ t_{i} \}_{i=1}^{N}) - \sum_{i=1}^{N} E_{t_{i}}
  = N \mu
    + \sigma \tilde{E}_{\mathrm{M3GNet}}(\mathbf{A}, \{ \mathbf{r}_{i} \}_{i=1}^{N}, \{ t_{i} \}_{i=1}^{N}),
```
where {math}`\mu = O(1), \sigma = O(1)` and {math}`\tilde{E}_{\mathrm{M3GNet}}(\mathbf{A}, \{ \mathbf{r}_{i} \}_{i=1}^{N}, \{ t_{i} \}_{i=1}^{N}) = O(N)`.

## Loss function

```{math}
l_{E} &=
\frac{1}{n_{\mathrm{train}}}
\sum_{n=1}^{n_{\mathrm{train}}}
  \left|
    \frac{1}{N^{(n)}} \left(
      E(\mathbf{A}^{(n)}, \{ \mathbf{r}^{(n)}_{i} \}, \{ t^{(n)}_{i} \}) - E^{(n)}
    \right)
  \right|^{2} \\
l_{F} &=
\frac{1}{n_{\mathrm{train}}}
\sum_{n=1}^{n_{\mathrm{train}}}
  \frac{1}{3N^{(n)}}
  \sum_{j=1}^{N^{(n)}} \sum_{\alpha=1}^{3}
    \left|
      F_{j\alpha}(\mathbf{A}^{(n)}, \{ \mathbf{r}^{(n)}_{i} \}, \{ t^{(n)}_{i} \}) - F_{j \alpha}^{(n)}
    \right|^{2} \\
  l_{S} &=
  \frac{1}{6 n_{\mathrm{train}}}
  \sum_{n=1}^{n_{\mathrm{train}}} \sum_{p=1}^{6}
    \left|
      \sigma_{p}(\mathbf{A}^{(n)}, \{ \mathbf{r}^{(n)}_{i} \}, \{ t^{(n)}_{i} \}) - \sigma_{p}^{(n)}
    \right|^{2} \\
l &= l_{E} + w_{F} l_{F} + w_{S} l_{S} \\
```

## References
```{bibliography}
:filter: docname in docnames
```
