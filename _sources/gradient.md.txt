# Forces and virial stress via autograd

We assume a potential energy model satisfies {math}`\mathrm{SE}(3)` and permutation symmtries.
Also we model a total energy as summation of atomic energies.

## Forces

```{math}
\newcommand{\pdev}[2]{\frac{\partial #1}{\partial #2}}

E(\{ \mathbf{r}_{i} \}_{i})
    &= \sum_{i} E_{i}(\{ \mathbf{r}_{j} \}_{j}) \\
\mathbf{F}_{i}
    &= \pdev{ E(\{ \mathbf{r}_{j} \}_{j}) }{ \mathbf{r}_{i} }
```

We can obtain the following gradients (here we call them *pair forces*) via `autograd`:
```{math}
\newcommand{\pdev}[2]{\frac{\partial #1}{\partial #2}}

\mathbf{F}_{ji} = -\pdev{ E_{i}(\{ \mathbf{r}_{k} \}_{k}) }{ \mathbf{r}_{j} }.
```

With the translation symmetry of {math}`E_{i}(\{ \mathbf{r}_{k} \}_{k})`, we can write it as a function of displacements {math}`\mathbf{r}_{ij} = \mathbf{r}_{j} - \mathbf{r}_{i}`,
```{math}
E_{i}(\{ \mathbf{r}_{k} \}_{k})
    = \Psi( \{ \mathbf{r}_{ij} \}_{j \in \mathcal{N}_{i} } ).
```
Then, we can prove a relation similar to the Newton's thrid law,
```{math}
\newcommand{\pdev}[2]{\frac{\partial #1}{\partial #2}}

\mathbf{F}_{ii}
    &= -\pdev{ E_{i}(\{ \mathbf{r}_{j} \}_{j}) }{ \mathbf{r}_{i} } \\
    &= \sum_{j \in \mathcal{N}_{i}} \pdev{\Psi}{\mathbf{r}_{ij}}(\{ \mathbf{r}_{ik} \}_{k \in \mathcal{N}_{i}}) \\
    &= \sum_{j \in \mathcal{N}_{i}} \pdev{ \Psi(\{ \mathbf{r}_{ik} \}_{k \in \mathcal{N}_{i}}) }{\mathbf{r}_{j}} \\
    &= -\sum_{j \in \mathcal{N}_{i}} \mathbf{F}_{ji}. \\
```

The forces can be written by the pair forces as
```{math}
\mathbf{F}_{i}
    &= \mathbf{F}_{ii} + \sum_{ j \in \mathcal{N}_{i} } \mathbf{F}_{ij} \\
    &= \sum_{ j \in \mathcal{N}_{i} } \left( \mathbf{F}_{ij} - \mathbf{F}_{ji} \right).
```

## Virial stress

Ref. {cite}`doi:10.1063/1.3245303,Buko2005`

Consider fractional coordinates {math}`\mathbf{r}_{i} = \mathbf{A} \mathbf{s}_{i}^{0}` with basis vectors {math}`\mathbf{A} = \left( \mathbf{a}_{1}, \mathbf{a}_{2}, \mathbf{a}_{3} \right)`

```{math}
A_{\beta p}( \epsilon )
    &:= \sum_{\alpha} A_{\alpha p} (\delta_{\alpha\beta} + \epsilon_{\alpha\beta}) \\
\tilde{E}( \epsilon )
    &:= E( \{ \mathbf{r}_{i} \leftarrow \mathbf{A}(\epsilon)\mathbf{s}_{i}^{0} \} )
```

We define a virial stress tensor as
```{math}
\newcommand{\pdev}[2]{\frac{\partial #1}{\partial #2}}

\sigma_{\alpha\beta}
    &:= \lim_{ \epsilon \to \mathbf{O} } -\frac{1}{V} \pdev{ \tilde{E}(\epsilon) }{ \epsilon_{\alpha\beta} } \\
    &= \lim_{ \epsilon \to \mathbf{O} } - \frac{1}{V} \sum_{i\gamma} \pdev{E}{r_{i\gamma}}( \{ \mathbf{A}(\epsilon)\mathbf{s}_{j}^{0} \} ) \pdev{}{\epsilon_{\alpha\beta}} \sum_{p} A_{\gamma p}(\epsilon) s_{ip}^{0} \\
    &= \lim_{ \epsilon \to \mathbf{O} } - \frac{1}{V} \sum_{i p} \pdev{E}{r_{i\beta}}( \{ \mathbf{A}(\epsilon)\mathbf{s}_{j}^{0} \} ) A_{\alpha p} s_{ip}^{0} \\
    &= \frac{1}{V} \sum_{i} r_{i\alpha} F_{i\beta} .
```

Rewrite the virial stress tensor with the pair forces:
```{math}
\mathbf{\sigma}V
    &= \sum_{i} \mathbf{r}_{i} \otimes \mathbf{F}_{i} \\
    &= \sum_{i} \sum_{j \in \mathcal{N}_{i}}
        \mathbf{r}_{i} \otimes \left( \mathbf{F}_{ij} - \mathbf{F}_{ji} \right) \\
    &= \frac{1}{2} \sum_{i} \sum_{j \in \mathcal{N}_{i}}
        \left(
            \mathbf{r}_{i} \otimes \left( \mathbf{F}_{ij} - \mathbf{F}_{ji} \right)
            + \mathbf{r}_{j} \otimes \left( \mathbf{F}_{ji} - \mathbf{F}_{ij} \right) \\
        \right) \\
    &= \frac{1}{2} \sum_{i} \sum_{j \in \mathcal{N}_{i}}
            \mathbf{r}_{ij} \otimes \left( \mathbf{F}_{ji} - \mathbf{F}_{ij} \right) \\
```

## References
```{bibliography}
:filter: docname in docnames
```
