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
