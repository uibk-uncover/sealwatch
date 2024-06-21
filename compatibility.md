# Unit tests

To ensure that our features are equivalent to the features produced by the Matlab, we slightly altered the original Matlab implementations.
In particular, we replaced the rounding operator and methods that draw random numbers with a custom implementation that can be reproduced in Python.
For more details, see [this blog post](https://www.benediktlorch.com/blog/2023/matlab-to-python/).

## Other differences

- The Matlab implementation of the PHARM features contains a bug in the symmetrization. We provide both the original implementation (*pharm_original*) and a variant with correct symmetrization (*pharm_revisited*). The revisited implementation also does not crop the image borders to simplify the indexing. Therefore, it gives slightly different results compared to the original implementation when symmetrization is disabled.
- The Matlab implementation of the FLD ensemble contains two peculiarities:
  * The function `add_gridpoints` in lines 588-605 looks like it implements insertion into a sorted list, but it inserts the new element one position too early.
  * While searching for the optimal `d_sub`, the condition in line 355 uses `settings.d_sub`. However, `settings.d_sub` is the `d_sub` that was previously evaluated. We believe that `settings.d_sub` should be replaced by `SEARCH.x(minE_id)` in line 355 and in line 358.