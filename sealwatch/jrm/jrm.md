# Submodel description of the JPEG Rich Model (JRM)

**Absolute value features**

- Intra-block relationships
  * `Ah_T3`
  * `Ad_T3`
  * `Ad_T3_semidiag`
  * `Aoh_T3`
  * `Ax_T3`
  * `Aod_T3_diag_skip`
  * `Aod_T3_semidiag`
  * `Am_T3`

- Inter-block relationships
  * `Aih_T3`
  * `Aid_T3`
  * `Ais_T3`
  * `Aix_T3`

**Difference features**

Each submodel takes two differences `X1` and `X2`, extracts features from `X1` and other features from the transposed position in `X2`, and add up the two subfeatures.

- Intra-block differences (horizontal/vertical between `abs_X_Dh` and `abs_X_Dv`): `intra_block_hv_<cooc_direction>`
  * `intra_block_hv_Dh_T2`
  * `intra_block_hv_Dd_T2_diag`
  * `intra_block_hv_Dd_T2_semidiag`
  * `intra_block_hv_Doh_T2`
  * `intra_block_hv_Dx_T2`
  * `intra_block_hv_Dod_T2_diag_skip`
  * `intra_block_hv_Dod_T2_semidiag_skip`
  * `intra_block_hv_Dm_T2`
  * `intra_block_hv_Dih_T2`
  * `intra_block_hv_Did_T2`
  * `intra_block_hv_Dis_T2`
  * `intra_block_hv_Dix_T2`

- Intra-block differences (diagonal between `abs_X_Dd` and `abs_X_Dd`): `intra_block_diag_<cooc_direction>`
  * `intra_block_diag_Dh_T2`
  * `intra_block_diag_Dd_T2`
  * `intra_block_diag_Doh_T2`
  * `intra_block_diag_Dx_T2`
  * `intra_block_diag_Dod_T2_diag_skip`
  * `intra_block_diag_Dod_T2_semidiag_skip`
  * `intra_block_diag_Dm_T2`
  * `intra_block_diag_Dih_T2`
  * `intra_block_diag_Did_T2`
  * `intra_block_diag_Dis_T2`
  * `intra_block_diag_Dix_T2`

Inter-block differences (horizontal/vertical between `abs_X_Dih` and `abs_X_Div`): `inter_block_hv_<cooc_direction>`
  * `inter_block_hv_Dh_T2`
  * `inter_block_hv_Dd_T2_diag`
  * `inter_block_hv_Dd_T2_semidiag`
  * `inter_block_hv_Doh_T2`
  * `inter_block_hv_Dx_T2`
  * `inter_block_hv_Dod_T2_diag_skip`
  * `inter_block_hv_Dod_T2_semidiag_skip`
  * `inter_block_hv_Dm_T2`
  * `inter_block_hv_Dih_T2`
  * `inter_block_hv_Did_T2`
  * `inter_block_hv_Dis_T2`
  * `inter_block_hv_Dix_T2`

**Integral features**

- Absolute value features `Ax_T5`
  * `MXh` = horizontal and vertical direction
  * `MXd` = diagonal direction
  * `MXs` = semi-diagonal direction
  * `MXih` = inter-block horizontal and vertical direction
  * `MXid` = inter-block diagonal direction 

- Frequency dependencies `Df<difference_direction>_<cooc_direction>`

  Difference directions:
  * `H` = horizontal differences
  * `V` = vertical differences
  * `D` = diagonal differences
  * `IH` = inter-block horizontal differences
  * `IV` = inter-block vertical differences

  Co-occurrence directions:
  * `MXh` = horizontal (0, 1)
  * `MXv` = vertical (1, 0)
  * `MXd` = diagonal (1, 1)
  * `MXs` = semi-diagonal (1, -1)

- Spatial dependencies `Ds<difference_direction>_<cooc_direction>`

  Co-occurrence direction:
  * `MXih` = horizontal (0, 8)
  * `MXiv` = vertical (8, 0)
  * `MXid` = diagonal (8, 8)
  * `MXis` = semi-diagonal (8, -8)
