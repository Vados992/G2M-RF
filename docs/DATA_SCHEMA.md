# Analysis-ready data schema

The executable core begins after upstream sequencing/scanning QC. Every array must describe the same ordered participants.

| Field | Shape | Meaning |
|---|---:|---|
| `sample_ids` | `N` | unique participant identifiers |
| `family_ids` | `N` | connected kinship/family component IDs |
| `G` | `N x M` | diploid dosage, 0/1/2, NaN allowed before TRAIN-only imputation |
| `C` | `N x Q` | allowed technical/biological covariates; no intercept required |
| `radii` | `N x L` | normalized radial geometry |
| `angles` | `L` | locked landmark angles in radians |
| `landmark_classes` | `L` | preregistered landmark classes |

## Leakage firewall

The code fits the genotype standardizer on TRAIN and only then transforms INTERNAL/EXTERNAL. Monomorphic TRAIN variants are removed. Missing values outside TRAIN are imputed with TRAIN genotype means.

No TEST or EXTERNAL phenotype may influence:

- allele frequencies;
- scaling;
- variant filtering;
- covariate definition;
- PCA basis;
- morphology-family selection;
- regularization settings;
- gate thresholds.

## Whole-genome production note

A dense `N x M` NumPy matrix is intended for executable validation and moderate analysis-ready studies. Raw WGS at hundreds of thousands of samples requires chunked/distributed storage. The statistical interface can consume standardized blocks or a future backend without changing the scientific definitions.
