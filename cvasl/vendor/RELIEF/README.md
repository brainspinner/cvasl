# RELIEF

**REmoval of Latent Inter-scanner Effects through Factorization**

R package to apply **RELIEF harmonization** to multi-site or multi-scanner neuroimaging studies. This method has been previously called **UNIFAC harmonization**.

### References

> Zhang, R., Oliver, L. D., Voineskos, A. N., Park, J. Y. (2023). RELIEF: a structured multivariate approach for removal of latent inter-scanner effects. Imaging Neuroscience. [link](https://doi.org/10.1162/imag_a_00011)

### Update logs

> April 27, 2023: RELIEF method implemented in the package.

## Contents

1. [Background](#id-background)
2. [Installation](#id-installation)
3. [Usage](#id-relief)

---

<div id='id-background'/>

### Background
RELIEF supports harmonizing neuroimaging data collected from multiple sites and/or scanners. It is equivalent to the *batch effect correction* in genomic studies. Compared to existing approaches such as ComBat (Johnson *et al*, 2007), RELIEF supports the removal of latent scanner-specific patterns, which results in better data quality after harmonization.


<div id='id-installation'/>

---

### Installation
To install the latest development builds directly from GitHub, please run the followings:

```R
if (!require("devtools"))
  install.packages("devtools")
devtools::install_github("junjypark/RELIEF")
```

Note: If you see an error message, please update your `devtools` package first.

```R
update.packages("devtools")
```

After installation, the package can be loaded directly in R.
```R
library(RELIEF)
``` 

---

<div id='id-relief'/>

### Usage

`relief()` is the main function that takes the same input names as [neuroCombat](https://github.com/Jfortin1/neuroCombat_Rpackage). Specifically, the following need to be provided.

* `dat`: a (p x n) data matrix, where p is the number of features and n is the number of subjects (required)
* `batch`: Batch variable for the scanner id (required)
* `mod`: a (n x q) matrix containing biological covariates (optional). This can be obtained by using `model.matrix()` function in R. However, when you use `mod` in RELIEF, ensure your covariate of interest (for hypothesis testing) is excluded. In practice, RELIEF preserves covariate effects well even though it is not specified as an input.

Now, the `relief` can be applied to obtain the harmonized data.

```R
relief.harmonized = relief(dat=dat, batch=batch, mod=mod)
``` 



---
