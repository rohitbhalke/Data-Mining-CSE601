# Part 1 - Dimensionality Reduction 

In this part, you are expected to conduct dimensionality reduction on three biomedical data files (pca_a.txt, pca_b.txt, pca_c.txt).

## Getting Started
Make sure you have Python3, Jupyter Notebook installed on your computer.
You need following packages in order set up the development enviornment.
1. ***numpy***
2. ***matplotlib***
3. ***random***
4. ***sklearn***

### Installing

```buildoutcfg
pip install --upgrade pip
pip install numpy
import matplotlib.pyplot as plt
import random
from sklearn.manifold import TSNE
```

### Input Files

This project requires ***pca_a.txt***, ***pca_b.txt***, ***pca_c.txt***,
***pca_demo.txt*** as input.

In each file, each row represents the record of a patient/sample; the last column is the
disease name, and the rest columns are features

### How to Run

To run the PCA, open the ***PCA.ipynb*** in Jupyter file and
run the whole notebook.


### Code Output 
The output would be a two-dimensional data points plotted on a scattered plot.
For each input three plots would be generated namely ***PCA***, ***SVD*** and ***TNSE***.


## Authors

* **Pooja Shingavi** -  *pshingav*
* **Madhushri Patil** - *mvpatil*
* **Rohit Bhalke** -    *rrbhalke*
