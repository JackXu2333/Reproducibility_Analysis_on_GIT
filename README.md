# Reproducibility Analysis on Genomic Impact Transformer (GIT)

## Introduction

This repository contains PyTorch implementation of GIT model
and GIT variants from reproducibility analysis by Sijie Xu, and in the original paper by
Yifeng Tao, Chunhui Cai, William W. Cohen, Xinghua Lu. [From genome to phenome: Predicting multiple cancer phenotypes based on somatic genomic alterations via the genomic impact transformer](https://arxiv.org/abs/1902.00078). Proceedings of the Pacific Symposium on Biocomputing (PSB). 2020.

The preprocessed TCGA dataset, and gene embeddings mentioned in the paper, and reproducibility analysis are also released below.

Link to the Tao et al.'s [github repository](https://github.com/yifengtao/genome-transformer)

## Prerequisites

The code runs on `Python 3.6` (The original implementation by Tao et al. is in `Python 2.7`), and following packages are used:
* `PyTorch`, `Jupyter Notebook`, `pandas`, `pickle`, `numpy`, `random`, `argparse`, `os`.

The code automatically enable CUDA when available.

## Data

The required data can be downloaded to the directory:
```
cd genome-transformer
wget www.cs.cmu.edu/~yifengt/paper/2019a/data.tar.gz -O data.tar.gz
tar -zxvf data.tar.gz
```

Additionally, the pretrained gene embedding files need to be copied to `data_degraded/` and `data_noBin/` 
in order to run tests in the reproducibility analysis:
```
find data_degraded data_noBin -exec cp data/gene_emb_pretrain.npy {} \;
```

### TCGA dataset

The preprocessed SGA-DEG TCGA dataset is available at `data/dataset.pkl`, which contains SGAs, DEGs, cancer types, and barcodes of 4,468 cancer samples from TCGA:
```
data = pickle.load( open("data/dataset.pkl", "rb") )
```

### Gene embeddings

Two types of gene embeddings are available:
* Gene2Vec-pretrained gene embeddings: `data/gene_emb_pretrain.npy`;
* Gene2Vec-pretrained + GIT-finetuned gene embeddings: `data/gene_emb_finetune.npy`.

You may extract the gene embeddings in Python:
```
gene_emb_matrix = numpy.load("data/gene_emb_finetune.npy")
```

It is a 19782 by 512 matrix, where the index of each row can be mapped to a gene name through `data/idx2gene.txt`.

### Trained GIT model

The parameters of trained GIT model are stored at `data/trained_model.pth`.


## Replicate experiments

### Vanilla GIT model

You can train the GIT from scratch and then evaluate its performance on test set:
```
python test_run.py
```
You may run more GIT variants, e.g., `GIT-init`, `GIT-attn`, `GIT-can` etc., by checking the code and comments of `test_run.py`, or:
```
python test_run.py --help
```

### Reproducibility experiments

Please refer to the [MLP_Testing.ipynb](https://github.com/JackXu2333/Reproducibility_Analysis_on_GIT/blob/master/MLP_Testing.ipynb)
for code used in verifying baseline result from the original paper. 

Please refer to the [GIT_Testing.ipynb](https://github.com/JackXu2333/Reproducibility_Analysis_on_GIT/blob/master/GIT_Testing.ipynb) 
for code used in GIT model verifying empirical result from the original paper and performing **modified input test** described in the reproducibility analysis. 

Please also refer to [GIT_Testing_NOBIN.ipynb](https://github.com/JackXu2333/Reproducibility_Analysis_on_GIT/blob/master/GIT_Testing_NOBIN.ipynb)
for code used in performing **non-binary target test** described in the reproducibility analysis.

The file used in constructing the dataset for the **modified input test** and **non-binary target test**
can be found via [Data_Uilts.ipynb](https://github.com/JackXu2333/Reproducibility_Analysis_on_GIT/blob/master/Data_Uilts.ipynb)

Note that [utlis.py](https://github.com/JackXu2333/Reproducibility_Analysis_on_GIT/blob/master/utils.py)
has been modified in order to perform test in the reproducibility analysis, but its usage remains unchanged

## Others

* Q: How to prepare my own input data and run experiments on the new data?

* A: You can directly prepare the `dataset.pkl` pickle file. However, we also provide pieces of code (`prepare_dataset.py`) to faciliate you to convert more general data types (`csv` and `txt`) into the `dataset.pkl`.

First, you need to format your own data into three files: `mydata/cancer_type.txt` (each line a cancer type), `deg.csv` (each row a sample, each column an over/under-expressed gene), `sga.txt` (each row a sample name and mutated genes, comma separated).

Second, run the following code to convert and merge these files into a single `mydata/dataset.pkl` file:
```
python prepare_dataset.py --input_dir mydata --output_dir mydata
```

Third, run experiments on the new data:
```
python test_run.py --initializtion False --input_dir mydata --output_dir mydata
```


If you find the data or code from this repository helpful, please cite the original [paper](https://arxiv.org/abs/1902.00078):
```
@inproceedings{tao2020git,
  title = {From Genome to Phenome: Predicting Multiple Cancer Phenotypes based on Somatic Genomic Alterations via the Genomic Impact Transformer},
  author = {Tao, Yifeng  and 
    Cai, Chunhui  and 
    Cohen, William W.  and 
    Lu, Xinghua},
  booktitle={Pacific Symposium on Biocomputing},
  year = {2020},
}
```

