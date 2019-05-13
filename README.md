# Disk Embeddings

Anonymized version of implementation of Disk Embeddings.
Full version will be available online in camera-ready version.

This code is derived from [Ganea et al. 2018](https://github.com/dalab/hyperbolic_cones),
and there might be comments by them or their collaborators, however, the authors are not aware of it.



We conducted the experiments with python 3.6.6,
Ubuntu 18.04.1 LTS in AWS EC2 c5.18xlarge instance.

### Baseline methods

Evaluated baseline methods are as follows:

- Poincare Embeddings (Nickel et.al., 2017)
- Order Embeddings (Vendrov et.al., 2016)
- Hyperbolic Entailment Cones (Ganea et.al., 2018)


### Data Preparation

- We used WordNet data already processed by [Ganea et al. 2018](https://github.com/dalab/hyperbolic_cones).
- Preprocess on Hep-Th was made in `notebook/Citation.ipynb`


### Evaluation

For single worker,

```
python run.py RunAll --local-scheduler
```


For parallelization, `luigid` scheduler is required to be running.

```
python run.py RunAll --workers=32
```

