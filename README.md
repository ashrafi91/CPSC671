# MMACNet


## Preparation
Please put the MIMIC-III `csv.gz` files (v1.4) under `datasets/mimic3/csv/`. You can also create symbolic links pointing to the files.


## Pre-processing
```
$ python run_preprocessing.py --config_path configs/preprocessing/default/mimic3_full.yml
```


## Training / Testing
You can evaluate the model with `--test` options and use other config files under `configs`.
```
$ python run.py --config_path configs/MMACNet/MMACNet_mimic3_50.yml         # Train
$ python run.py --config_path configs/MMACNet/MMACNet_mimic3_50.yml --test  # Test
```
Training is logged through TensorBoard graph (located in the output dir under `results/`).
Also, logging through text files is performed on pre-processing, training, and evaluation. Log files will be located under `logs/`.


## Results
- MIMIC-III full

| Model        |     macro AUC      |     micro AUC      |      macro F1      |      micro F1      |         P@8        |        P@15        |
|--------------|--------------------|--------------------|--------------------|--------------------|--------------------|--------------------|
| CNN          | 0.835&plusmn;0.001 | 0.974&plusmn;0.000 | 0.034&plusmn;0.001 | 0.420&plusmn;0.006 | 0.619&plusmn;0.002 | 0.474&plusmn;0.004 |
| CAML         | 0.893&plusmn;0.002 | 0.985&plusmn;0.000 | 0.056&plusmn;0.006 | 0.506&plusmn;0.006 | 0.704&plusmn;0.001 | 0.555&plusmn;0.001 |
| MultiResCNN  | 0.912&plusmn;0.004 | 0.987&plusmn;0.000 | 0.078&plusmn;0.005 | 0.555&plusmn;0.004 | 0.741&plusmn;0.002 | 0.589&plusmn;0.002 |
| DCAN         | 0.848&plusmn;0.009 | 0.979&plusmn;0.001 | 0.066&plusmn;0.005 | 0.533&plusmn;0.006 | 0.721&plusmn;0.001 | 0.573&plusmn;0.000 |
| TransICD     | 0.886&plusmn;0.010 | 0.983&plusmn;0.002 | 0.058&plusmn;0.001 | 0.497&plusmn;0.001 | 0.666&plusmn;0.000 | 0.524&plusmn;0.001 |
| Fusion       | 0.910&plusmn;0.003 | 0.986&plusmn;0.000 | 0.081&plusmn;0.002 | 0.560&plusmn;0.003 | 0.744&plusmn;0.002 | 0.589&plusmn;0.001 |

- MIMIC-III top-50

| Model        |     macro AUC      |     micro AUC      |      macro F1      |      micro F1      |         P@5        |
|--------------|--------------------|--------------------|--------------------|--------------------|--------------------|
| CNN          | 0.913&plusmn;0.002 | 0.936&plusmn;0.002 | 0.627&plusmn;0.001 | 0.693&plusmn;0.003 | 0.649&plusmn;0.001 |
| CAML         | 0.918&plusmn;0.000 | 0.942&plusmn;0.000 | 0.614&plusmn;0.005 | 0.690&plusmn;0.001 | 0.661&plusmn;0.002 |
| MultiResCNN  | 0.928&plusmn;0.001 | 0.950&plusmn;0.000 | 0.652&plusmn;0.006 | 0.720&plusmn;0.002 | 0.674&plusmn;0.001 |
| DCAN         | 0.934&plusmn;0.001 | 0.953&plusmn;0.001 | 0.651&plusmn;0.010 | 0.724&plusmn;0.005 | 0.682&plusmn;0.003 |
| TransICD     | 0.917&plusmn;0.002 | 0.939&plusmn;0.001 | 0.602&plusmn;0.002 | 0.679&plusmn;0.001 | 0.643&plusmn;0.001 |
| Fusion       | 0.932&plusmn;0.001 | 0.952&plusmn;0.000 | 0.664&plusmn;0.003 | 0.727&plusmn;0.003 | 0.679&plusmn;0.001 |

- MIMIC-III full (old)

| Model        |     macro AUC      |     micro AUC      |      macro F1      |      micro F1      |         P@8        |        P@15        |
|--------------|--------------------|--------------------|--------------------|--------------------|--------------------|--------------------|
| CNN          | 0.833&plusmn;0.003 | 0.974&plusmn;0.000 | 0.027&plusmn;0.005 | 0.419&plusmn;0.006 | 0.612&plusmn;0.004 | 0.467&plusmn;0.001 |
| CAML         | 0.880&plusmn;0.003 | 0.983&plusmn;0.000 | 0.057&plusmn;0.000 | 0.502&plusmn;0.002 | 0.698&plusmn;0.002 | 0.548&plusmn;0.001 |
| MultiResCNN  | 0.905&plusmn;0.003 | 0.986&plusmn;0.000 | 0.076&plusmn;0.002 | 0.551&plusmn;0.005 | 0.738&plusmn;0.003 | 0.586&plusmn;0.003 |
| DCAN         | 0.837&plusmn;0.005 | 0.977&plusmn;0.001 | 0.063&plusmn;0.002 | 0.527&plusmn;0.002 | 0.721&plusmn;0.001 | 0.572&plusmn;0.001 |
| TransICD     | 0.882&plusmn;0.010 | 0.982&plusmn;0.001 | 0.059&plusmn;0.008 | 0.495&plusmn;0.005 | 0.663&plusmn;0.007 | 0.521&plusmn;0.006 |
| Fusion       | 0.910&plusmn;0.003 | 0.986&plusmn;0.000 | 0.076&plusmn;0.007 | 0.555&plusmn;0.008 | 0.744&plusmn;0.003 | 0.588&plusmn;0.003 |

- MIMIC-III top-50 (old)

| Model        |     macro AUC      |     micro AUC      |      macro F1      |      micro F1      |         P@5        |
|--------------|--------------------|--------------------|--------------------|--------------------|--------------------|
| CNN          | 0.892&plusmn;0.003 | 0.920&plusmn;0.003 | 0.583&plusmn;0.006 | 0.652&plusmn;0.008 | 0.627&plusmn;0.007 |
| CAML         | 0.865&plusmn;0.017 | 0.899&plusmn;0.008 | 0.495&plusmn;0.035 | 0.593&plusmn;0.020 | 0.597&plusmn;0.016 |
| MultiResCNN  | 0.898&plusmn;0.006 | 0.928&plusmn;0.003 | 0.590&plusmn;0.012 | 0.666&plusmn;0.013 | 0.638&plusmn;0.005 |
| DCAN         | 0.915&plusmn;0.002 | 0.938&plusmn;0.001 | 0.614&plusmn;0.001 | 0.690&plusmn;0.002 | 0.653&plusmn;0.004 |
| TransICD     | 0.895&plusmn;0.003 | 0.924&plusmn;0.002 | 0.541&plusmn;0.010 | 0.637&plusmn;0.003 | 0.617&plusmn;0.005 |
| Fusion       | 0.904&plusmn;0.002 | 0.930&plusmn;0.001 | 0.606&plusmn;0.009 | 0.677&plusmn;0.003 | 0.640&plusmn;0.001 |


## Authors

- Adnan Ferdous Ashrafi [@ashrafi91](https://github.com/ashrafi91)


## Code Helpers from public repos

[^1]: Also referred to as medical coding, clinical coding, or simply ICD coding in other literature. They may have different meanings in detail.
[^2]: Mullenbach, et al., Explainable Prediction of Medical Codes from Clinical Text, NAACL 2018 ([paper](https://arxiv.org/abs/1802.05695), [code](https://github.com/jamesmullenbach/caml-mimic))
[^3]: Li and Yu, ICD Coding from Clinical Text Using Multi-Filter Residual Convolutional Neural Network, AAAI 2020 ([paper](https://arxiv.org/abs/1912.00862), [code](https://github.com/foxlf823/Multi-Filter-Residual-Convolutional-Neural-Network))
[^4]: Ji, et al., Dilated Convolutional Attention Network for Medical Code Assignment from Clinical Text, Clinical NLP Workshop 2020 ([paper](https://aclanthology.org/2020.clinicalnlp-1.8/), [code](https://github.com/shaoxiongji/DCAN))
[^5]: Biswas, et al., TransICD: Transformer Based Code-wise Attention Model for Explainable ICD Coding, AIME 2021 ([paper](https://arxiv.org/abs/2104.10652), [code](https://github.com/AIMedLab/TransICD))
[^6]: Luo, et al., Fusion: Towards Automated ICD Coding via Feature Compression, ACL 2020 Findings ([paper](https://aclanthology.org/2021.findings-acl.184/), [code](https://github.com/machinelearning4health/Fusion-Towards-Automated-ICD-Coding))
