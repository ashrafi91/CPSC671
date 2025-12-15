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
| MMCANet       | 0.889             | 0.985              | 0.057              | 0.504              | 0.709              | -                  |

## Authors

- Adnan Ferdous Ashrafi [@ashrafi91](https://github.com/ashrafi91)


## Code Helpers from public repos

- Also referred to as medical coding, clinical coding, or simply ICD coding in other literature. They may have different meanings in detail.
- Mullenbach, et al., Explainable Prediction of Medical Codes from Clinical Text, NAACL 2018 ([paper](https://arxiv.org/abs/1802.05695), [code](https://github.com/jamesmullenbach/caml-mimic))
- Li and Yu, ICD Coding from Clinical Text Using Multi-Filter Residual Convolutional Neural Network, AAAI 2020 ([paper](https://arxiv.org/abs/1912.00862), [code](https://github.com/foxlf823/Multi-Filter-Residual-Convolutional-Neural-Network))
- Ji, et al., Dilated Convolutional Attention Network for Medical Code Assignment from Clinical Text, Clinical NLP Workshop 2020 ([paper](https://aclanthology.org/2020.clinicalnlp-1.8/), [code](https://github.com/shaoxiongji/DCAN))
- Biswas, et al., TransICD: Transformer Based Code-wise Attention Model for Explainable ICD Coding, AIME 2021 ([paper](https://arxiv.org/abs/2104.10652), [code](https://github.com/AIMedLab/TransICD))
- Luo, et al., Fusion: Towards Automated ICD Coding via Feature Compression, ACL 2020 Findings ([paper](https://aclanthology.org/2021.findings-acl.184/), [code](https://github.com/machinelearning4health/Fusion-Towards-Automated-ICD-Coding))
