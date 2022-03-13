# GSLRDA
============

This is a tensorflow implementation with a model for ncRNA-drug resistance association prediction as described in our paper：

“Graph Neural Network with Self-supervised Learning for ncRNA-Drug Resistance Association Prediction”

By 









## Requirements
* TensorFlow 1.15
* python 3.7
* numpy 1.19
* pandas 1.1
* scikit-learn 1.0
* scipy 1.5

## Run the demo

```bash
python mian.py
```

## Data
Known ncRNA-drug resistance association pairs come from the NoncoRNA and ncDR databases.
We sorted out two databases and obtained 121 drug resistance-related 625 ncRNAs by removing redundancy. There are 2,693 association pairs between them.


## Cite

Please cite our paper if you use this code in your own work:

“Graph Neural Network with Self-supervised Learning for ncRNA-Drug Resistance Association Prediction”
