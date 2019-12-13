# CoPER-ConvE & ConvE

## Requirements
```$pip install -r requirements.txt```

## Experiment Pipeline
We describe how to run experiments using a demo example. Let us say that we wanted to examine the performance of CoPER-ConvE on the WN18RR dataset. 
1. We first need to specify the exact experiment configuration such as training time, model hyperparameters, and parameter generation architecture. Thus, navigate to 'configs/config_[dataset]_[experiment_type].yaml', where '[dataset]' in this case is WN18RR and '[experiment_type]' is 'cpg'


Please follow the steps below to run our code.


Further, please note that our code is compatible with python3.6

1) pip install -r requirements.txt 
2) To set the experiment configuration, navigate to src/qa_cpg/run_cpg.py:
    - To change the Dataset, replace 'X' in 'data.X' on line 43 to one of the following: 
        - 'NationsLoader', 'UMLSLoader', 'KinshipLoader', 'WN18RRLoader', 'YAGO310Loader', 
           'FB15k237Loader', 'CountriesS1Loader', 'CountriesS2Loader', 'CountriesS3Loader', 'NELL995Loader'
    - To run CoPER, set 'use_cpg' to True on line 39, otherwise to run ConvE set it to False
3) Our model parameters for each dataset and model type (i.e. CoPER or ConvE) is specified in the config 
    files found in /src/cpg/configs/. Each filename follows the following naming pattern: 'config_X_Y.yaml':
      - X is the dataset name.
      - Y is the model name (cpg is CoPER, plain is ConvE)
   If you would like to change the hyperparameters of your desired experiment before you run, you may change
     them in the respective config file associated with the settings you chose in step 2.
4) Once you have decided on the experiment type and configuration, navigate back to the /src/
5) execute python3.6 -m qa_cpg.run_cpg to start the experiment.
