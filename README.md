# cnnPTBXL
Pipeline for ECG analysis using CNN´s (PTBXL data set)

Model can be fitted/trained by running command; 'train.py fit -c config_v1.yaml' in terminal

Parametres and hyper-parametres are set in config file.
Additional config files can be created with .yaml file ending, to test different configurations.
Checkpoints from vaious runs are saved in folder; 'lightning logs'

To test model run command; 'train.py test -c config_v1.yaml'
