# pytorch-lightning-project-skeleton

This is an example project skeleton for a project with pytorch lighting.


Everything needed to specify a given run of a model is specified in a config file (`model/configs/base_model.json`), so that a given run can be easily tracked and replicated. 


### over all workflow. 

- Define layers. The arguments in the layer must match the arguments in the config 