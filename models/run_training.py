#%%
from re import M
from lightning_modules import *
from models import *
from pathlib import Path 
import sys
import json 
import wandb
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

## This makes it easier to switch between different machines

args = sys.argv[1:]
if len(args) == 1:
    ## using a run 
    configfile =  f"{sys.argv[1]}"
    with open(configfile) as m_stream:
        mconfig = json.load(m_stream)
elif args[1] == "TEST":
    configfile =  f"/data/ChIP-in-a-chip//{sys.argv[1]}"
    with open(configfile) as m_stream:
        mconfig = json.load(m_stream)
    mconfig['training']['max_epochs']=1
    mconfig['data']["max_steps_per_epoch"]=2000
    mconfig['wandb_project']='test'
    # mconfig['training']['devices']=1
    # del mconfig['training']['strategy']
else:
    ## This is for running a hyperparamter sweep via the wandb sweep program, or by manually passing commandline arguments. Commenting it out fo rnow 
    # params_to_update = {}

    # if args[0][:2] != "--": # if we are manually passing a sweep
    #     with open(args[0]) as m_stream:
    #         mconfig = json.load(m_stream)
    #     args = args[1:]
    #     for a in args:
    #         k,v = a.replace("--", '').split("=")
    #         params_to_update[k]=v
    # else:
    #     for a in args:
    #         k,v = a.replace("--", '').split("=")
    #         params_to_update[k]=v
    #     with open(f"/data/ChIP-in-a-chip/{params_to_update['mconfig']}") as m_stream:
    #         mconfig = json.load(m_stream)
    
    # mconfig = update_mconfig_from_wandb(mconfig, params_to_update, set(['mconfig', 'project']) )
    pass




## Load data
data_conf=mconfig['data']
datamodule = LitDataModule(**data_conf)
train_dl = datamodule.train_dataloader()
val_dl = datamodule.val_dataloader()
test_dl = datamodule.test_dataloader()

## instance model 
_model = eval(mconfig['model_fn'])
litmodel = LitModelWrapper(model = _model(**mconfig['model_kwargs']), loss_config= mconfig['loss'], optim_config = mconfig['optim'])


## instance wandb logger object
plg= WandbLogger(project = mconfig['wandb_project'],
                 entity = 'vinay-swamy', 
                 config=mconfig) ## include your run config so that it gets logged to wandb 
plg.watch(litmodel) ## this logs the gradients for your model 

## add the logger object to the training config portion of the run config 
trainer_conf = mconfig['training']
trainer_conf['logger'] = plg

## pytorch lightning saves the best checkpoint of your model by default,
## but I like to save every checkpoint, which this lets you do 
checkpoint_cb = ModelCheckpoint(save_top_k=-1, every_n_epochs = None,every_n_train_steps = None, train_time_interval = None)
trainer_conf['callbacks'] = [checkpoint_cb]

trainer = pl.Trainer(**trainer_conf)

if mconfig['dryrun']: ## the dry run parameter lets you check if everythign can be loaded properly 
    print("Successfully loaded everything. Quitting")
    sys.exit()

trainer.fit(litmodel, train_dataloaders = train_dl, val_dataloaders=val_dl) ## this starts training 

out = trainer.predict(litmodel, dataloaders = test_dl)


# %%
