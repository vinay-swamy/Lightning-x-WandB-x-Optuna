#%%
### This script is used to train model via pytorch lightning logging to wandb 
### It must be called from the modelling repo, which is set in the config  
from lightning_modules import *
from models import *
from pathlib import Path
import wandb
import json 
import os
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import argparse
import random
import string
import subprocess as sp
import glob 
import time 
import pandas as pd 

def generate_random_string(length):
    alphanumeric = string.ascii_letters + string.digits
    return ''.join(random.choice(alphanumeric) for i in range(length))
def update_mconfig_from_wandb(mconfig, sweep_dict, ignore_keys):
    sweep_keys = [k for k in sweep_dict.keys() if k not in ignore_keys ]
    for skey in sweep_keys:
        key_path = skey.split("-")
        c_dict = mconfig
        print("")
        for _key in key_path[:-1]:
            c_dict = c_dict[_key]
        ## preserve original datatype of parameter
        orig_dtype = type(c_dict[key_path[-1]])
        c_dict[key_path[-1]]=orig_dtype(sweep_dict[skey])
    return mconfig

torch.set_float32_matmul_precision('medium') ## this sets the gpu precision for 32 bit ops, lower means less precision but faster 
filesystem = os.environ["WHEREAMI"]
## ^This makes it easier to switch between different machines;  WHEREAMI is set in the .bashrc file and is the location of where we store repos; 
## on manitou its /manitou/pmg/users/vss2134, exxmini its /data/vss2134, aws its /data and so on 

parser = argparse.ArgumentParser(
                    prog='Lighting+W&B model training' ,
                    description='This script is used to train model via pytorch lightning',
                    )
parser.add_argument('--config', type=str, help='path to model config file',required=True)
parser.add_argument('--version', type=str, help='run version', default=None)
parser.add_argument('--name', type=str, help='run name', default=None, required=True)
parser.add_argument('--mode', type=str, help='[train, resume, test]', default=None, required=True)
parser.add_argument('--ckpt-file', type=str, help='name of checkpoint file only, no paths', default=None)
parser.add_argument('--override-config', type=str, help='wandb sweep style cl args that will be parsed and will update config accordingly', default=None)

args = parser.parse_args()

def main(args):
    mconfig_file = args.config
    with open(mconfig_file) as m_stream:
        mconfig = json.load(m_stream)

    name = args.name
    mode = args.mode
    if mode == "test":
        ### run a miminal test run to make sure everything is working
        print("***TEST***")
        mconfig['training']['max_epochs']=1
        mconfig['data']["max_steps_per_epoch"]=2000
        mconfig['data']["num_workers"]=1
        mconfig['wandb_project']='test'
        name = "test"
        mconfig['training']['devices']=1
        del mconfig['training']['strategy']
    version = args.version
    if version is None:
        ## this does not break for ddp processes 
        version = generate_random_string(8)
    project = mconfig['wandb_project']
    repo_name = mconfig['repo_name']
    root_dir = f"{filesystem}/{repo_name}/model_out"
    run_dir = Path(f"{root_dir}/{project}/{version}")
    if run_dir.exists() and (mode != "resume"):
        
        raise NotImplementedError(f"run_dir {str(run_dir)} already exists, bad input ")
    run_dir.mkdir(exist_ok=True, parents=True)
    
    
    override_config = args.override_config
    if override_config is not None:
        ## pass in commandline style args to override config. 
        override_args = override_config.split(",")
        params_to_update = {}
        for a in override_args:
            k,v = a.split("=")
            params_to_update[k]=v

        mconfig = update_mconfig_from_wandb(mconfig, params_to_update, set(['mconfig', 'project']) )
    ### load data 
    data_conf=mconfig['data']
    datamodule = LitDataModule(**data_conf)
    train_dl = datamodule.train_dataloader()
    val_dl = datamodule.val_dataloader()

    ## redundant but just to be safe
    outdir = Path(root_dir)

    ## init wandb logger; folder being logged to is based on version/id 
    wblg= WandbLogger(project = mconfig['wandb_project'],
                 name = name,
                 entity = 'vinay-swamy',
                 version = version,
                 id = version,
                 config=mconfig,
                 save_dir = str(outdir))
    ### save config to run dir in  case we change on the fly and want to resume 
    with open(f"{str(run_dir)}/mconfig_used.json", 'w+') as m_stream:
        json.dump(mconfig, m_stream)

    ### Set up PTL trainer 
    ## Set up multiple checkpoints for tracking each metric 
    latest_checkpoint_cb = ModelCheckpoint(filename = "latest-{epoch}-{step}")

    trainer_conf = mconfig['training']
    print(trainer_conf)

    trainer_conf['logger'] = wblg
    trainer_conf['callbacks'] = [latest_checkpoint_cb]
    trainer = pl.Trainer(**trainer_conf, default_root_dir=str(outdir))

    if (mode == "train") or (mode == "test"):
        _model = eval(mconfig['model']['name']) ## load model form config 
        litmodel = LitModelWrapper(_model(mconfig), mconfig['loss'], mconfig['optim'], mconfig['task'])
        trainer.fit(litmodel, train_dataloaders = train_dl, val_dataloaders=val_dl)
    elif mode == "resume":
        ckpt = args.ckpt_file
        ckpt_file = f"{root_dir}/{project}/{version}/checkpoints/{ckpt}"

        _model = eval(mconfig['model']['name']) ## load model fromm config  and checkpoint
        litmodel = LitModelWrapper.load_from_checkpoint(ckpt_file, 
                    model = _model(mconfig), 
                    loss_config = mconfig['loss'],
                    optim_config = mconfig['optim'], 
                    task_config = mconfig['task'])
        ### ckpt_path is required to resume training 
        trainer.fit(litmodel, train_dataloaders = train_dl, val_dataloaders=val_dl, ckpt_path = ckpt_file)
    else:
        raise NotImplementedError(f"mode {mode} not implemented")

if __name__ == "__main__":
    main(args)