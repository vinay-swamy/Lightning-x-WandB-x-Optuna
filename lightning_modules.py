import torch 
from torch import nn 
import pytorch_lightning as pl
from torchvision import datasets
from torchvision.transforms import ToTensor

class LitDataModule(pl.LightningDataModule):
    def ___init__(self):
        train_data = datasets.MNIST(
            root = 'data',
            train = True,                         
            transform = ToTensor(), 
            download = True,            
        )
        test_data = datasets.MNIST(
            root = 'data', 
            train = False, 
            transform = ToTensor()
        )
        self.train_data=  torch.utils.data.DataLoader(train_data, 
                                          batch_size=100, 
                                          shuffle=True, 
                                          num_workers=1),
    
        self.validation_data  = torch.utils.data.DataLoader(test_data, 
                                          batch_size=100, 
                                          shuffle=True, 
                                          num_workers=1)
        self.test_data  = torch.utils.data.DataLoader(test_data, 
                                          batch_size=100, 
                                          shuffle=True, 
                                          num_workers=1)
    
    ### for train_dataloader return a single dataloader, 
    ## for test and validation, you can return a list of dataloaders, and all will be used 
    def train_dataloader(self):
        return self.train_data
    def val_dataloader(self):
        return self.validation_data
    def test_dataloader(self):
        return self.test_data




class LitModelWrapper(pl.LightningModule):
    def __init__(self,model, loss_config, optim_config  ):
        """
        Given a model, a loss function and optimizer, implement the methods train_step, validation_step and test_step for them 
        Args:
            model (nn.Module): an instanced pytorch model
            loss_config (dict: dict from mconfig
            optim_config (dict): "" ""
        """        
        super().__init__() ## Lightning modules do not need the named super like vanilla pytorch 
        self.model = model
        _loss_fn = eval(loss_config['loss_fn'])
        self.loss_fn = _loss_fn(**loss_config['loss_fn_kwargs'])
        self.optim_config = optim_config
    def forward(self, x):
        logits,pred_prob = self.model(x)
        return logits,pred_prob
    def training_step(self, batch,batch_idx):
        X, label = batch
        batch_size = X.shape[0]
        logits,pred_prob = self.model(X)
        loss = self.loss_fn(logits, label)
        ## use self.log to log data to whatever type of logger you want( logger is handled by pl::Trainer)
        ## its important to include batch size to make sure things are averaged properly across multiple devices 
        self.log('train_loss', loss, batch_size=batch_size)
        opt = self.optimizers()
        self.log('learning_rate', opt.optimizer.param_groups[0]['lr'], batch_size=batch_size)
        return loss 
    def validation_step(self, batch, batch_idx, dataset_idx): ## these arguments are required for Lighting, but you dont need to use them 
        ## dont need to worry about with torch.no_grad() and model.eval(), Lighting handles it for you 
        X, label = batch
        batch_size = X.shape[0]
        logits,pred_prob = self.model(X)
        loss = self.loss_fn(logits, label)

        self.log(f"validation", loss, sync_dist = True, batch_size=batch_size) # sync_dists = True makes sure metric is averaged across multiple gpus; if set false, only gives bck data from 0th process 
        
    def test_step(self, batch, batch_idx, dataset_idx):## these arguments are required for Lighting, but you dont need to use them 
        ## dont need to worry about with torch.no_grad() and model.eval(), Lighting handles it for you 
        X, label = batch
        batch_size = X.shape[0]
        logits,pred_prob = self.model(X)
        loss = self.loss_fn(logits, label)

        self.log(f"test", loss, sync_dist = True, batch_size=batch_size) # sync_dists = True makes sure metric is averaged across multiple gpus; if set false, only gives bck data from 0th process 
        

    def configure_optimizers(self):
        optim_fn = eval(self.optim_config['optim_fn'])
        optimizer = optim_fn(self.parameters(), **self.optim_config['optim_kwargs'])
        
        ### You can also use a learning rate scheduler here, but Ive commented it out for simplicity
        # sched_fn = eval(self.optim_config["scheduler"]) 
        # scheduler = sched_fn(optimizer,  **self.optim_config['scheduler_kwargs'])
        # scheduler_config = {
        #     "scheduler": scheduler,
        #     "interval": "step",
        #     "name":"learning_rate"
        # }
        optimizer_dict = {"optimizer" : optimizer#, 
            #"lr_scheduler" : scheduler_config
         }
        return optimizer_dict
