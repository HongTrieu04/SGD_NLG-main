import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
import torch
import transformer_modules
import data_modules
import argparse
from util import load_config, get_class_object, get_tokenizer
import os.path
from pytorch_lightning.callbacks import ModelCheckpoint
from shutil import copyfile
import wandb



parser = argparse.ArgumentParser(description="Model training")
parser.add_argument('--model_name', type=str, help='path to the config file')

if __name__ == '__main__':
    args = parser.parse_args()
    config_file = os.path.join('models', args.model_name, 'config.yaml')
    config = load_config(config_file)
    root_dir = os.path.join('models', args.model_name) 
    exp_name = config['TrainLoggerParas']['name']
    exp_version = config['TrainLoggerParas']['version']
    exp_dir =os.path.join(root_dir, 'logs', exp_name, exp_version)
    os.makedirs(exp_dir, exist_ok=True)

    if config.get('LightningModuleParas',{}).get("model_path", None):
        if os.path.isfile(os.path.join(root_dir, config['LightningModuleParas']['model_path'])):
            config['LightningModuleParas']['model_path'] = os.path.join(root_dir, config['LightningModuleParas']['model_path'])
    # test_dir = os.path.join(root_dir, exp_name, exp_version)
    print("Save config file to logging directory")
    copyfile(os.path.join(root_dir, 'config.yaml'), \
        os.path.join(exp_dir, 'config.yaml'))
    
    tokenizer = get_tokenizer(config)

    DataModuleClass = get_class_object(data_modules, config['LightningDataModuleName']) 
    dm = DataModuleClass(tokenizer=tokenizer, **config['LightningDataModuleParas'])
        
    LightningModuleClass = get_class_object(transformer_modules, config['LightningModuleName'])
    model = None
    if os.path.isfile(config['LightningModuleParas']['model_path']): # load from checkpoint
        checkpoint_path = config['LightningModuleParas']['model_path']
        del config['LightningModuleParas']['model_path']
        model = LightningModuleClass.load_from_checkpoint(tokenizer=tokenizer, checkpoint_path=checkpoint_path, **config['LightningModuleParas'])
    else:
        model = LightningModuleClass(tokenizer=tokenizer, **config['LightningModuleParas'])

    trainer_paras = config['TrainerParas']
    if torch.cuda.is_available():
        trainer_paras.update({'accelerator': 'gpu', 'devices': 1})
    trainer_paras.update({'default_root_dir': root_dir})
    LoggerClass = get_class_object(pl_loggers, config['TrainLoggerName']) 
    if LoggerClass == pl_loggers.WandbLogger:
        # wandb.init(settings=wandb.Settings(start_method="fork"))
        logger = LoggerClass(**config['TrainLoggerParas'], config=config)
    else:
        logger = LoggerClass(os.path.join(root_dir, 'logs'), **config['TrainLoggerParas'])
    
    checkpoint_callback = ModelCheckpoint(**config.get('ModelCheckpointParas'))

    trainer = pl.Trainer(**trainer_paras, logger=logger, callbacks=[checkpoint_callback])
    trainer.fit(model, dm)