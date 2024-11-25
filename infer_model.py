import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
import torch
import transformer_modules
import data_modules
import argparse
from util import load_config, get_class_object, get_tokenizer
import os.path
from shutil import copyfile


parser = argparse.ArgumentParser(description="Model training")
parser.add_argument('--model_name', type=str, help='path to the config file')
parser.add_argument('--checkpoint_path', type=str, help='Path to checkpoint file')
if __name__ == '__main__':
    args = parser.parse_args()
    config = load_config(os.path.join('models', args.model_name, 'config.yaml'))
    root_dir = os.path.join('models', args.model_name) 
    config['LoadingParas']['checkpoint_path'] \
        = os.path.join(root_dir, config['LoadingParas']['checkpoint_path']) 
    config['LoadingParas']['hparams_file'] \
        = os.path.join(os.path.join(*config['LoadingParas']['checkpoint_path'].split('/')[:-2]), 'hparams.yaml')
    decode_dir = os.path.join(root_dir, config['LoadingParas']['decode_path'])
    config['LoadingParas']['decode_path'] = decode_dir
    if not os.path.exists(decode_dir):
      os.makedirs(decode_dir, exist_ok=True)
    print("Save config file to logging directory")
    copyfile(os.path.join(root_dir, 'config.yaml'),\
        os.path.join(decode_dir, 'config.yaml'), )
    
    tokenizer = get_tokenizer(config)

    DataModuleClass = get_class_object(data_modules, config['LightningDataModuleName']) 
    dm = DataModuleClass(tokenizer=tokenizer, **config['LightningDataModuleParas'])

    LightningModuleClass = get_class_object(transformer_modules, config['LightningModuleName'])
    model = LightningModuleClass.load_from_checkpoint(tokenizer=tokenizer, **config['LoadingParas'], **config['LightningModuleParas'])
        # model = HFT5GenerationModel.load_from_checkpoint(\
        # checkpoint_path="lightning_logs/version_48538859/checkpoints/epoch=2-step=61868.ckpt",\
        # hparams_file="lightning_logs/version_48538859/hparams.yaml")
    trainer = None
    trainer_paras = config['TrainerParas']
    if torch.cuda.is_available():
        trainer_paras.update({'gpus':1})
    trainer_paras.update({'default_root_dir': root_dir})
    logger = get_class_object(pl_loggers, config['TestLoggerName']) \
                (os.path.join(root_dir, 'logs'), **config['TestLoggerParas'])
    trainer = pl.Trainer(**trainer_paras, logger=logger)
    trainer.test(model, dm)