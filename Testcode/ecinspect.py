import sys
import transformer_modules
import transformers
import pytorch_lightning as pl
from datasets import load_from_disk
from util import load_config, get_class_object
from numpy.random import default_rng
import pandas as pd
import click
import os.path
rng = default_rng()

def make_base_dir(model_name, exp_name, ver_name):
    return f"models/{model_name}/logs/{exp_name}/{ver_name}"

def get_dataset_path(tokenizer_name, linearizer_name, split='test'):
    return os.path.join('data', f'GEMSGD_{tokenizer_name}{linearizer_name}_{split}')

def sample_from_dataset(dataset_path, chosen_idx=[], seed=20, size=10):
    dataset = load_from_disk(dataset_path)
    dataset.set_format('torch', columns=['input_ids', 'labels'])
    dataset = dataset[chosen_idx] if len(chosen_idx) else dataset[rng.choice(seed, size=size, replace=False)]
    return dataset

def get_checkpoint_path(model_name, exp_name, ver_name, epoch_num=7, step_num=5139, checkpoint_name=None):
    base_dir = make_base_dir(model_name, exp_name, ver_name)
    checkpoint_path = os.path.join(base_dir, 'checkpoints', checkpoint_name or f'epoch={epoch_num}-step={step_num}.ckpt')
    return checkpoint_path

def load_model(model_class_obj, checkpoint_path):
    model = model_class_obj.load_from_checkpoint(checkpoint_path=checkpoint_path)
    return model

# def load_tokenizer(tokenizer_class):
#     if tokenizer

def forward_model(model, dataset):
    outputs = model.forward(dataset['input_ids'])
    return outputs

def batch_decode(texts, tokenizer):
    decoded_text = tokenizer.batch_decode(texts, skip_special_tokens=True)
    return decoded_text

def construct_df(model, tokenizer, dataset, with_model=False):
    input_text = tokenizer.batch_decode(dataset['input_ids'], skip_special_tokens=True)
    target_text = tokenizer.batch_decode(dataset['labels'], skip_special_tokens=True)
    if with_model:
        output = model.forward(dataset['input_ids'])
        pred_text = tokenizer.batch_decode(output, skip_special_tokens=True)
        df['pred'] = pred_text
    df = pd.DataFrame()
    df['input'] = input_text
    df['target'] = target_text
    return df

def predict_with_model(model_name, exp_name, ver_name, epoch_num=7, step_num=5139, checkpoint_name=None, chosen_idx=[], seed=20, size=20, with_model=False):
    base_dir = make_base_dir(model_name, exp_name, ver_name)
    config_file = os.path.join(base_dir, 'config.yaml')
    config = load_config(config_file)
    model_class_object = get_class_object(transformer_modules, config['LightningModuleName'])
    checkpoint_path = get_checkpoint_path(model_name, exp_name, ver_name, epoch_num=epoch_num, step_num=step_num, checkpoint_name=checkpoint_name)
    tokenizer_name = config['LightningDataModuleParas']['tokenizer_class']
    tokenizer_class = get_class_object(transformers, tokenizer_name)
    linearizer_class = config['LightningDataModuleParas']['linearizer_class']
    dataset_path = get_dataset_path(tokenizer_name, linearizer_class)

    model = None
    if with_model:
        model = load_model(model_class_object, checkpoint_path)
    tokenizer = tokenizer_class.from_pretrained(config['LightningDataModuleParas']['tokenizer_path'])
    test_dataset = sample_from_dataset(dataset_path, chosen_idx=chosen_idx, seed=seed, size=size)
    res_df = construct_df(model, tokenizer, test_dataset, with_model=with_model)
    return res_df

def get_bad_case(annotated_file, by='has_slot_error', limit=20):
    anno_df = pd.read_csv(annotated_file)
    if by == 'has_slot_error':
        df  =anno_df[anno_df[by] == True] 
    elif by in {'PARENT-recall', 'PARENT-precision'}:
        df = anno_df.sort_values(by, na_position='last')
    else:
        raise ValueError("Invalid bad case clue!")
    if limit > 0:
        return df[:limit]
    else:
        return df
    

import ast
class PythonLiteralOption(click.Option):

    def type_cast_value(self, ctx, value):
        try:
            return ast.literal_eval(value)
        except:
            raise click.BadParameter(value)

def inspect_model(model_name, exp_name, ver_name, test_name, bad_case_by="", limit=-1, chosen_idx=[], checkpoint_name="", seed=20, dst="", with_model=False):
    click.echo(f"Inspecting {model_name}, {exp_name}, {ver_name}")
    res_df = None
    if test_name and bad_case_by:
        test_dir = make_base_dir(model_name, exp_name, test_name)
        annotated_file = os.path.join(test_dir, 'annotated_cases.csv')
        anno_df = pd.read_csv(annotated_file)
        bad_anno_df = None
        if not len(chosen_idx):
            bad_anno_df = get_bad_case(annotated_file, by=bad_case_by, limit=limit)
        chosen_idx = chosen_idx if len(chosen_idx) else bad_anno_df.index
        res_df = predict_with_model(model_name, exp_name, ver_name, chosen_idx=chosen_idx, with_model=with_model)
        res_df.index = chosen_idx
        if not with_model:
            res_df['pred'] = anno_df.iloc[chosen_idx]['pred'].tolist()
        res_df[bad_case_by] = anno_df.iloc[res_df.index][bad_case_by].tolist() # join bad_case_by information
    else:
        res_df = predict_with_model(model_name, exp_name, ver_name, checkpoint_name=checkpoint_name, seed=seed, size=limit, chosen_idx=chosen_idx)
        res_df.index = chosen_idx if len(chosen_idx) else res_df.index
    return res_df

@click.command()
@click.option('--model_name', default="", help="name of the model")
@click.option('--exp_name', default="", help="name of the experiment")
@click.option('--ver_name', default="", help="version identifier")
@click.option('--test_name', default='', help='name of the test (decoding)')
@click.option('--bad_case_by', default='', help='Using which criterion to get bad cases [has_slot_error, PARENT-recall, PARENT-precision]')
@click.option('--limit', default=20, type=int, help='upper limit of bad cases')
@click.option('--chosen_idx', cls=PythonLiteralOption, default='[]')
@click.option('--checkpoint_name', help="name of checkpoint")
@click.option('--seed', default=20, type=int, help='random seed')
@click.option('--dst', default='', help='location to store the file')
@click.option('--config_file', default='', help='configuration file to perform comparison')
@click.option('--with_model', default=False, type=bool, help="If true do inference real time")
def ecinspect(model_name, exp_name, ver_name, test_name, bad_case_by, limit, chosen_idx, checkpoint_name, seed, dst, config_file, with_model):
    if not config_file: # single inspection
        res_df = inspect_model(model_name, exp_name, ver_name, test_name, bad_case_by, limit, chosen_idx, checkpoint_name, seed, dst, with_model) 
    else:
        config = load_config(config_file)
        res_df = inspect_model(**config['reference_model'])
        for k, vdict in config.items():
            if k == 'reference_model':
                continue
            tmp_df = inspect_model(**vdict, chosen_idx=res_df.index)
            res_df = res_df.join(tmp_df, rsuffix=f"_{k}", how='left')
    click.echo(res_df.head(len(res_df)))
    if dst:
        res_df.to_csv(dst)

if __name__ == '__main__':
    ecinspect()