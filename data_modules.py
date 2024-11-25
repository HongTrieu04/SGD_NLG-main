import pytorch_lightning as pl 
from datasets import load_dataset, Dataset
import transformers
from torch.utils.data import DataLoader
import os.path
from typing import Optional
from util import get_class_object
import pandas as pd
from linearizer import (
    SGD_NaiveLinearizer, 
    SGD_PaperNaiveLinearizer, 
    SGD_SchemaGuidedLinearizer, 
    SGD_TemplateGuidedLinearizer,
    SGD_SchemaGuidedWithServiceLinearizer
)
from pyarrow import Table

def linearize_with_fn(input_field, output_field, linear_fn):
    def linear_wrapped(example):
        linearized_dict = {output_field: linear_fn(example[input_field])}
        return linearized_dict
    return linear_wrapped

def tokenize_function(tokenizer, field, **kwargs):
    def tokenize_function_wrapped(example):
        return tokenizer.batch_encode_plus(example[field], **kwargs)
    return tokenize_function_wrapped

class GEMSGD_DataModule(pl.LightningDataModule):
    def __init__(self, batch_size=4, tokenizer=None, tokenizer_name='t5-small', \
                    force_process=False, save_cache=True, encode_args=None, linearizer_class='NONE', schema_paths=None, template_dir=None):
        super().__init__()
        self.batch_size = batch_size
        self.tokenizer_name = tokenizer_name
        # self.tokenizer = get_class_object(transformers, self.tokenizer_class).from_pretrained(self.tokenizer_path)
        if tokenizer is None:
            raise ValueError("Tokenizer must not be None")
        self.tokenizer = tokenizer 
        self.force_process = force_process
        self.save_cache = save_cache
        self.encode_args = encode_args
        self.linearizer_class = linearizer_class
        self.schema_paths = schema_paths
        self.template_dir = template_dir
        self.dataset_prefix = self.__class__.__name__.replace('DataModule', '').replace('_','')
        self.dataset_dir = os.path.join('data',self.dataset_prefix+'_'+self.tokenizer_name+'_'+self.linearizer_class)

    def prepare_data(self):
        self.dataset = load_dataset('gem', 'schema_guided_dialog')
        self.act_id2name_map = {i : n for i, n in enumerate(self.dataset['train'].info.features['dialog_acts'][0]['act'].names)}

        if self.linearizer_class == 'SGD_NaiveLinearizer':
            self.linearizer = SGD_NaiveLinearizer(self.act_id2name_map)
        elif self.linearizer_class == 'SGD_PaperNaiveLinearizer':
            self.linearizer = SGD_PaperNaiveLinearizer(self.act_id2name_map)
        elif self.linearizer_class == 'SGD_SchemaGuidedLinearizer':
            self.linearizer = SGD_SchemaGuidedLinearizer(self.act_id2name_map, self.schema_paths)
        elif self.linearizer_class == 'SGD_TemplateGuidedLinearizer':
            self.linearizer = SGD_TemplateGuidedLinearizer(self.act_id2name_map, self.template_dir)
        elif self.linearizer_class == 'SGD_SchemaGuidedWithServiceLinearizer':
            self.linearizer = SGD_SchemaGuidedWithServiceLinearizer(self.act_id2name_map, self.schema_paths)
        else:
            print('')
            raise ValueError('Invalid linearizer class')

    def _process_dataset_split(self, dataset, tokenizer):
        linearized_dataset = dataset.map(self.linearizer, load_from_cache_file=False)
        print(linearized_dataset[:20])
        
        tokenized_dataset = linearized_dataset \
                        .map(tokenize_function(tokenizer, 'target', **self.encode_args), batched=True, load_from_cache_file=False) \
                        .rename_column('input_ids','labels')\
                        .rename_column('attention_mask', 'output_mask')\
                        .map(tokenize_function(tokenizer, '_linearized', **self.encode_args), batched=True, load_from_cache_file=False)\
                        .map(lambda example: {'dialog_acts': [{k: self.act_id2name_map[v] if k=='act' else v for k,v in action.items()} for action in example['dialog_acts']]})
        res_dataset = tokenized_dataset
        return res_dataset

    def setup(self, stage: Optional[str] = None):
        train_data_dir = self.dataset_dir+'_train'
        val_data_dir = self.dataset_dir+'_val'
        test_data_dir = self.dataset_dir+'_test'
        if stage in (None, "fit"):
            if os.path.isdir(train_data_dir) and not self.force_process:
                print("Read", train_data_dir)
                self.train_dataset = Dataset.load_from_disk(train_data_dir)
            else:
                print("Processing", train_data_dir)
                self.train_dataset = self._process_dataset_split(self.dataset['train'], self.tokenizer)
                if self.save_cache:
                    self.train_dataset.save_to_disk(train_data_dir)
            if os.path.isdir(val_data_dir) and not self.force_process:
                print("Read", val_data_dir)
                self.val_dataset = Dataset.load_from_disk(val_data_dir)
            else:
                print("Processing", val_data_dir)
                self.val_dataset = self._process_dataset_split(self.dataset['validation'], self.tokenizer)
                if self.save_cache:
                    self.val_dataset.save_to_disk(val_data_dir)
            # self.train_dataset.set_format('torch', columns=['input_ids', 'labels', 'attention_mask'])
            # self.val_dataset.set_format('torch', columns=['input_ids', 'labels', 'attention_mask'])
        # Assign Test split(s) for use in Dataloaders
        if stage in (None, "test"):
            if os.path.isdir(test_data_dir) and not self.force_process:
                print("READ", test_data_dir)
                self.test_dataset = Dataset.load_from_disk(test_data_dir)
            else:
                print("Processing", test_data_dir)
                self.test_dataset = self._process_dataset_split(self.dataset['test'], self.tokenizer) 
                if self.save_cache:
                    self.test_dataset.save_to_disk(test_data_dir)
            # self.test_dataset.set_format('torch', columns=['input_ids', 'labels'])
    
    def train_dataloader(self):
        self.train_dataset.set_format('torch', columns=['input_ids', 'labels', 'attention_mask'])
        return DataLoader(self.train_dataset, shuffle=True, batch_size=self.batch_size, num_workers=8)
    
    def val_dataloader(self):
        self.val_dataset.set_format('torch', columns=['input_ids', 'labels', 'attention_mask'])
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=8)
    
    def test_dataloader(self):
        self.test_dataset.set_format('torch', columns=['input_ids', 'labels'])
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=8)


class KALESGD_DataModule(pl.LightningDataModule):
    def __init__(self, batch_size=4, tokenizer_path='t5-small', tokenizer_class='T5Tokenizer', \
                    force_process=False, save_cache=True, linearizer_class='NONE', dataset_path='NONE', encode_args=None):
        '''
        linearizer_class = [naive | schema_guided | t2g2]
        '''
        super().__init__()
        self.batch_size = batch_size
        self.tokenizer_class = tokenizer_class
        self.tokenizer = get_class_object(transformers, self.tokenizer_class).from_pretrained(self.tokenizer_path)
        self.linearizer_class=linearizer_class
        self.dataset_path = dataset_path
        self.encode_args = encode_args
        self.force_process = force_process
        self.save_cache = save_cache
    
    def prepare_data(self):
        file_name = f"{self.linearizer_class}_all.tsv"
        self.train_path = os.path.join(self.dataset_path, 'train', file_name)
        self.val_path = os.path.join(self.dataset_path, 'dev', file_name)
        self.test_path = os.path.join(self.dataset_path, 'test', file_name)
        assert os.path.isfile(self.train_path)
        assert os.path.isfile(self.val_path)
        assert os.path.isfile(self.test_path)
    
    def _process_dataset_split(self, dataset, tokenizer):
        dataset = dataset \
                    .map(tokenize_function(self.tokenizer, 'target', **self.encode_args), batched=True, load_from_cache_file=False)\
                    .rename_column('input_ids', 'labels')\
                    .map(tokenize_function(self.tokenizer, 'input', **self.encode_args), batched=True, load_from_cache_file=False)
        return dataset

    def setup(self, stage: Optional[str] = None):
        self.train_df = pd.read_csv(os.path.join(self.train_path), sep='\t', names=['input', 'target', 'metainfo', 'dialog_id', 'turn_id'])
        self.val_df = pd.read_csv(os.path.join(self.val_path), sep='\t', names=['input', 'target', 'metainfo', 'dialog_id', 'turn_id'])
        self.test_df = pd.read_csv(os.path.join(self.test_path), sep='\t', names=['input', 'target', 'metainfo', 'dialog_id', 'turn_id'])
        dataset_prefix = os.path.join('data', 'KALESGD_'+self.tokenizer_class)
        train_data_dir = dataset_prefix+self.linearizer_class+'_train'
        val_data_dir = dataset_prefix+self.linearizer_class+'_val'
        test_data_dir = dataset_prefix+self.linearizer_class+'_test'
        if stage in (None, "fit"):
            if os.path.isdir(train_data_dir) and not self.force_process:
                print("Read", train_data_dir)
                self.train_dataset = Dataset.load_from_disk(train_data_dir)
            else:
                print("Processing", train_data_dir)
                self.train_tb = Table.from_pandas(self.train_df[['input', 'target']])
                self.train_dataset = Dataset(self.train_tb)
                self.train_dataset = self._process_dataset_split(self.train_dataset, self.tokenizer)
                if self.save_cache:
                    self.train_dataset.save_to_disk(train_data_dir)

            if os.path.isdir(val_data_dir) and not self.force_process:
                print("Read", val_data_dir)
                self.val_dataset = Dataset.load_from_disk(val_data_dir) 
            else:
                print("Processing", val_data_dir)
                self.val_tb = Table.from_pandas(self.val_df[['input', 'target']])
                self.val_dataset = Dataset(self.val_tb)
                self.val_dataset = self._process_dataset_split(self.val_dataset, self.tokenizer)
                if self.save_cache:
                    self.val_dataset.save_to_disk(val_data_dir)

            self.train_dataset.set_format('torch',columns=['input_ids', 'labels'])
            self.val_dataset.set_format('torch', columns=['input_ids', 'labels'])
        if stage in (None, "test"):
            if os.path.isdir(test_data_dir) and not self.force_process:
                print("Read", test_data_dir)
                self.test_dataset = Dataset.load_from_disk(test_data_dir)
            else:
                self.test_tb = Table.from_pandas(self.test_df[['input', 'target']])
                self.test_dataset = Dataset(self.test_tb)
                self.test_dataset = self._process_dataset_split(self.test_dataset, self.tokenizer)
                if self.save_cache:
                    self.test_dataset.save_to_disk(test_data_dir)

            self.test_dataset.set_format('torch', columns=['input_ids', 'labels'])
        
    def train_dataloader(self):
        return DataLoader(self.train_dataset, shuffle=True, batch_size=self.batch_size)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)


class GEMSGD_GPT2DataModule(GEMSGD_DataModule):
    def __init__(self, batch_size=4, tokenizer=None, tokenizer_name='gpt2', \
                    force_process=False, save_cache=True, encode_args={}, linearizer_class='NONE', schema_paths=[], template_dir=None):
        super().__init__(
            batch_size=batch_size,
            tokenizer=tokenizer,
            tokenizer_name=tokenizer_name,
            force_process=force_process,
            save_cache=save_cache,
            encode_args=encode_args,
            linearizer_class=linearizer_class,
            schema_paths=schema_paths,
            template_dir=template_dir
        )
                
    def _process_dataset_split(self, dataset, tokenizer):
        # linearized_dataset = dataset.map(linearize_with_fn('dialog_acts', '_linearized', self.naive_linearize))
        linearized_dataset = dataset.map(self.linearizer, load_from_cache_file=False)
        print(linearized_dataset[:20])

        def _pad(seq, pad_token, pad_to=1024):
            return seq + (pad_to-len(seq)) * [pad_token]
        def _gpt_input_label_fn(example):
            bos_token_id, eos_token_id, pad_token_id = tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.pad_token_id
            example['input_ids'] = [input_id for input_id in example['input_ids'] if input_id != pad_token_id]
            example['labels'] = [label for label in example['labels'] if label != pad_token_id]
            gpt_input_ids = example['input_ids'] + [bos_token_id] + example['labels'] + [eos_token_id]
            gpt_labels = (len(example ['input_ids'])+1)*[-100] + example['labels'] + [eos_token_id]
            attention_mask = len(gpt_input_ids)*[1]
            raw_input_mask = (len(example['input_ids'])+1) * [1]
            raw_label_mask = [0 if elem == -100 else 1 for elem in gpt_labels]
            if self.encode_args.get('padding', "") in {'max_length', 'longest'}:
                gpt_input_ids = _pad(gpt_input_ids, tokenizer.pad_token_id)
                gpt_labels = _pad(gpt_labels, -100)
                attention_mask = _pad(attention_mask, 0)
                raw_input_mask = _pad(raw_input_mask, 0)
                raw_label_mask = _pad(raw_label_mask, 0)
            return {
                'input_ids': gpt_input_ids, 
                'labels': gpt_labels,
                'attention_mask': attention_mask,
                'raw_input_mask': raw_input_mask,
                'raw_label_mask': raw_label_mask
            }

        tokenized_dataset = linearized_dataset \
                        .map(tokenize_function(tokenizer, 'target', **self.encode_args), batched=True, load_from_cache_file=False) \
                        .rename_column('input_ids','labels')\
                        .rename_column('attention_mask', 'output_mask')\
                        .map(tokenize_function(tokenizer, '_linearized', **self.encode_args), batched=True, load_from_cache_file=False)\
                        .map(lambda example: {'dialog_acts': [{k: self.act_id2name_map[v] if k=='act' else v for k,v in action.items()} for action in example['dialog_acts']]})\
                        .map(_gpt_input_label_fn, load_from_cache_file=False)
        res_dataset = tokenized_dataset
        return res_dataset
    
    def test_dataloader(self):
        self.test_dataset.set_format('torch', columns=['input_ids', 'labels', 'raw_input_mask', 'raw_label_mask'])
        # return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=8)
        return DataLoader(self.test_dataset, batch_size=self.batch_size)