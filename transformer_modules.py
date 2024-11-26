import pytorch_lightning as pl
import torch
import transformers
from transformers import T5Tokenizer, GPT2Tokenizer, GPT2Config, T5Config
from transformers import AutoConfig, T5ForConditionalGeneration, GPT2LMHeadModel
from transformers import AdamW, Adafactor, get_scheduler
from torch.optim import Adam
from datasets import load_metric
from datetime import datetime
import pandas as pd
import os.path
from util import get_class_object
from prefix_models import (
    PrefixLMModelConfig,
)
import prefix_models

class HFGenerationModel(pl.LightningModule):
    def __init__(self, model_class='T5ForConditionalGeneration', model_path='t5-small', tokenizer=None, optimizer='AdamW', \
            optimizer_params=None, decode_path='default_test/', save_decode=False, generate_params=None, **kwargs):
        super().__init__()
        if 'prefix_config' in kwargs and kwargs['prefix_config'] is not None:
            model_class_object = get_class_object(prefix_models, model_class)
            if os.path.isdir(model_path):
                self.model = model_class_object.from_pretrained(model_path)
            else:
                self.model = model_class_object(kwargs.pop('prefix_config'))
        else:
            model_class_object = get_class_object(transformers, model_class)
            self.model = model_class_object.from_pretrained(model_path)
        if tokenizer is None:
            raise ValueError("Tokenizer must not be None")
        self.tokenizer = tokenizer
        # self.model.resize_token_embeddings(len(self.tokenizer)) # resize so that new tokens have corresponding embeddings
        self.decode_path = decode_path
        self.optimizer = optimizer
        self.save_decode = save_decode
        self.test_cnt = 0
        self.optimizer_params = optimizer_params
        self.generate_params = generate_params
        if self.save_decode:
            self.log_file = open(os.path.join(decode_path, 'test_case.txt'), 'w')
        self.save_hyperparameters()
        

    def forward(self, x):
        return self.model.generate(x, **self.generate_params)
    
    def training_step(self, batch, batch_idx):
        x, y, mask = batch['input_ids'], batch['labels'], batch['attention_mask']
        outputs = self.model(input_ids=x, labels=y, attention_mask=mask)
        loss = outputs.loss
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, mask = batch['input_ids'], batch['labels'], batch['attention_mask']
        outputs = self.model(input_ids=x, labels=y, attention_mask=mask)
        loss = outputs.loss
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        # print(f'{__class__.__name__} is on cuda:', next(self.model.parameters()).is_cuda)
        x, y = batch['input_ids'], batch['labels']
        outputs = self.forward(x)
        refs = self.tokenizer.batch_decode(y, skip_special_tokens=True)
        preds = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        inps = self.tokenizer.batch_decode(x, skip_special_tokens=True)

        if self.save_decode:
            for r, p, inp in zip(refs, preds, inps):
                self.print(str(self.test_cnt)+'|', end='', file=self.log_file)
                self.print(r.replace('|','<vstripe>'), p.replace('|',"<vstripe>"), inp.replace('|','<vstripe>'), sep=' | ', file=self.log_file)
                self.test_cnt += 1

    def test_epoch_end(self, outputs):
        pass
    
    def configure_optimizers(self):
        if self.optimizer == 'AdamW':
            optimizer = AdamW(self.parameters(), **self.optimizer_params)
        elif self.optimizer == 'Adafactor':
            optimizer = Adafactor(self.parameters(), **self.optimizer_params)
        else:
            raise ValueError(f"Invaild optimizer name: {self.optimizer}")
        return optimizer



    def test_epoch_end(self, outputs):
        pass
        # rouge_res = self.rouge.compute()
        # bleu_res = self.bleu.compute()
        # for rtype, rvalue in rouge_res.items():
        #     print(rtype, rvalue)
        #     self.log(rtype+'-prec', rvalue.mid.precision)
        #     self.log(rtype+'-recall', rvalue.mid.recall)
        #     self.log(rtype+'-fvalue', rvalue.mid.fmeasure)
        # # self.log('bleu_score', bleu_res['score'])
        # if self.save_decode:
        #     test_df = pd.DataFrame(self.test_dict)
        #     test_df.to_csv(os.path.join(self.decode_path, 'test_case'))
    
    def configure_optimizers(self):
        if self.optimizer == 'AdamW':
            optimizer = AdamW(self.parameters(), **self.optimizer_params)
        elif self.optimizer == 'Adafactor':
            optimizer = Adafactor(self.parameters(), **self.optimizer_params)
        elif self.optimizer == 'Adam':
            optimizer = Adam(self.parameters(), **self.optimizer_params)
        else:
            raise ValueError(f"Invaild optimizer name: {self.optimizer}")
        return optimizer

class GPT2GenerationModel(HFGenerationModel):
    def __init__(self, model_class='GPT2LMHeadModel', model_path='gpt2', tokenizer=None, optimizer='AdamW', \
            optimizer_params=None, decode_path='default_test/', save_decode=False, generate_params=None, **kwargs):
        super().__init__(
            model_class=model_class,
            model_path=model_path,
            tokenizer=tokenizer,
            optimizer=optimizer,
            optimizer_params=optimizer_params,
            decode_path=decode_path,
            save_decode=save_decode,
            generate_params=generate_params,
            **kwargs
        )
    
    def test_step(self, batch, batch_idx):
        x, y, input_mask, label_mask = batch['input_ids'], batch['labels'], batch['raw_input_mask'], batch['raw_label_mask']
        refs = []
        preds = []
        inps = []
        for ind in range(x.shape[0]):
            xm = torch.masked_select(x[ind], input_mask[ind].type(torch.bool))
            ym = torch.masked_select(y[ind], label_mask[ind].type(torch.bool))
            output = self.forward(xm[None,:])
            for out in output:
                ref = self.tokenizer.decode(ym).split('<|endoftext|>')[0]
                pred = self.tokenizer.decode(out)
                inp = pred.split('<|endoftext|>')[0]
                pred = pred.split('<|endoftext|>')[1]
                refs.append(ref)
                preds.append(pred)
                inps.append(inp)
        if self.save_decode:
            for r, p, inp in zip(refs, preds, inps):
                self.print(str(self.test_cnt)+'|', end='', file=self.log_file)
                self.print(r.replace('|','<vstripe>'), p.replace('|',"<vstripe>"), inp.replace('|','<vstripe>'), sep=' | ', file=self.log_file)
                self.test_cnt += 1
        
class PrefixGPT2GenerationModel(GPT2GenerationModel):
    def __init__(
        self,
        model_class='PT_GPT2Model',
        model_path='new', 
        tokenizer=None, 
        optimizer='AdamW',
        optimizer_params=None,
        decode_path='default_test/', 
        save_decode=False, 
        generate_params=None,
        prefix_model_name='MLPPrefixModel',
        gpt2_model_name='gpt2',
        prefix_length=10,
        hidden_dims=None,
        activation_fn_name='Tanh',
        prefix_dropout=0.0,
        **kwargs
        ):
        prefix_config = PrefixLMModelConfig(
            lm_model_name=gpt2_model_name,
            prefix_model_name=prefix_model_name,
            prefix_len=prefix_length,
            hidden_dims=hidden_dims,
            activation_fn_name=activation_fn_name,
            is_encoder_decoder=False,
            prefix_dropout=prefix_dropout
        )
        super().__init__(
            model_class=model_class,
            model_path=model_path,
            tokenizer=tokenizer,
            optimizer=optimizer,
            optimizer_params=optimizer_params,
            decode_path=decode_path,
            save_decode=save_decode,
            generate_params=generate_params,
            prefix_config=prefix_config,
            **kwargs
        )
        # super().__init__(
        #     model_class='GPT2LMHeadModel',
        #     model_path='gpt2',
        #     tokenizer=tokenizer,
        #     optimizer=optimizer,
        #     optimizer_params=optimizer_params,
        #     decode_path=decode_path,
        #     save_decode=save_decode,
        #     generate_params=generate_params,
        #     **kwargs
        # )
    
    def test_step(self, batch, batch_idx):
        # self.model.set_eval_mode(on=True)
        super().test_step(batch, batch_idx)
        
class PrefixT5GenerationModel(HFGenerationModel):
    def __init__(
        self,
        model_class='PT_T5Model',
        model_path='new', 
        tokenizer=None, 
        optimizer='AdamW',
        optimizer_params=None,
        decode_path='default_test/', 
        save_decode=False, 
        generate_params=None,
        prefix_model_name='MLPPrefixModel',
        t5_model_name='t5-small',
        prefix_length=10,
        hidden_dims=None,
        activation_fn_name='Tanh',
        prefix_dropout=0.0,
        **kwargs
        ):
        prefix_config = PrefixLMModelConfig(
            lm_model_name=t5_model_name,
            prefix_model_name=prefix_model_name,
            prefix_len=prefix_length,
            hidden_dims=hidden_dims,
            activation_fn_name=activation_fn_name,
            is_encoder_decoder=True,
            prefix_dropout=prefix_dropout
        )
        super().__init__(
            model_class=model_class,
            model_path=model_path,
            tokenizer=tokenizer,
            optimizer=optimizer,
            optimizer_params=optimizer_params,
            decode_path=decode_path,
            save_decode=save_decode,
            generate_params=generate_params,
            prefix_config=prefix_config,
            **kwargs
        )
