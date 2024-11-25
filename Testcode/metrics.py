from abc import ABC, abstractmethod
import pandas as pd
import collections
from google_nlg.ser import get_ser_slots, example_ser
import json

class Metric(ABC):
    def __init__(self, name, *args, **kwargs):
        self.name = name
        self.records = {'ref':[], 'pred':[], 'tags': []}
        self.setup(*args, **kwargs)
    
    @abstractmethod
    def setup(self, *args, **kwargs):
        pass
    
    def __repr__(self):
        return f"Metric: {self.name}"
    
    def add_batch(self, references=None, predictions=None, tags=None):
        if references is None or predictions is None:
            raise ValueError("reference or predictions is None")
        self.records['ref'].extend(references)
        self.records['pred'].extend(predictions)
        self.records['tags'].extend(tags or [None for i in range(len(references))])

    def compute(self, reference=None, prediction=None, tag=None, full=True, avg=False):
        if reference or prediction or tag:
            return self.compute_row({'ref': reference, 'pred': prediction, 'tag': tag})
        else:
            self.record_df = pd.DataFrame(self.records)
            res = self.record_df.apply(self.compute_row, axis='columns')
            self.record_df[self.name] = res
            if full:
                return self.record_df
            elif not avg:
                return self.record_df[self.name]
            else:
                return self.record_df[self.name].mean()

    @abstractmethod
    def compute_row(self, row):
        pass

class SER_metric(Metric):
    '''
    ref: list dialog_acts
    pred: string 
    This applies to non-categorical values only

    '''
    def __init__(self, name='SER', data_dir="", contain_fn=None):
        super().__init__(name, data_dir=data_dir, contain_fn=contain_fn)
        
    def setup(self, data_dir="", contain_fn=None, *args, **kwargs):
        self.permissible_slots = get_ser_slots(data_dir)
        self.contain_fn = contain_fn or (lambda pred, v: v in pred)
    
    def compute_row(self, row):
        '''
        Compute if there is a slot value for the current record: True: error, False: no error, None: not applicable
        '''
        ref = row['ref']
        pred = row['pred'].lower()
        tags = row['tags']
        service = tags['service']
        has_permissible = False
        for action in tags['dialog_acts']:
            slot = action['slot']
            if slot not in self.permissible_slots[service]:
                continue
            values = action['values']
            for v in values:
                has_permissible = True # it could be that the slot is permissible but no value is given. We shouldn't include these bonus
                v = v.lower()
                if not self.contain_fn(pred, v):
                    return True
        return False if has_permissible else None
