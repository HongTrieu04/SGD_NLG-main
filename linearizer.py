from abc import ABC
from abc import abstractmethod
import json
from google_nlg.utterance_generator import TemplateUtteranceGenerator

class LinearizerBase(ABC):
    def __init__(self, input_field, output_field):
        self.input_field = input_field
        self.output_field = output_field

    @abstractmethod
    def tokenize_from_acts(self, dialogue_act_list) -> str:
        return ""

    def __call__(self, example):
        return {self.output_field: self.tokenize_from_acts(example[self.input_field])}

class SGD_NaiveLinearizer(LinearizerBase):
    def __init__(self, act_idx_name_map, **kwargs):
        super().__init__("dialog_acts", '_linearized')
        self.act_idx_name_map = act_idx_name_map
    
    def tokenize_from_acts(self, dialogue_act_list) -> str:
        res = ''
        for dialog_act in dialogue_act_list:
            # res += self.act_idx_name_map[dialog_act['act']] + ' ' + dialog_act['slot'] +  ' ' + ' '.join(dialog_act['values']) + ' <SEP> '
            res += f"{self.act_idx_name_map[dialog_act['act']]} {dialog_act['slot']} {' '.join(dialog_act['values'])} <SEP> "
        return res[:-7] # remove trailing <SEP>

class SGD_PaperNaiveLinearizer(LinearizerBase):
    def __init__(self, act_idx_name_map, **kwargs):
        super().__init__("dialog_acts", '_linearized')
        self.act_idx_name_map = act_idx_name_map

    def tokenize_from_acts(self, dialogue_act_list) -> str:
        res = ''
        for dialog_act in dialogue_act_list:
            if dialog_act['values']:
                # res += self.act_idx_name_map[dialog_act['act']] + ' ( ' + dialog_act['slot'] + ' = ' + ' '.join(dialog_act['values']) +  ' ) ' + ' '
                res += f"{self.act_idx_name_map[dialog_act['act']]} ( {dialog_act['slot']} = {' '.join(dialog_act['values'])} ) "
            elif dialog_act['slot']:
                # res += self.act_idx_name_map[dialog_act['act']] + ' ( ' + dialog_act['slot'] + ' ) ' 
                res += f"{self.act_idx_name_map[dialog_act['act']]} ( {dialog_act['slot']} ) "
            else:
                res += f"{self.act_idx_name_map[dialog_act['act']]} "

        return res[:-1] # remove trailing <SEP>

class SGD_SchemaGuidedLinearizer(LinearizerBase):
    def __init__(self, act_idx_name_map, schema_paths):
        super().__init__("dialog_acts", '_linearized')
        self.act_idx_name_map = act_idx_name_map
        schema_arr = []
        for schema_file in schema_paths:
            with open(schema_file, 'r') as f:
                schema_arr.extend(json.loads(f.read()))

        self.service_desc_dict = {}
        self.schema_slots_desc_dict = {}
        for schema in schema_arr:
            service_name = schema['service_name']
            self.service_desc_dict[service_name] = schema['description']
            self.schema_slots_desc_dict[service_name] = {}
            for slot in schema['slots']:
                self.schema_slots_desc_dict[service_name][slot['name']] = slot['description']

    def tokenize_from_acts(self, dialogue_act_list, service) -> str:
        res = ''
        for dialog_act in dialogue_act_list:
            act_name = self.act_idx_name_map[dialog_act['act']]
            if not service in self.schema_slots_desc_dict:
                print("service not in schema", service, dialog_act['slot'], dialog_act)
                continue
            if not dialog_act['slot'] in self.schema_slots_desc_dict[service] and not dialog_act['slot'] in {'count', 'intent', ''}:
                print("ERROR! slot doesn't exist in schema", service, dialog_act['slot'])
                continue
            slot_desc = self.schema_slots_desc_dict[service][dialog_act['slot']] if not dialog_act['slot'] in {'', 'count', 'intent'} else dialog_act['slot'] # if slot = '' or 'count' or 'intent', keep it as it is
            if dialog_act['values']:
                res += f"{act_name} ( {slot_desc} = {' '.join(dialog_act['values'])} ) "
            elif dialog_act['slot']:
                res += f"{act_name} ( {slot_desc} ) "
            else:
                res += f"{act_name} "
        return res[:-1]

    
    # override __call__ as we need additional information
    def __call__(self, example):
        return {self.output_field: self.tokenize_from_acts(example[self.input_field], example['service'])}

class SGD_SchemaGuidedWithServiceLinearizer(SGD_SchemaGuidedLinearizer):
    def __init__(self, act_idx_name_map, schema_paths):
        super().__init__(act_idx_name_map, schema_paths)
    
    def tokenize_from_acts(self, dialog_act_list, service) -> str:
        res = super().tokenize_from_acts(dialog_act_list, service)
        res = f"SERVICE ( {service} = {self.service_desc_dict[service]} ) " + res
        return res

class SGD_TemplateGuidedLinearizer(LinearizerBase):
    def __init__(self, act_idx_name_map, template_dir):
        super().__init__("dialog_acts", '_linearized')
        self.act_idx_name_map = act_idx_name_map
        self.utter_generator = TemplateUtteranceGenerator(template_dir)

    def tokenize_from_acts(self, dialog_act_list):
        pass # not suitable for template guided, use tokenize_turn to reuse Google's code
    
    def tokenize_turn(self, turn) -> str:
        utterance = self.utter_generator.get_robot_utterance(turn, None)
        return utterance
    
    # override __call__ as we need additional information
    def __call__(self, example):
        example['actions'] = example[self.input_field]
        for action in example['actions']:
            action['act'] = self.act_idx_name_map[action['act']]
        turn = {'frames': [example]}
        return {self.output_field: self.tokenize_turn(turn)}
 