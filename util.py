import yaml
import os.path
import inspect
import transformers

def load_config(config_path):
    if not os.path.isfile(config_path):
        print(config_path)
        raise FileNotFoundError(f"config file not found at {config_path}")
    stream = open(config_path, 'r')
    return yaml.safe_load(stream)

def get_class_object(module, classname):
    # print(classname)
    for name, obj in inspect.getmembers(module):
        # print(name)
        if name == classname:
            return obj
    raise ValueError(f"{classname} not found")

def get_tokenizer(config):
    tokenizer_dict = config['TokenizerInfo']
    tokenizer_name = tokenizer_dict['tokenizer_name']
    tokenizer_class = tokenizer_dict['tokenizer_class']
    tokenizer_class_obj = get_class_object(transformers, tokenizer_class)
    tokenizer = tokenizer_class_obj.from_pretrained(tokenizer_name)
    special_tokens = {}
    if tokenizer_class[:4] == 'GPT2':
        # special_tokens.update({'pad_token': '[PAD]'})
        tokenizer.pad_token = '[PAD]'
        # tokenizer.pad_token_id = tokenizer.eos_token_id
    special_tokens.update(tokenizer_dict.get('special_tokens', {}))
    tokenizer.add_special_tokens(special_tokens)
    return tokenizer

def prefix_root_dir(root_dir, rel_path):
    return os.path.join(root_dir, rel_path)
