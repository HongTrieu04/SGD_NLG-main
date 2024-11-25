from datasets import load_metric, load_dataset
import pandas as pd
from tabulate import tabulate
import argparse
import os.path
from tqdm import tqdm
import json
import sys
sys.path.append('..')
from google_nlg.ser import get_ser_slots, example_ser
from parent.parent import parent
import numpy as np
from metrics import SER_metric
from fuzzywuzzy import fuzz
import wandb
from util import load_config

parser = argparse.ArgumentParser()
parser.add_argument('--file', type=str, help='path to the decoded file')
parser.add_argument('--model_name', type=str, help='name of the model')
parser.add_argument('--exp_name', type=str, help='name of the experiment')
parser.add_argument('--ver_name', type=str, help='name of the version')
parser.add_argument('--wandb_project', type=str, default=None, help='name of the wandb project')

group = parser.add_argument_group('Files to compute the metric with')
# group.add_argument('--predictions', '-p', dest='predictions',
#                     help='path to predictions file')
# group.add_argument('--tables', '-t', dest='tables',
#                     help='path to tables file. Should be a json-line file, '
#                             'with one json-encoded table per line')
# group.add_argument('--references', '-r', dest='references', nargs='+',
#                     help='path to references file. In case of multiple '\
#                             'references, files should have the same number of'\
#                             ' lines, empty line means no reference.'),
# group.add_argument('--dest', dest='dest', default=None,
#                     help="write the results to this file")

group = parser.add_argument_group('Arguments regarding PARENT')
group.add_argument('--lambda_weight', dest='lambda_weight', type=float,
                    default=.5, help='Float weight to multiply table recall.')
group.add_argument('--smoothing', dest='smoothing', type=float,
                    default=.00001, help='Float value for replace zero values '
                                        'of precision and recall.')
group.add_argument('--max_order', dest='max_order', type=int,
                    default=4, help='Maximum order of the ngrams to use.')
group.add_argument('--avg_results', dest='avg_results', 
                    action='store_true', help='average results')

group = parser.add_argument_group('Arguments regarding multiprocessing')
group.add_argument('--n_jobs', dest='n_jobs', type=int, default=8,
                    help='number of processes to use. <0 for cpu_count()')
group.add_argument('--tqdm', dest='tqdm', default=True,
                    choices=['True', 'False', 'classic', 'notebook'],
                    help='Which tqdm to use.')

def has_slot_error(turn, pred, permissible_slots):
    pred = pred.lower()
    service = turn['service']
    for action in turn['dialog_acts']:
        slot = action['slot']
        if slot not in permissible_slots[service]:
            continue
        values = action['values']
        for v in values:
            v = v.lower()
            if v not in pred:
                return True
    return False

def dialog_act_to_table(dialog_acts):
    example = []
    for action in dialog_acts:
        slot = action['slot']
        if slot in {'intent'}:
            continue
        values = action['values']
        value_tokens = []
        for v in values:
            for vv in v.split():
                if vv.lower() in {'true', 'false'}:
                    continue
                value_tokens.append(vv)
        if slot and value_tokens:
            example.append(([slot],value_tokens))
    return example

if __name__ == '__main__':
    args = parser.parse_args()
    decode_file = args.file
    print("Marking", decode_file, "...")
    test_dataset = load_dataset('gem', 'schema_guided_dialog', split='test')
    df = pd.read_csv(decode_file, delimiter='|', names=['ref','pred', 'sr'], error_bad_lines=False)
    record_df = df
    record_df['dialog_acts'] = test_dataset['dialog_acts']
    record_df['domain'] = list(map(lambda x: x.split('_')[0], test_dataset['service']))
    record_df['has_slot_error'] = -1
    # record_df = record_df[~record_df['sr'].str.contains('GOODBYE')]
    record_df.loc[record_df['sr'].str.contains('GOODBYE'),'pred'] = record_df.loc[record_df['sr'].str.contains('GOODBYE'),'ref']
    # record_df['bleu'] = -1
    bleu = load_metric('sacrebleu')
    # turn_bleu = load_metric('sacrebleu')
    meteor = load_metric('meteor')
    rouge = load_metric('rouge')
    permissible_slots = get_ser_slots("data/dstc8-schema-guided-dialogue/")
    ser_metric = SER_metric(name='ser-strict', data_dir='data/dstc8-schema-guided-dialogue/')
    def fuzzy_match(pred, value):
        matched = False
        value_length = len(value.split())
        split_tokens = pred.split(" ")
        tokens = [" ".join(split_tokens[i:i+value_length]) for i in range(len(split_tokens))]
        for token in tokens:
            if fuzz.token_sort_ratio(token, value) / 100.0 > 0.9:
                matched = True
                break
        return matched
    ser_fuzzy = SER_metric(name='ser-fuzzy', data_dir='data/dstc8-schema-guided-dialogue/', \
        contain_fn=fuzzy_match)
    slot_errors = 0
    for idx, row in tqdm(record_df.iterrows()):
        ref = row['ref']
        pred = row['pred']
        # turn_bleu_score = turn_bleu.compute(references=[[ref]], predictions=[pred])
        # record_df.iloc[idx]['bleu'] = turn_bleu_score['score'] extremely slow
        test_ref = test_dataset[idx]
        bleu.add_batch(references=[[ref]], predictions=[pred])
        meteor.add_batch(references=[ref], predictions=[pred])
        rouge.add_batch(references=[ref], predictions=[pred])
        ser_metric.add_batch(references=[ref], predictions=[pred], tags=[test_ref])
        ser_fuzzy.add_batch(references=[ref], predictions=[pred], tags=[test_ref])
        # record_df.iloc[idx, -1] = has_slot_error(test_ref, pred, permissible_slots)
        # slot_errors += 1 if record_df.iloc[idx]['has_slot_error'] else 0
    res_dict = {}
    ser_res = ser_metric.compute()
    ser_fuzzy_res = ser_fuzzy.compute()
    record_df['ser-fuzzy'] = ser_fuzzy_res['ser-fuzzy']
    bleu_res = bleu.compute()
    meteor_res = meteor.compute()
    rouge_res = rouge.compute()
    # df_ser = record_df.groupby('domain').apply(lambda x: x["has_slot_error"].mean()).reset_index(name='ser').T
    # df_ser.columns = df_ser.iloc[0]
    # df_ser = df_ser.drop(['domain'])
    domain_bleu_dict = {}
    for domain in set(record_df['domain']):
        domain_record_df = record_df[record_df['domain']==domain] 
        domain_bleu_res = \
            bleu.compute(references=domain_record_df['ref'].apply(lambda x: [x]).tolist(), predictions=domain_record_df['pred'].tolist())
        domain_bleu_dict[domain] = [domain_bleu_res['score']]
    domain_bleu_df = pd.DataFrame(domain_bleu_dict)
    # domain_df = pd.concat([df_ser, domain_bleu_df])
    # parent_df['ref'] = parent_df['ref'].apply(lambda x: x.split(' '))
    # parent_df['pred'] = parent_df['pred'].apply(lambda x: x.split(' '))
    res_dict['bleu'] = [bleu_res['score']]
    # res_dict['ser'] = slot_errors / len(df)
    # res_dict['ser-fuzzy'] = ser_fuzzy_res['ser-fuzzy'].fillna(False).mean()
    res_dict['meteor'] = [meteor_res['meteor']]
    res_dict['rouge1-prec'] = [rouge_res['rouge1'].mid.precision]
    res_dict['rouge1-recall'] = [rouge_res['rouge1'].mid.recall]
    res_dict['rouge2-prec'] = [rouge_res['rouge2'].mid.precision]
    res_dict['rouge2-recall'] = [rouge_res['rouge2'].mid.recall]
    res_dict['rougeL-prec'] = [rouge_res['rougeL'].mid.precision]
    res_dict['rougeL-recall'] = [rouge_res['rougeL'].mid.recall]
    res_dict['rougeLsum-prec'] = [rouge_res['rougeL'].mid.precision]
    res_dict['rougeLsum-recall'] = [rouge_res['rougeL'].mid.recall]
    # PARENT
    parent_df = record_df.copy()
    parent_df['ref'] = parent_df['ref'].apply(lambda x: x.split())
    parent_df['pred'] = parent_df['pred'].apply(lambda x: x.split())
    parent_df['table'] = parent_df['dialog_acts'].apply(dialog_act_to_table)
    parent_df = parent_df[parent_df['ref'].map(len)>=1]
    parent_df = parent_df[parent_df['table'].map(len)>=1]
    parent_precisions, parent_recalls, parent_fscores = parent(
        parent_df['ref'].tolist(),
        parent_df['pred'].tolist(), 
        parent_df['table'].tolist(),
        lambda_weight=args.lambda_weight,
        smoothing=args.smoothing,
        max_order=args.max_order,
        avg_results=args.avg_results,
        n_jobs=args.n_jobs,
        use_tqdm=args.tqdm)
    parent_df['PARENT-precision'] = np.round(parent_precisions, 3)
    parent_df['PARENT-recall'] = np.round(parent_recalls, 3)
    parent_df['PARENT-fscore'] = np.round(parent_fscores, 3)
    res_dict['PARENT-precision'] = np.round(np.mean(parent_precisions),3)
    res_dict['PARENT-recall'] = np.round(np.mean(parent_recalls),3)
    res_dict['PARENT-fscore'] = np.round(np.mean(parent_fscores),3)
    record_df = record_df.join(parent_df[['PARENT-precision','PARENT-recall', 'PARENT-fscore']], how='left')
    record_df.fillna(-1)

    res_df = pd.DataFrame(res_dict).T
    res_df.columns = [decode_file.split('/')[1]+'---'+decode_file.split('/')[3]]
    print(tabulate(res_df, headers='keys', tablefmt='psql', showindex=True)  )

    # saving evaluation results
    save_path = os.path.dirname(decode_file)
    # res_df.to_csv(os.path.join(save_path, 'metrics.csv'))
    # record_df.to_csv(os.path.join(save_path, 'annotated_cases.csv'))
    # domain_df.to_csv(os.path.join(save_path, 'domain_metrics.csv'))

    # Parent data for examination
    parent_dir = os.path.join(save_path, 'parent_data')
    os.makedirs(parent_dir, exist_ok=True)
    parent_df[['ref']].to_csv(os.path.join(parent_dir, 'ref.txt'), index=False, header=False)   
    parent_df[['pred']].to_csv(os.path.join(parent_dir, 'pred.txt'), index=False, header=False)
    with open(os.path.join(parent_dir, 'table.jl'), 'w') as f:
        for idx, row in tqdm(parent_df[['table']].iterrows(), desc='saving table'):
            f.write(json.dumps(row['table'])) # json line file
            f.write('\n')
    # parent_df[['table']].to_csv(os.path.join(parent_dir, 'table.jl'), index=False, header=False)
    if args.wandb_project:
        config_file = decode_file.split('/')
        config_file[-1] = 'config.yaml'
        config_file = '/'.join(config_file)
        config_dict = load_config(config_file)
        metric_tb = wandb.Table(dataframe=res_df.T)
        annotated_cases_tb = wandb.Table(dataframe=record_df)
        domain_ser_tb = wandb.Table(dataframe=domain_df)
        run = wandb.init(project=args.wandb_project, name='test-'+args.exp_name, config=config_dict)
        run.log({'Eval Metric Table': metric_tb, 'Annotated Cases': annotated_cases_tb, 'Domain SER Table': domain_ser_tb})