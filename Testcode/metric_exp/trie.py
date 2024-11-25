import heapq
import pandas as pd
from datasets import load_dataset, load_from_disk, Dataset
import numpy as np
from tqdm import tqdm

class TrieNode:
    def __init__(self, token, count, pattern):
        self.children = {}
        self.token = token
        self.pattern = pattern
        self.count = count
        self.rank_metric = self.count # Rank by count by default
        self.is_leaf = True
    
    def __str__(self):
        return f"{' '.join(self.pattern)}, count={self.count}, rank_metric={self.rank_metric}"

class Trie:
    def __init__(self):
        self.root = TrieNode("", 0, [])
        self.active_nodes = [self.root]
        self.active_patterns = {""}

    def feed(self, token):
        for a_node in self.active_nodes:
            if token == a_node.token:
                continue
            if a_node.is_leaf:
                if token in a_node.children:
                    child = a_node.children[token]
                    child.count += 1
                else:
                    child = TrieNode(token, 1, a_node.pattern + [token])
                    a_node.children[child.token] = child
            else:
                if token in a_node.children:
                    child = a_node.children[token]
                    if " ".join(child.pattern) not in self.active_patterns:
                        self.active_nodes.append(child)
                        self.active_patterns.add(" ".join(child.pattern))
    
    def reset_active_nodes(self):
        self.active_nodes = [self.root]
        self.active_patterns = {""}
    
    def prune(self, min_count=5, top_k=5, rank_func=lambda x:x.count):
        queue = [self.root]
        leaves = []
        while len(queue):
            cur = queue.pop(0)
            if cur.is_leaf:
                leaves.append(cur)
                cur.is_leaf = False
            else:
                queue.extend([tnode for _, tnode in cur.children.items()])
        for leaf in leaves:
            leaf.children = {tnode for token, tnode in leaf.children.items() if tnode.count > min_count}
            for tnode in leaf.children:
                tnode.rank_metric = rank_func(tnode)
            leaf.children = {tnode.token : tnode for tnode in heapq.nlargest(top_k, leaf.children, key=lambda tnode: tnode.rank_metric) if tnode.rank_metric > 0}
    
    def get_all_patterns(self):
        queue = [self.root]
        all_patterns = []
        while len(queue):
            cur = queue.pop(0)
            all_patterns.append((" ".join(cur.pattern), cur.count, cur.rank_metric))
            queue.extend([tnode for _, tnode in cur.children.items()])
        return all_patterns

    def get_current_patterns(self):
        return [(" ".join(tnode.pattern), tnode.count) for tnode in self.active_nodes]


class SequenceExtractor:
    def __init__(self, dataset):
        self.dataset = dataset
        self.unigram_count = {}
        self.slot_count = {}
        self.slot_value_set_map = {}

        # populating global statistics in one-pass
        for example in tqdm(dataset):
            for token in example['target'].split(' '):
                self.unigram_count[token] = self.unigram_count[token]+1 if token in self.unigram_count else 1
            for action in example['dialog_acts']:
                slot = action['slot'] 
                self.slot_count[slot] = self.slot_count[slot]+1 if slot in self.slot_count else 1
                for value in action['values']:
                    for vv in value.split(" "):
                        if slot not in self.slot_value_set_map:
                            self.slot_value_set_map[slot] = set()
                        self.slot_value_set_map[slot].add(vv)

        self.total_cnt = sum(self.unigram_count.values())
        self.unigram_freq = {k: v / self.total_cnt for k, v in self.unigram_count.items()}
        self.slot_freq = {k: v / len(self.dataset) for k, v in self.slot_count.items()}
    
    def get_patterns_with_slot(self, slot_name, n_grams=4, min_count=10, top_k=5):
        test_examples = []
        for example in tqdm(self.dataset):
            for action in example['dialog_acts']:
                if action['slot'] == slot_name:
                    test_examples.append(example['target'])

        def _rank_func(tnode):
            slot_prob = self.slot_freq[slot_name]
            seq_prob = 1
            for gram in tnode.pattern:
                if gram not in self.unigram_count or self.unigram_count[gram] < 20 or gram in self.slot_value_set_map[slot_name]: # guard against rare cases and slot values, which are indicative of the slot
                    return -1
                seq_prob *= self.unigram_freq[gram]
            seq_con_prob = tnode.count/len(test_examples)
            return seq_con_prob * slot_prob / (seq_prob) # Bayesian probability

        trie = Trie()
        for _ in range(n_grams):
            for example in tqdm(test_examples):
                for sentence in example.split("."):
                    for token in sentence.split(" "):
                        trie.feed(token)
                    trie.reset_active_nodes() # reset active nodes per sentence level
            trie.prune(min_count=min_count, top_k=top_k, rank_func=_rank_func)
        return trie.get_all_patterns()
    
SLOT_NAME = 'phone_number'
if __name__ == '__main__':
    test_dataset = load_dataset('gem', 'schema_guided_dialog', split='train')
    SeqExt = SequenceExtractor(test_dataset)
    slot_patterns = SeqExt.get_patterns_with_slot(SLOT_NAME)
    print(slot_patterns)

    # test_examples = []
    # for example in tqdm(test_dataset):
    #     for action in example['dialog_acts']:
    #         if action['slot'] == SLOT_NAME:
    #             test_examples.append(example['target'])
    
    # trie = Trie()
    # for n in range(4):
    #     for example in tqdm(test_examples):
    #         for sentence in example.split("."):
    #             for token in sentence.split(" "):
    #                 trie.feed(token)
    #             trie.reset_active_nodes() # reset active nodes per sentence level
    #     trie.prune(min_count=10, top_k=5)
    # print(trie.get_all_patterns())
    # print(sorted(trie.get_all_patterns(), key=lambda x: x[1]*np.exp(len(x[0].split())-1), reverse=True))