from typing import List
import json
from data_utils.data_reader import Dataset

class dawonDataset(Dataset):
    def __init__(self, data_path: str, data_type:str, max_seq_length=512, split='train'):
        if data_type == 'dawon':
            if split == 'train' or split == 'dev':
                data_path = f'{data_path}/dawon.jsonl'
            else:
                assert False

            with open(data_path) as f:
                data = []
                for dic in f:
                    data.append(json.loads(dic))    
                
            self.process_dawon(data)

            if split == 'train':
                num_data = len(self.examples)
                self.examples = self.examples[:-num_data//10]
            elif split == 'dev':
                num_data = len(self.examples)
                self.examples = self.examples[-num_data//10:]
                print(num_data, len(self.examples))
        

        self.split = split
    
    def process_dawon(self, data: List[dict]):
        self.examples = []
        for dic in data:
            
            ins = dic['instruction']
            input = dic['input']
            output = dic['output']
            self.examples.append({
                'instruction': ins,
                'input': input,
                'output': output,    
            })

    

