import argparse
import json
import logging
import numpy as np
import os
import random
from os import listdir
from os.path import isfile, join
from collections import Counter, defaultdict

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from tqdm import trange

import settings
from constants import SPECIAL_TOKENS
from datareaders import get_reader


from sequentialize import get_sequence
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO
)
LOGGER = logging.getLogger(__name__)


def get_instructions(instruction_files):
    instruction_folder = settings.INSTRUCTION_FOLDER

    allinstruction_files = [f for f in listdir(instruction_folder) if isfile(join(instruction_folder, f))]
    useful_files = []
    for filename in instruction_files:
        for fname in allinstruction_files:
            if filename in fname:
                useful_files.append(fname)
    print(useful_files)

    instructions_all = []
    for fname in useful_files:
        fdata = json.load(open(join(instruction_folder, fname), 'r'))
        instructions_all.append(fdata)

    return instructions_all


def print_samples(dataset_reader, instructions_all):
    for instruction in instructions_all:
        dataset_reader.idx=0
        dp = dataset_reader.get_next()
        print(instruction)
        while dp is not None:
            print(dp)
            sequences = get_sequence(dataset_reader, dp, instruction)
            print("SEQUENCES:", sequences)
            import pdb;pdb.set_trace()
            # break

            dp = dataset_reader.get_next()

def print_examples(dataset_reader):
    dataset_reader.idx=0
    dp = dataset_reader.get_next()
    iterator_index = 0
    while dp is not None:
        print(dp)
        iterator_index+=1
        dp = dataset_reader.get_next()
        if iterator_index>10:
            break


def read_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--configfile", default='configs/config1.json', type=str)
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()

def read_examples(args):
    config = json.load(open(args.configfile, 'r'))
    LOGGER.info(config)
        # Data readers

    config['datasets'] = ['eval']
    for dataset in config['datasets']:
        if dataset not in config: continue
        datasetconfig = config[dataset]
        instruction_files = datasetconfig['instruction_files']
        if dataset=='intent-clinc':
            token_vocab_name = os.path.basename(datasetconfig['token_vocab_path']).replace(".txt", "")
            dataset_reader = dialoglue_reader.IntentDataset(settings.DIALOGUE_PATH+datasetconfig['train_data_path'],
                                        datasetconfig['max_seq_length'],
                                        token_vocab_name)
        if dataset=='slot-restaurant8k':
            token_vocab_name = os.path.basename(datasetconfig['token_vocab_path']).replace(".txt", "")
            dataset_reader = dialoglue_reader.SlotDataset(settings.DIALOGUE_PATH+datasetconfig['train_data_path'],
                                        datasetconfig['max_seq_length'],
                                        token_vocab_name)
        if dataset=='wow':
            dataset_reader = wow_reader.WoWDataset(settings.WOW_PATH+datasetconfig['train_data_path'],
                                        datasetconfig['max_seq_length'])
        instructions_all = get_instructions(instruction_files)

        print_examples(dataset_reader)
        # instructions_all = get_instructions(instruction_files)            
        # print_samples(dataset_reader, instructions_all)

def test_readers(args):
    # Data readers
    config = json.load(open(args.configfile, 'r'))
    dataset = args.dataset

    datasetconfig = config.get(dataset, None) 
    if datasetconfig is not None:
        instruction_files = datasetconfig.get('instruction_files', []) #근데 거기서 instruction_files가 없으면 아래 else문과 같이 또 []이 instruction에 저장된다.
    else: #config에 dataset의 이름에 해당하는 key가 없을때
        instruction_files = []

    dataset_reader = get_reader(args, dataset)  #리더를 갖고 온다. -> coqa데이터셋의 경우 examples로 리스트에 접근가능

    #  top, airdialogue, deal, casino
    if dataset == 'top':
        dataset_reader = dialoglue_reader.TOPDataset(settings.DIALOGUE_PATH, datasetconfig)

    if dataset == 'airdialogue':
        dataset_reader = airdialoglue_reader.AirDialogueDataset(settings.AIRDIALOGLUE_PATH,
                                                                split=datasetconfig['split'])
    if dataset == 'deal':
        dataset_reader = deal_reader.DealDataset(settings.DEAL_PATH, split=datasetconfig['split'])

    if dataset == 'casino':
        dataset_reader = casino_reader.CasinoDataset(settings.CASINO_PATH, split=datasetconfig['split'])

    if dataset == 'empathy':
        dataset_reader = empathy_reader.EmpathyDataset(settings.EMPATHY_PATH)

    if dataset=='persuasion':
        dataset_reader = persuasion_reader.PersuasionDataset(settings.PERSUASION_PATH)

    print("*"*50,"\nhere is reader!!! : \n", f"the number of dataset is {len(dataset_reader.examples)}\n")
    
    print_examples(dataset_reader) # 걍 데이터셋이 제데로 존재하는지 확인하는 python 파일인듯하다.


if __name__ == "__main__":
    args = read_args()
    print(args)
    random.seed(args.seed)

    test_readers(args)

    # to be used for testing tasks
    # read_examples(args)
