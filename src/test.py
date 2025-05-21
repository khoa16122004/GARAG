import random
from src.option import Options
from src.util import init_logger, timestr
from src.task import ReaderDataset, evaluate
from src.attacker import build_attack
from textattack.augmentation import Augmenter
from textattack.attack_args import AttackArgs

import tqdm
import os
import json
import logging

from textattack.metrics.quality_metrics import Perplexity, USEMetric
from textattack.shared import AttackedText, utils

logger = logging.getLogger(__name__)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def main():
    option = Options("attack")
    opt, message = option.parse(timestr()) # create 
    logger = init_logger(opt)
    logger.info(message)
    logger.info("The name of experiment is {}".format(opt.name))
    logger.info("Attack type is {}".format(opt.method))

    
    # is the json file
    dataset = ReaderDataset(opt) # get document dataset
    
    # return genetic algorithm and dataset
    attack, dataset = build_attack(opt, dataset)

    for i, d in enumerate(dataset):
        answers = d["answers"] # list answers     
        question = d["question"] # question
        ctxs = d["ctxs"] # contextutal đã được search   
        q_id = i # id
        texts = [ctx["context"] for ctx in ctxs]
        
        golds_preds = attack.goal_function(texts, question)
        print("Gold preds: ", golds_preds)
        
        scores = attack.goal_function.eval(texts,
                                           question,
                                           answers[0])
        print("Scores: ", scores)
        break
        # result = attack.attack_dataset(dataset)
        # print(result)
    print("Attacker and dataset done")
    

   
    

if __name__=="__main__":
    main()