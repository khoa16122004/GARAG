import random
from src.option import Options
from src.util import init_logger, timestr
from src.task import ReaderDataset, evaluate
from src.attacker import build_attack
from textattack.augmentation import Augmenter
from textattack.attack_args import AttackArgs
from utils import set_seed_everything
import tqdm
import os
import json
import logging

from textattack.metrics.quality_metrics import Perplexity, USEMetric
from textattack.shared import AttackedText, utils

logger = logging.getLogger(__name__)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def main():
    set_seed_everything(22520691)
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
    
    
    print("Attacker and dataset done")


    result = attack.attack_dataset(dataset)
    

    logger.info("Attack finished")
    evaluate(result)
    if opt.is_save:
        # data_dir = os.path.join(os.path.split(opt.data_dir)[0], "noise", "g_p_{}_seq_{}".format(opt.perturbation_level, opt.transformations_per_example))
        # os.makedirs(data_dir, exist_ok=True)
        with open(os.path.join(opt.output_dir, "{}.json".format(opt.method)), 'w') as f: json.dump(result,f)
    

if __name__=="__main__":
    main()