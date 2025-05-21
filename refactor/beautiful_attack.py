import os
import json
import logging
import random
import numpy as np
from src.option import Options
from src.util import init_logger, timestr
from src.task import ReaderDataset, evaluate
from src.attacker import build_attack

def set_seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass

os.environ["TOKENIZERS_PARALLELISM"] = "false"
logger = logging.getLogger(__name__)

def main():
    set_seed_everything(22520691)
    option = Options("attack")
    opt, message = option.parse(timestr())
    logger = init_logger(opt)
    logger.info(message)
    logger.info(f"The name of experiment is {opt.name}")
    logger.info(f"Attack type is {opt.method}")

    dataset = ReaderDataset(opt)
    attack, dataset = build_attack(opt, dataset)
    print("Attacker and dataset done")

    result = attack.attack_dataset(dataset)
    logger.info("Attack finished")
    evaluate(result)
    if getattr(opt, 'is_save', False):
        output_path = os.path.join(opt.output_dir, f"{opt.method}.json")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        logger.info(f"Results saved to {output_path}")

if __name__ == "__main__":
    main()
