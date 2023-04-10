from lib import *

#from tool.finetuning import TrainSynth90k
from tool.base import TrainSynth90k
from utils.model.crnn import CRNN

parser = argparse.ArgumentParser(description='CRNN Implementation')
parser.add_argument('--yaml_config', type=str, default='configs/config.yaml')
parser.add_argument('--phase', type=str, default='train', help='Phase choice= {train, test}')    
args = parser.parse_args()

# config
cfg = yaml.load(open(args.yaml_config,'r'), Loader=yaml.FullLoader)


def main():
    if args.phase == 'train':
        trainmodule = TrainSynth90k(CRNN)
        trainmodule.train_network(cfg)

if __name__ == "__main__":
    main()
