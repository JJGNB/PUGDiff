import warnings
warnings.filterwarnings("ignore")
import argparse
from omegaconf import OmegaConf
from sampler_psr import SamplerPSR

def get_parser(**parser_kwargs):
    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument("--bs", type=int, default=1, help="Batchsize for loading image")
    parser.add_argument(
        "--cfg_path", type=str, default="./configs/sample-sd-turbo.yaml", help="Configuration path.",
    )
    args = parser.parse_args()

    return args

def get_configs(args):
    configs = OmegaConf.load(args.cfg_path)
    return configs

def main():
    args = get_parser()
    configs = get_configs(args)
    sampler = SamplerPSR(configs)
    sampler.inference(configs, bs=args.bs)

if __name__ == '__main__':
    main()
