import pandas as pd
import argparse
from generate import generate_result_cot

# Setup command-line argument parsing
parser = argparse.ArgumentParser(description='Run generate_cot with input parameters.')
parser.add_argument('--model_path', type=str, required=True, help='Model path')
parser.add_argument('--options', type=str, nargs='+', required=True, help='Options for answers')
parser.add_argument('--language', type=str, required=True, help='Language of the dataset')
parser.add_argument('--iteration', type=str, required=True, help='nth Iteration')
parser.add_argument('--bs', type=int, required=True, help='Batch size')
parser.add_argument('--device', type=str, required=True, help='Device type')

args = parser.parse_args()

save_path = args.model_path.split('/')[1]
df = pd.read_csv(f'results/{save_path}-{args.language}-{args.iteration}-cot.csv')

df = generate_result_cot(args.model_path, df, args.options, args.bs, args.device)
df.to_csv(f'results/{save_path}-{args.language}-{args.iteration}-cot.csv',index=False)