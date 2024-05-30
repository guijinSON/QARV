import pandas as pd
import argparse
from datasets import load_dataset
from generate import generate_direct_result

# Setup command-line argument parsing
parser = argparse.ArgumentParser(description='Run generate_cot with input parameters.')
parser.add_argument('--model_path', type=str, required=True, help='Model path')
parser.add_argument('--dataset', type=str, required=True, help='Dataset identifier')
parser.add_argument('--options', type=str, nargs='+', required=True, help='Options for answers')
parser.add_argument('--last_option', type=str, required=True, help='Last option phrase')
parser.add_argument('--language', type=str, required=True, help='Language of the dataset')
parser.add_argument('--bs', type=int, required=True, help='Batch size')
parser.add_argument('--device', type=str, required=True, help='Device type')

args = parser.parse_args()

# Load dataset and prepare data frame
df = pd.DataFrame(load_dataset(args.dataset, args.language)['test'])

# Call the function
output = generate_direct_result(args.model_path, df, args.last_option, args.options, args.bs, args.device)

save_path = args.model_path.split('/')[1]
pd.DataFrame(output).to_csv(f'results/{save_path}-{args.language}-mc.csv',index=False)
