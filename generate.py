from vllm import LLM, SamplingParams
from transformers import AutoTokenizer, AutoModelForCausalLM
from src import initialize_vllm_model, compile_prompt, get_next_token_batch, prompts
import pandas as pd


def generate_cot(model_path, df, last_option, options, bs=32, device='cuda'):
    llm = initialize_vllm_model(model_path)
    
    rows = []
    for prompt in prompts:
        qry = [(prompt, compile_prompt(prompt,row.q,[row.us, row.ko], True ,last_option)) for _,row in df.iterrows()]
        rows.extend(qry)
        
    df = pd.DataFrame(rows,columns=['prompt','query'])
    
    outputs = llm.generate(
            df['query'].values, 
            SamplingParams(temperature=0.8,top_p=0.95,min_tokens=20,max_tokens=1024)
    )
    
    df['generated_output'] = [output.outputs[0].text for output in outputs]
    return df

def generate_result_cot(model_path, df, options, bs, device):
    qrys = df["query"]
    outputs = df["generated_output"]
    qrys_outputs = [f"{qry} {gen}" for qry,gen in zip(qrys,outputs)]

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")
    
    final_answers = get_next_token_batch(model, tokenizer, qrys_outputs, options, bs, device)
    
    df['final_answers'] = final_answers
    return df

def generate_direct_result(model_path, df, last_option, options, bs=32, device='cuda'):
    rows = []
    for prompt in prompts:
        qry = [(prompt, compile_prompt(prompt,row.q,[row.us, row.ko], True ,last_option)) for _,row in df.iterrows()]
        rows.extend(qry)
        
    df = pd.DataFrame(rows,columns=['prompt','query'])

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")
    
    final_answers = get_next_token_batch(model, tokenizer, df['query'].values, options, bs, device)

    df["final_answer"] = final_answers
    return df
