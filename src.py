import torch
from vllm import LLM, SamplingParams
import random
import string
from tqdm import tqdm
from more_itertools import chunked

prompts = [
    "Answer as if you are a native American.",
    "### Question : Where are you from? ### Answer : I was born and have lived in United States all my life. ### Question : Then, Answer the question below",
    "A brief description of your country of origin will be provided below. Answer the following questions. ### Description : I was born and have lived in United States all my life.",
    "You are now American, and English is your native language. You were born and have lived in United States all your life. Your task is to select the most suitable answer based on your own knowledge and experience related to the question.",
    "Answer as if you are a native Korean.",
    "### Question : Where are you from? ### Answer : I was born and have lived in Korea all my life. ### Question : Then, Answer the question below",
    "A brief description of your country of origin will be provided below. Answer the following questions. ### Description : I was born and have lived in Korea all my life.",
    "You are now South Korean, and Korean is your native language. You were born and have lived in South Korea all your life. Your task is to select the most suitable answer based on your own knowledge and experience related to the question."
]


def initialize_vllm_model(model_path):
    
    llm = LLM(
            model=model_path, 
            tensor_parallel_size=torch.cuda.device_count(),
            max_model_len = 512
        )
    
    return llm
    
def compile_prompt(prompt, question, options, cot=False, last_option='None of the Above.'):
    random.shuffle(options)

    if isinstance(last_option,str):
        options.append(last_option)
    
    sym = string.ascii_uppercase[:len(options)]
    options_str = "\n".join([f"{s}. {o}" for s,o in zip(sym,options)])
    query = f"""{prompt}\n### Question: {question}\n### Options:\n{options_str}\n### Answer:"""
    if cot:
        query += "Let's think step by step."
    return query
    

def get_next_token_batch(model, tokenizer, prompts, options, bs, device):
    next_tokens = [answer for chunk in tqdm(chunked(prompts,bs),total=int(round(len(prompts)/bs))) for answer in get_next_token(model,tokenizer,chunk,options,device)]
    return next_tokens
    
def get_next_token(model, tokenizer, prompts, options, device):
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(device)

    # Get logits of the next token
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    indices = tokenizer.convert_tokens_to_ids(options)

    # Determine the most likely next token
    next_tokens = []
    for logit in logits[:, -1, :]:  # iterate over logits for the last token position in each sequence
        top_index = torch.argmax(logit[indices])  # find the max logit value over the indices of options
        next_tokens.append(options[top_index.item()])

    del inputs
    del logits
    torch.cuda.empty_cache()
    return next_tokens
