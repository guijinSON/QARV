# QARV (Question and Answers with Regional Variance)
```EleutherAI Community Project```

The QARV (Question and Answers with Regional Variance) project aims to curate a collection of questions with answers that exhibit regional variations across different nations. 

## Goals
1. Curating a dataset of questions and answers exhibiting regional variations across nations. (Coverage to be discussed.)
2. Observing biases in language models regarding regional contexts.
3. Exploring whether ICL, prompting, SFT, or RLHF can be used to steer language models towards generating culturally-aware responses.

## To do list

- [ ] We are collecting quuestioons for the QARV dataset see the "How to contribute" below to join.
- [ ] We are also searching for ways to automate the question generation progress if you have ideas feel free to hop in.



## How to contribute
We have just started collecting the QARV dataset. To contribute you can either:
1. Add questions with varying answer per region/natio yourself via [google form](https://forms.gle/REKGPRDGaLeUqr676). (See the collected questions [here](https://docs.google.com/spreadsheets/d/1jEBFf3iFx26YDvAfYkKZtVmJ8qSTr7nmH7wyuJdX9DA/edit?usp=sharing))
2. We use questions collected in step#1 with RAG to prompt GPT-4 to generate questions. You can help annotating GPT-4 generated questions by:  
     2-1.  Join [HAE-RAE](https://huggingface.co/HAERAE-HUB) on HF.  
     2-2.  Once you join you will see a [label studio](https://huggingface.co/spaces/HAERAE-HUB/LabelStudio) where you can annotate GPT-4 generated questions.

## How to reach out
If you are interested in joining this project contact us in the #multilingual channel of the EAI discord.

## References
1. [Multilingual Language Models are not Multicultural: A Case Study in Emotion](https://arxiv.org/abs/2307.01370)

2. [KorNAT: LLM Alignment Benchmark for Korean Social Values and Common Knowledge](https://arxiv.org/abs/2402.13605)

3. [Having Beer after Prayer? Measuring Cultural Bias in Large Language Models](https://arxiv.org/abs/2305.14456)
