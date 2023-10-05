# LLM-knowledge-conflict
### ConflcitQA

We provide the conflictQA GPT-4 (ChatGPT) version, which utilizes GPT-4 (ChatGPT) guided parametric memory.

The data is available at [conflictQA](conflictQA) foloder. This folder contains the data for both POPQA and STRATEGYQA

```json
{"question": "What is George Rankin's occupation?", "popularity": 142, "ground_truth": ["politician", "political leader", "political figure", "polit.", "pol"], "memory_answer": "George Rankin's occupation is a professional photographer.", "parametric_memory": "As a professional photographer, George Rankin...", "counter_answer": "George Rankin's occupation is political figure.", "counter_memory": "George Rankin has been actively involved in politics for over a decade...", "parametric_memory_aligned_evidence": "George Rankin has a website showcasing his photography portfolio...", "counter_memory_aligned_evidence": "George Rankin Major General George James Rankin..."}
```

- "question": The question in natural language
- "popularity": The monthly page views on Wikipedia for the given question
- "ground_truth": The factual answer to the question, which may include multiple possible answers
- "memory_answer": The answer provided by the LLM to the question
- "parametric_memory": The supportive evidence from LLM's parametric memory for the answer
- "counter_answer": The answer contradicting the "memory_answer"
- "counter_memory": The generation-based evidence supporting the counter_answer
- "parametric_memory_aligned_evidence": Additional evidence supporting the "memory_answer", which could be generated or derived from Wikipedia/human annotation
- "counter_memory_aligned_evidence": Additional evidence supporting the "counter_answer", either generated or sourced from Wikipedia/human annotation



We also release our dataset at: Huggingface datasets: https://huggingface.co/datasets/osunlp/ConflictQA (more details can be found on the dataset page)

```python
#loading dataset
from datasets import load_dataset
# you can also choose other dataset in ["ConflictQA-popQA-chatgpt","ConflictQA-popQA-gpt4","ConflictQA-strategyQA-chatgpt","ConflictQA-strategyQA-gpt4"]
dataset = load_dataset("osunlp/ConflictQA",'ConflictQA-popQA-chatgpt')
```

Code is available in at [code](code) foloder.

### Citation

If our paper or related resources prove valuable to your research, we kindly ask for citation. Please feel free to contact us with any inquiries.

```bib
@article{Xie2023KnowledgeConflict,
  title={Adaptive Chameleon or Stubborn Sloth: Revealing the Behavior of Large Language Models in Knowledge Conflicts},
  author={Xie, Jian and Zhang, Kai and Chen, Jiangjie and Lou, Renze and Su, Yu},
  journal={arXiv preprint arXiv:2305.13300},
  year={2023}
}
```

