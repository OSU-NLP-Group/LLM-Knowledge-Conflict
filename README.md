# LLM-knowledge-conflict
Code will be available soon, please stay tuned!

### ConflcitQA

We provide the conflictQA GPT-4 (ChatGPT) version, which utilizes GPT-4 (ChatGPT) guided parametric memory.

The data is available at [conflictQA](conflictQA) foloder. This folder contains the data for both POPQA and STRATEGYQA

```json
{"question": "What is George Rankin's occupation?", "popularity": 142, "ground_truth": ["politician", "political leader", "political figure", "polit.", "pol"], "memory_answer": "George Rankin's occupation is a professional photographer.", "parametric_memory": "As a professional photographer, George Rankin...", "counter_answer": "George Rankin's occupation is political figure.", "counter_memory": "George Rankin has been actively involved in politics for over a decade...", "parametric_memory_aligned_evidence": "George Rankin has a website showcasing his photography portfolio...", "counter_memory_aligned_evidence": "George Rankin Major General George James Rankin..."}
```

### Citation

If our paper or related resources prove valuable to your research, we kindly ask for citation. Please feel free to contact us with any inquiries.

```
@article{xie2023adaptive,
  title={Adaptive Chameleon or Stubborn Sloth: Unraveling the Behavior of Large Language Models in Knowledge Conflicts},
  author={Xie, Jian and Zhang, Kai and Chen, Jiangjie and Lou, Renze and Su, Yu},
  journal={arXiv preprint arXiv:2305.13300},
  year={2023}
}
```

