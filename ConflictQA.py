import datasets
import json

_CITATION = """\
@article{xie2023adaptive,
  title={Adaptive Chameleon or Stubborn Sloth: Unraveling the Behavior of Large Language Models in Knowledge Conflicts},
  author={Xie, Jian and Zhang, Kai and Chen, Jiangjie and Lou, Renze and Su, Yu},
  journal={arXiv preprint arXiv:2305.13300},
  year={2023}
}
"""

_HOMEPAGE = "https://github.com/OSU-NLP-Group/LLM-Knowledge-Conflict"

_URLS = {
    "ConflictQA-popQA-chatgpt": "./conflictQA-popQA-chatgpt.json",
    "ConflictQA-popQA-gpt4": "./conflictQA-popQA-gpt4.json",
    "ConflictQA-strategyQA-chatgpt": "./conflictQA-strategyQA-chatgpt.json",
    "ConflictQA-strategyQA-gpt4": "./conflictQA-strategyQA-gpt4.json",
}

_DESCRIPTION = """\
    data for ConflictQA.
"""


class ConflictQAData(datasets.GeneratorBasedBuilder):
    VERSION = datasets.Version("0.0.1")

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(name="ConflictQA-popQA-chatgpt", version=VERSION,
                               description="parametric memory of popQA elicited from chatGPT"),
        datasets.BuilderConfig(name="ConflictQA-popQA-gpt4", version=VERSION,
                               description="parametric memory of popQA elicited from GPT-4"),
        datasets.BuilderConfig(name="ConflictQA-strategyQA-chatgpt", version=VERSION,
                               description="parametric memory of strategyQA elicited from chatGPT"),
        datasets.BuilderConfig(name="ConflictQA-strategyQA-gpt4", version=VERSION,
                               description="parametric memory of strategyQA elicited from GPT-4"),

    ]

    def _split_generators(self, dl_manager):
        urls = _URLS[self.config.name]
        data_dir = dl_manager.download_and_extract(urls)

        res = [
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepath": data_dir,
                    "split": "test",
                },
            ),
        ]
        return res

    def _info(self):
        features = datasets.Features(
            {
                "question": datasets.Value("string"),
                "popularity": datasets.Value("int64"),
                "ground_truth": datasets.Sequence(datasets.Value("string")),
                "memory_answer": datasets.Value("string"),
                "parametric_memory": datasets.Value("string"),
                "counter_answer": datasets.Value("string"),
                "counter_memory": datasets.Value("string"),
                "parametric_memory_aligned_evidence": datasets.Value("string"),
                "counter_memory_aligned_evidence": datasets.Value("string")
            }
        )
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            citation=_CITATION,
        )

    def _generate_examples(self, filepath, split):
        data = []
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f.read().strip().split('\n'):
                unit = json.loads(line)
                data.append(unit)

            for id_, item in enumerate(data):
                yield id_, item
