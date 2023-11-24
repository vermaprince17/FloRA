import copy
import json

import numpy as np
from torch.utils.data import Dataset
from transformers import AutoTokenizer

TEMPLATE = {
    "description": "Template used by LLM.",
    "prompt_no_input_format": "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:\n",
    "prompt_with_input_format": "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n",
    "response_split": "### Response:",
}


class LMPrompter:
    def __call__(self, instruction, input=None):
        if input is None or len(input) == 0:
            return TEMPLATE["prompt_no_input_format"].format(instruction=instruction)
        else:
            return TEMPLATE["prompt_with_input_format"].format(
                instruction=instruction, input=input
            )

    def get_response(self, output: str) -> str:
        return output.split(TEMPLATE["response_split"])[-1].strip()


class PubMedQADataset(Dataset):
    """Each line of the annotation file is a json object with the following fields:
    {
      'QUESTION': 'Do mitochondria play a role in remodelling lace plant leaves during programmed cell death?',
      'CONTEXTS': ['Programmed cell death (PCD) is the regulated death of cells within an organism. The lace plant (Aponogeton madagascariensis) produces perforations in its leaves through PCD. The leaves of the plant consist of a latticework of longitudinal and transverse veins enclosing areoles. PCD occurs in the cells at the center of these areoles and progresses outwards, stopping approximately five cells from the vasculature. The role of mitochondria during PCD has been recognized in animals; however, it has been less studied during PCD in plants.',
        'The following paper elucidates the role of mitochondrial dynamics during developmentally regulated PCD in vivo in A. madagascariensis. A single areole within a window stage leaf (PCD is occurring) was divided into three areas based on the progression of PCD; cells that will not undergo PCD (NPCD), cells in early stages of PCD (EPCD), and cells in late stages of PCD (LPCD). Window stage leaves were stained with the mitochondrial dye MitoTracker Red CMXRos and examined. Mitochondrial dynamics were delineated into four categories (M1-M4) based on characteristics including distribution, motility, and membrane potential (ΔΨm). A TUNEL assay showed fragmented nDNA in a gradient over these mitochondrial stages. Chloroplasts and transvacuolar strands were also examined using live cell imaging. The possible importance of mitochondrial permeability transition pore (PTP) formation during PCD was indirectly examined via in vivo cyclosporine A (CsA) treatment. This treatment resulted in lace plant leaves with a significantly lower number of perforations compared to controls, and that displayed mitochondrial dynamics similar to that of non-PCD cells.'],
      'LABELS': ['BACKGROUND', 'RESULTS'],
      'MESHES': ['Alismataceae',
        'Apoptosis',
        'Cell Differentiation',
        'Mitochondria',
        'Plant Leaves'],
      'YEAR': '2011',
      'reasoning_required_pred': 'yes',
      'reasoning_free_pred': 'yes',
      'final_decision': 'yes',
      'LONG_ANSWER': 'Results depicted mitochondrial dynamics in vivo as PCD progresses within the lace plant, and highlight the correlation of this organelle with other organelles during developmental PCD. To the best of our knowledge, this is the first report of mitochondria and chloroplasts moving on transvacuolar strands to form a ring structure surrounding the nucleus during developmental PCD. Also, for the first time, we have shown the feasibility for the use of CsA in a whole plant system. Overall, our findings implicate the mitochondria as playing a critical and early role in developmentally regulated PCD in the lace plant.'
    }
    """

    def __init__(
        self, tokenizer, ann_path: str, add_eos=True, ignore_instruction=True, **kwargs
    ):
        """
        ann_path (string): directory to store the annotation file
        """
        # assert tokenizer.add_eos_token is False, "tokenizer should not add eos token by default"
        self.tokenizer: AutoTokenizer = tokenizer

        self.annotation = []
        self.prompter = LMPrompter()
        self.add_eos = add_eos
        self.ignore_instruction = ignore_instruction
        self.load_annotation(ann_path)

    def load_annotation(self, ann_path):
        self.annotation = []
        # for line in open(ann_path, "r").readlines():
        #     print(line)
        #     self.annotation.append(json.loads(line))
        with open(ann_path, "r") as file:
            data = json.load(file)

        for key in data.keys():
            self.annotation.append(data[key])

    def __len__(self):
        return len(self.annotation)

    def process_text(self, ann):
        instruction = ann["QUESTION"]
        context = ann["CONTEXTS"]
        response = ann["LONG_ANSWER"]
        instruction = self.prompter(instruction=instruction, input=context)
        return dict(instruction=instruction, answer=response)

    def tokenize(self, text):
        res = self.tokenizer(
            text["instruction"] + text["answer"],
            return_tensors=None,
            padding="do_not_pad",
            truncation=True,
            max_length=512,
        )

        # manually add eos token
        if (
            res["input_ids"][-1] != self.tokenizer.eos_token_id
            and len(res["input_ids"]) < 512
            and self.add_eos
        ):
            res["input_ids"].append(self.tokenizer.eos_token_id)
            res["attention_mask"].append(1)
        labels = copy.deepcopy(res["input_ids"])
        # ignore instruction_token
        if self.ignore_instruction:
            instruction_token = self.tokenizer(
                text["instruction"],
                return_tensors=None,
                padding="do_not_pad",
                truncation=True,
                max_length=512,
            )
            labels = [-100] * len(instruction_token["input_ids"]) + labels[
                len(instruction_token["input_ids"]) :
            ]

        res.update(labels=labels)
        return res

    def __getitem__(self, index):
        ann = self.annotation[index]
        text = self.process_text(ann)
        res = self.tokenize(text)
        res.update(text)
        return res

    def collater(self, samples):
        question_list, answer_list, input_id_list, attention_mask_list, labels_list = (
            [],
            [],
            [],
            [],
            [],
        )

        for sample in samples:
            question_list.append(sample["instruction"])
            answer_list.append(sample["answer"])
            input_id_list.append(sample["input_ids"])
            attention_mask_list.append(sample["attention_mask"])
            labels_list.append(sample["labels"])

        # We have to pad the labels before calling `tokenizer.pad` as this method won't pad them and needs them of the
        # same length to return tensors.
        max_label_length = max(len(l) for l in labels_list)
        padding_side = self.tokenizer.padding_side
        padded_labels = []
        for l in labels_list:
            remainder = [-100] * (max_label_length - len(l))
            if isinstance(l, list):
                l = l + remainder if padding_side == "right" else remainder + l
            elif padding_side == "right":
                l = np.concatenate([l, remainder]).astype(np.int64)
            else:
                l = np.concatenate([remainder, l]).astype(np.int64)
            padded_labels.append(l)

        padded_samples = self.tokenizer.pad(
            {
                "input_ids": input_id_list,
                "attention_mask": attention_mask_list,
                "labels": padded_labels,
            },
            return_tensors="pt",
            padding="longest",
        )

        labels = padded_samples["labels"]
        labels[labels == self.tokenizer.pad_token_id] = -100
        labels[:, 0] = -100
        return {
            "input_ids": padded_samples["input_ids"],
            "attention_mask": padded_samples["attention_mask"],
            "labels": labels,
            "instruction": question_list,
            "answer": answer_list,
        }


def build_pubmedqa_dataset(
    tokenizer,
    ann_path="data/pubmedqa/ori_pqal.json",
    **kwargs,
):
    return PubMedQADataset(
        tokenizer=tokenizer,
        ann_path=ann_path,
        **kwargs,
    )
