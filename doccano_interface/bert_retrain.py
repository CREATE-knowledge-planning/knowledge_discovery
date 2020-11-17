import json
import logging
import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import List, Optional, Union, Dict, Tuple, Callable
import numpy as np

import torch
from filelock import FileLock
from torch import nn
from torch.utils.data import Dataset, ChainDataset, RandomSampler, Subset, ConcatDataset
from transformers import AutoModelForTokenClassification, AutoTokenizer, Trainer, AutoConfig, TrainingArguments, \
    PreTrainedTokenizer, EvalPrediction
from seqeval.metrics import accuracy_score, f1_score, precision_score, recall_score

MODELS_DIR = "models"
DATASET_PATH = "./data/EOSS_sentences"


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

@dataclass
class InputExample:
    """
    A single training/test example for token classification.
    Args:
        guid: Unique id for the example.
        words: list. The words of the sequence.
        labels: (Optional) list. The labels for each word of the sequence. This should be
        specified for train and dev examples, but not for test examples.
    """

    guid: str
    sentence: str
    labels: Optional[Dict]


@dataclass
class InputFeatures:
    """
    A single set of features of data.
    Property names are the same names as the corresponding inputs to a model.
    """

    input_ids: List[int]
    attention_mask: List[int]
    token_type_ids: Optional[List[int]] = None
    label_ids: Optional[List[int]] = None


class Split(Enum):
    train = "train"
    dev = "dev"
    test = "test"


def read_examples_from_file(data_dir: Path, mode: Union[Split, str]) -> List[InputExample]:
    if isinstance(mode, Split):
        mode = mode.value
    paths_list = list(data_dir.glob("*.json"))
    guid_index = 1
    examples = []
    for file_path in paths_list:
        with file_path.open("r", encoding="utf-8") as data_file:
            data = json.load(data_file)
            max_num_examples = min(5000, len(data))
            for json_example in data[:max_num_examples]:
                labels = json_example[1]['entities']
                for label in labels:
                    if label[2] == "MEASUREMENT":
                        label[2] = "OBSERVABLE"
                examples.append(InputExample(guid=f"{mode}-{guid_index}",
                                             sentence=json_example[0],
                                             labels=labels))
                guid_index += 1
            print("File " + file_path.name + " processed")
    return examples


def read_doccano_examples(doccano_file_path: Path, mode: Union[Split, str]) -> List[InputExample]:
    if isinstance(mode, Split):
        mode = mode.value
    guid_index = 1
    examples = []
    logger.info("Reading Doccano file")
    with doccano_file_path.open("r", encoding="utf-8") as doccano_file:
        for line in doccano_file:
            if guid_index % 5000 == 0:
                logger.info(f"Reading sentence {guid_index}")
            example_json = json.loads(line)
            if example_json["annotation_approver"] is not None:
                examples.append(InputExample(guid=f"{mode}-{guid_index}",
                                             sentence=example_json["text"],
                                             labels=example_json["labels"]))
                guid_index += 1
    return examples


def convert_examples_to_features(
        examples: List[InputExample],
        label_list: List[str],
        max_seq_length: int,
        tokenizer: PreTrainedTokenizer,
        pad_token_label_id=-100,
) -> List[InputFeatures]:
    """Loads a data file into a list of `InputFeatures`
    `cls_token_at_end` define the location of the CLS token:
        - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
        - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
    `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """
    # TODO clean up all this to leverage built-in features of tokenizers

    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10_000 == 0:
            logger.info("Writing example %d of %d", ex_index, len(examples))

        tokens = []
        label_ids = []
        # First tokenize sentence and get splits
        sentence_tokens = tokenizer(example.sentence,
                                    add_special_tokens=True,
                                    padding='max_length',
                                    truncation=True,
                                    max_length=max_seq_length,
                                    return_token_type_ids=True,
                                    return_attention_mask=True,
                                    return_offsets_mapping=True)

        # Fill label_ids
        entity_idx = 0
        current_state = "O"
        for offsets in sentence_tokens["offset_mapping"]:
            if offsets[0] == 0 and offsets[1] == 0:
                label_ids.append(pad_token_label_id)
                current_state = "O"
            elif entity_idx < len(example.labels) and offsets[0] >= example.labels[entity_idx][0] and offsets[1] <= example.labels[entity_idx][1]:
                entity_label = example.labels[entity_idx][2]
                if current_state == "O":
                    label_ids.append(label_map[f"B-{entity_label}"])
                    current_state = entity_label
                elif current_state == entity_label:
                    label_ids.append(label_map[f"I-{entity_label}"])

                if offsets[1] == example.labels[entity_idx][1]:
                    entity_idx += 1
                    current_state = "O"
            else:
                label_ids.append(label_map["O"])
                current_state = "O"

        assert len(sentence_tokens["input_ids"]) == max_seq_length
        assert len(sentence_tokens["attention_mask"]) == max_seq_length
        assert len(sentence_tokens["token_type_ids"]) == max_seq_length
        assert len(label_ids) == max_seq_length

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s", example.guid)
            logger.info("tokens: %s", " ".join([str(x) for x in tokens]))
            logger.info("input_ids: %s", " ".join([str(x) for x in sentence_tokens["input_ids"]]))
            logger.info("input_mask: %s", " ".join([str(x) for x in sentence_tokens["attention_mask"]]))
            logger.info("segment_ids: %s", " ".join([str(x) for x in sentence_tokens["token_type_ids"]]))
            logger.info("label_ids: %s", " ".join([str(x) for x in label_ids]))

        features.append(
            InputFeatures(
                input_ids=sentence_tokens["input_ids"],
                attention_mask=sentence_tokens["attention_mask"],
                token_type_ids=sentence_tokens["token_type_ids"],
                label_ids=label_ids
            )
        )
    return features


class NERDataset(Dataset):
    features: List[InputFeatures]
    pad_token_label_id: int = nn.CrossEntropyLoss().ignore_index

    # Use cross entropy ignore_index as padding label id so that only
    # real label ids contribute to the loss later.

    def __init__(
            self,
            reader_func: Callable[[Path, Union[Split, str]], List[InputExample]],
            data_dir: Path,
            cached_file_path: Path,
            tokenizer: PreTrainedTokenizer,
            labels: List[str],
            max_seq_length: Optional[int] = None,
            overwrite_cache=False,
            mode: Split = Split.train,
    ):
        # Load data features from cache or dataset file


        # Make sure only the first process in distributed training processes the dataset,
        # and the others will use the cache.
        lock_path = str(cached_file_path.resolve()) + ".lock"
        with FileLock(lock_path):
            if os.path.exists(cached_file_path) and not overwrite_cache:
                logger.info(f"Loading features from cached file {str(cached_file_path)}")
                self.features = torch.load(cached_file_path)
            else:
                logger.info(f"Creating features from dataset file at {data_dir}")
                examples = reader_func(data_dir, mode)
                # TODO clean up all this to leverage built-in features of tokenizers
                self.features = convert_examples_to_features(
                    examples,
                    labels,
                    max_seq_length,
                    tokenizer,
                    pad_token_label_id=self.pad_token_label_id,
                )
                logger.info(f"Saving features into cached file {str(cached_file_path)}")
                torch.save(self.features, cached_file_path)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, i) -> InputFeatures:
        return self.features[i]


def main(model=None, new_model_name="entities_bert", models_dir=MODELS_DIR, n_iter=100):
    # Generate label types
    labels = ["INSTRUMENT", "ORBIT", "DESIGN_ID", "INSTRUMENT_PARAMETER", "OBSERVABLE", "MISSION", "OBJECTIVE",
              "SPACE_AGENCY", "STAKEHOLDER", "SUBOBJECTIVE", "TECHNOLOGY", "NOT_PARTIAL_FULL", "NUMBER",
              "YEAR", "AGENT", "WAVEBAND", "METHOD"]

    label_types = ['O']
    for label in labels:
        label_types.append('B-' + label)
        label_types.append('I-' + label)

    # Create dicts for mapping from labels to IDs and back
    tag2idx: Dict[str, int] = {t: i for i, t in enumerate(label_types)}
    idx2tag: Dict[int, str] = {i: t for t, i in tag2idx.items()}

    num_labels = len(label_types)
    cache_folder = Path('./cache')
    config = AutoConfig.from_pretrained(
        './model',
        num_labels=num_labels,
        id2label=idx2tag,
        label2id=tag2idx,
        cache_dir=cache_folder,
    )

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        './model',
        do_lower_case=False,
        use_fast=True,
        cache_dir=cache_folder,
    )

    """Obtain Training Data"""
    daphne_dataset = NERDataset(reader_func=read_examples_from_file,
                                data_dir=Path("./data/EOSS_sentences"),
                                cached_file_path=Path("./data/eoss_sentences_cached"),
                                tokenizer=tokenizer,
                                labels=label_types,
                                max_seq_length=128,
                                overwrite_cache=False)

    doccano_dataset = NERDataset(reader_func=read_doccano_examples,
                                 data_dir=Path("./data/doccano_output.json1"),
                                 cached_file_path=Path("./data/doccano_sentences_cached"),
                                 tokenizer=tokenizer,
                                 labels=label_types,
                                 max_seq_length=128,
                                 overwrite_cache=False)
    indices = torch.randint(low=0, high=len(daphne_dataset), size=(len(doccano_dataset),)).tolist()
    subdaphne_dataset = Subset(daphne_dataset, indices)
    train_dataset = ConcatDataset([subdaphne_dataset, doccano_dataset])
    eval_dataset = doccano_dataset

    """Set up the pipeline and entity recognizer, and train the new entity."""
    model = AutoModelForTokenClassification.from_pretrained(
        './model',
        config=config,
        cache_dir=cache_folder,
    )

    # Initialize our Trainer
    training_args = TrainingArguments(
        output_dir="./model",
        do_train=True,
        do_eval=True,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        save_total_limit=5,
        save_steps=5000,
        num_train_epochs=8.0,
    )

    def align_predictions(predictions: np.ndarray, label_ids: np.ndarray) -> Tuple[List[int], List[int]]:
        preds = np.argmax(predictions, axis=2)

        batch_size, seq_len = preds.shape

        out_label_list = [[] for _ in range(batch_size)]
        preds_list = [[] for _ in range(batch_size)]

        for i in range(batch_size):
            for j in range(seq_len):
                if label_ids[i, j] != nn.CrossEntropyLoss().ignore_index:
                    out_label_list[i].append(idx2tag[label_ids[i, j]])
                    preds_list[i].append(idx2tag[preds[i, j]])

        return preds_list, out_label_list

    def compute_metrics(p: EvalPrediction) -> Dict:
        preds_list, out_label_list = align_predictions(p.predictions, p.label_ids)
        return {
            "accuracy_score": accuracy_score(out_label_list, preds_list),
            "precision": precision_score(out_label_list, preds_list),
            "recall": recall_score(out_label_list, preds_list),
            "f1": f1_score(out_label_list, preds_list),
        }

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )

    # Training
    if training_args.do_train:
        trainer.train(model_path="./model")
        trainer.save_model()
        if trainer.is_world_master():
            tokenizer.save_pretrained(training_args.output_dir)

    # Evaluation
    results = {}
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        result = trainer.evaluate()

        output_eval_file = os.path.join(training_args.output_dir, "eval_results.txt")
        if trainer.is_world_master():
            with open(output_eval_file, "w") as writer:
                logger.info("***** Eval results *****")
                for key, value in result.items():
                    logger.info("  %s = %s", key, value)
                    writer.write("%s = %s\n" % (key, value))

            results.update(result)


if __name__ == "__main__":
    main()
