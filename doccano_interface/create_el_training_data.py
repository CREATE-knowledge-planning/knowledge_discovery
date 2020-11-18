import json
from pathlib import Path
from typing import Union, List

import numpy as np
import spacy
from spacy.kb import KnowledgeBase
from spacy.vocab import Vocab


def read_doccano_examples(doccano_file_path: Path) -> List[dict]:
    guid_index = 1
    examples = []
    with doccano_file_path.open("r", encoding="utf-8") as doccano_file:
        for line in doccano_file:
            example_json = json.loads(line)
            if example_json["annotation_approver"] is not None:
                examples.append({
                    "sentence": example_json["text"],
                    "labels": example_json["labels"]
                })
    return examples


def main():
    # 1. Read all training data
    doccano_file_path = Path("./data/doccano_output.json1")
    examples = read_doccano_examples(doccano_file_path)
    ent_alias_map = {}
    with open("./kg/ent_alias_map.txt", "r") as ent_alias_file:
        for line in ent_alias_file:
            ent_id, alias = line.split("\t")
            ent_alias_map[ent_id] = alias[:-1]

    # 2. Go through Knowledge Base and compute short-list of candidates for each entity from ordered distances in embedding space
    vocab = Vocab().from_disk("./kg/vocab")
    kb = KnowledgeBase(vocab=vocab)
    kb.load_bulk("./kg/kb")
    nlp = spacy.load("en_core_web_md")

    options_list = []

    for example in examples:
        options_list.append([])
        for label in example["labels"]:
            label_text = example["sentence"][label[0]:label[1]]
            label_vector = nlp(example["sentence"][label[0]:label[1]]).vector
            matching_candidates = []
            for kb_entity in kb.get_entity_strings():
                if kb_entity.lower().startswith(label[2].lower()):
                    matching_candidates.append((kb_entity, np.linalg.norm(kb.get_vector(kb_entity)-label_vector)))
            matching_candidates.sort(key=lambda el: el[1])
            options_list[-1].append({
                "text": label_text, "candidates": matching_candidates[0:5]
            })


    # 3. Let user choose correct entity from short-list (or keep it null)
    with open("./data/el_training_data.jsonl", "w") as training_dataset:
        for ex_idx, example in enumerate(options_list):
            for l_idx, label_options in enumerate(example):
                aliases = [ent_alias_map[option[0]] for option in label_options["candidates"]]
                print("Choose one of the options to train or leave it empty for none")
                output_str = f"For '{label_options['text']}', choose: "
                for idx, alias in enumerate(aliases):
                    output_str += f"{idx} - '{alias}'\t"
                output_str += "(empty) - None"
                print(output_str)
                choice = input()
                if choice == "":
                    chosen_kb_entity = None
                else:
                    chosen_kb_entity = label_options["candidates"][int(choice)][0]
                examples[ex_idx]["labels"][l_idx].append(chosen_kb_entity)
            training_dataset.write(json.dumps(examples[ex_idx]) + "\n")
            training_dataset.flush()


if __name__ == "__main__":
    main()
