import json
from http.cookiejar import Cookie
from typing import Tuple

import requests
import numpy as np
from torch import Tensor
from transformers import AutoTokenizer, AutoModelForTokenClassification


class Client(object):

    def __init__(self, entrypoint='http://127.0.0.1', username=None, password=None):
        self.entrypoint = entrypoint
        self.client = requests.Session()
        json_data = {
            "username": username,
            "password": password
        }
        response = self.client.post(f'{self.entrypoint}/v1/auth-token', json=json_data)
        token = response.json()["token"]
        self.client.cookies.set("jwt", token)
        self.client.headers.update({"Authorization": f"Token {token}"})

    def fetch_projects(self):
        url = f'{self.entrypoint}/v1/projects'
        response = self.client.get(url).json()
        return response

    def create_project(self, name, description, project_type):
        mapping = {'SequenceLabeling': 'SequenceLabelingProject',
                   'DocumentClassification': 'TextClassificationProject',
                   'Seq2seq': 'Seq2seqProject'}
        data = {
            'name': name,
            'project_type': project_type,
            'description': description,
            'guideline': 'Hello',
            'resourcetype': mapping[project_type]
        }
        url = f'{self.entrypoint}/v1/projects'
        response = self.client.post(url, json=data)
        return response.json()

    def fetch_documents(self, project_id: int):
        url = f'{self.entrypoint}/v1/projects/{project_id}/docs'
        response = self.client.get(url)
        return response.json()

    def fetch_more_documents(self, next_url: str):
        response = self.client.get(next_url)
        return response.json()

    def add_document(self, project_id, text):
        data = {
            'text': text
        }
        url = f'{self.entrypoint}/v1/projects/{project_id}/docs'
        response = self.client.post(url, json=data)
        return response.json()

    def update_document(self, project_id, document):
        url = f'{self.entrypoint}/v1/projects/{project_id}/docs/{document["id"]}'
        response = self.client.put(url, data=document)
        return response.json()

    def fetch_labels(self, project_id):
        url = f'{self.entrypoint}/v1/projects/{project_id}/labels'
        response = self.client.get(url)
        return response.json()

    def add_label(self, project_id, text):
        data = {
            'text': text
        }
        url = f'{self.entrypoint}/v1/projects/{project_id}/labels'
        response = self.client.post(url, json=data)
        return response.json()

    def fetch_annotations(self, project_id, doc_id):
        url = f'{self.entrypoint}/v1/projects/{project_id}/docs/{doc_id}/annotations'
        response = self.client.get(url)
        return response.json()

    def annotate(self, project_id, doc_id, data):
        url = f'{self.entrypoint}/v1/projects/{project_id}/docs/{doc_id}/annotations'
        response = self.client.post(url, json=data)
        return response.json()

    def remove_annotation(self, project_id, doc_id, annotation_id):
        url = f"{self.entrypoint}/v1/projects/{project_id}/docs/{doc_id}/annotations/{annotation_id}"
        response = self.client.delete(url)
        if response.ok:
            return "Deleted correctly"
        else:
            return "Error"


def get_labels(text, model, tokenizer, label_types):
    # Create dicts for mapping from labels to IDs and back
    tag2idx = {t: i for i, t in enumerate(label_types)}
    idx2tag = {i: t for t, i in tag2idx.items()}

    sentence_tokens_for_model = tokenizer(text,
                                          add_special_tokens=True,
                                          padding='max_length',
                                          truncation=True,
                                          max_length=128,
                                          return_token_type_ids=True,
                                          return_attention_mask=True,
                                          return_tensors='pt')

    sentence_tokens_for_postprocessing = tokenizer(text,
                                                   add_special_tokens=True,
                                                   padding='max_length',
                                                   truncation=True,
                                                   max_length=128,
                                                   return_token_type_ids=True,
                                                   return_attention_mask=True,
                                                   return_offsets_mapping=True)

    result: Tuple[Tensor] = model(**sentence_tokens_for_model)
    scores = result[0].detach().numpy()
    label_ids = np.argmax(scores, axis=2)
    predictions = [idx2tag[i] for i in label_ids[0]]

    # Converts tags to 1/word
    entities = []
    current_state = 'O'
    current_word = -1
    current_entity = None
    spans = sentence_tokens_for_postprocessing.data["offset_mapping"][1:]
    for idx, ent_tag in enumerate(predictions[1:]):
        word_idx = sentence_tokens_for_postprocessing.token_to_word(idx + 1)
        if word_idx is None:
            break
        # Ignore all subwords when creating tags
        if current_word != word_idx:
            current_word = word_idx
            tag_info = ent_tag.split('-')
            if tag_info[0] in ['B', 'I']:
                # If beginning, start new entity and close last one
                if tag_info[0] == 'B':
                    if current_entity is not None:
                        entities.append(current_entity)
                    current_entity = [spans[idx][0], spans[idx][1], tag_info[1]]
                    current_state = tag_info[1]
                elif tag_info[0] == 'I' and tag_info[1] == current_state:
                    current_entity[1] = spans[idx][1]
            else:
                # Close current entity if exists
                if current_entity is not None:
                    entities.append(current_entity)
                    current_entity = None
                current_state = 'O'
        else:
            if current_entity is not None:
                current_entity[1] = spans[idx][1]
    # Add last entity if it exists
    if current_entity is not None:
        entities.append(current_entity)

    return entities


if __name__ == '__main__':
    # Generate label types
    labels = ["INSTRUMENT", "ORBIT", "DESIGN_ID", "INSTRUMENT_PARAMETER", "OBSERVABLE", "MISSION", "OBJECTIVE",
              "SPACE_AGENCY", "STAKEHOLDER", "SUBOBJECTIVE", "TECHNOLOGY", "NOT_PARTIAL_FULL", "NUMBER",
              "YEAR", "AGENT", "WAVEBAND"]

    label_types = ['O']
    for label in labels:
        label_types.append('B-' + label)
        label_types.append('I-' + label)

    # Initialize model
    MODEL_USED = "./model"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_USED, do_lower_case=False, use_fast=True)
    model = AutoModelForTokenClassification.from_pretrained(MODEL_USED)

    client = Client(username='admin', password='password')
    projects = client.fetch_projects()

    # Get project id for later
    project_name: str = "CREATE NLP"
    project_id: int = -1
    for project in projects:
        if project["name"] == project_name:
            project_id = project["id"]

    # Map label names to ids
    labels = client.fetch_labels(project_id)
    label_map = {}
    for label in labels:
        label_map[label["text"]] = label["id"]

    docs = client.fetch_documents(project_id)
    num_processed_docs = 0
    while num_processed_docs < 100:
        for doc in docs["results"]:
            # Check if doc has already been computed
            doc_meta = json.loads(doc["meta"])
            if not "autofilled" in doc_meta:
                # Mark document as computed
                doc["meta"] = json.dumps({"autofilled": True})
                client.update_document(project_id, doc)
                # Run inference on the text
                new_labels = get_labels(doc["text"], model, tokenizer, label_types)
                # Remove old annotations
                for annotation in doc["annotations"]:
                    client.remove_annotation(project_id, doc["id"], annotation["id"])
                # Add the new annotations
                for new_label in new_labels:
                    if new_label[2] in label_map:
                        data = {
                            'start_offset': new_label[0],
                            'end_offset': new_label[1],
                            'label': label_map[new_label[2]],
                            'prob': 0.9
                        }
                        client.annotate(project_id=project_id,
                                        doc_id=doc['id'],
                                        data=data)
            num_processed_docs += 1
        docs = client.fetch_more_documents(docs["next"])
