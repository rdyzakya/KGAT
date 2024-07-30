from utils import (
    KG_MASK,
    EMPTY_OBJECT,
    TRUE_FLAG,
    FALSE_FLAG,
)
import numpy as np
import re


class Prompt:
    def __init__(self):
        self.prompts = [
            {
                "prefix" : lambda subject, relation, n_tokens=1 : f"Based on the knowledge graph `{KG_MASK*n_tokens}`, complete the following triple with the format (subject : SUBJECT | relation : RELATION | object : OBJECT | T/F) and stop after close bracket, fill OBJECT with {EMPTY_OBJECT} if nothing satisfy, fill T/F with {TRUE_FLAG} if you think the triple is true else fill with {FALSE_FLAG} : (subject : {subject} | relation : {relation} | object : ",
                "suffix" : lambda object, negative_sample=False : f"{object} | {TRUE_FLAG if not negative_sample else FALSE_FLAG})",
                "regex" : re.compile(rf"(.+) \| ({TRUE_FLAG}|{FALSE_FLAG})")
            },
            {
                "prefix" : lambda subject, relation, n_tokens=1 : f"Given the knowledge graph {KG_MASK*n_tokens}, complete the following statement using the format (subject: SUBJECT | relation: RELATION | object: OBJECT | true/false). If no object satisfies the relation, use {EMPTY_OBJECT}. Indicate if the statement is true or false: (subject: {subject} | relation: {relation} | object: ",
                "suffix" : lambda object, negative_sample=False : f"{object} | {TRUE_FLAG if not negative_sample else FALSE_FLAG})",
                "regex" : re.compile(rf"(.+) \| ({TRUE_FLAG}|{FALSE_FLAG})")
            },
            {
                "prefix" : lambda subject, relation, n_tokens=1 : f"Using the knowledge graph `{KG_MASK*n_tokens}`, fill in the triple with the format (S: SUBJECT, R: RELATION, O: OBJECT, Valid: {TRUE_FLAG}/{FALSE_FLAG}). If no object fits, use {EMPTY_OBJECT} for OBJECT. Indicate the validity of the triple: (S: {subject}, R: {relation}, O: ",
                "suffix" : lambda object, negative_sample=False : f"{object}, Valid: {TRUE_FLAG if not negative_sample else FALSE_FLAG})",
                "regex" : re.compile(rf"(.+) , Valid: ({TRUE_FLAG}|{FALSE_FLAG})")
            },
            {
                "prefix" : lambda subject, relation, n_tokens=1 : f"Referencing the knowledge graph {KG_MASK*n_tokens}, complete the triple in the format (subject: SUBJECT, predicate: RELATION, object: OBJECT, valid: {TRUE_FLAG}/{FALSE_FLAG}). If no object satisfies, use {EMPTY_OBJECT}. Indicate if the triple is true or false: (subject: {subject}, predicate: {relation}, object: ",
                "suffix" : lambda object, negative_sample=False : f"{object}, valid: {TRUE_FLAG if not negative_sample else FALSE_FLAG})",
                "regex" : re.compile(rf"(.+), valid: ({TRUE_FLAG}|{FALSE_FLAG})")
            },
            {
                "prefix" : lambda subject, relation, n_tokens=1 : f"Based on the knowledge graph `{KG_MASK*n_tokens}`, fill in the missing information in the format (subj: SUBJECT | pred: RELATION | obj: OBJECT | correct: T/F). If there is no suitable object, use {EMPTY_OBJECT}. Mark T/F depending on the truth of the statement: (subj: {subject} | pred: {relation} | obj: ",
                "suffix" : lambda object, negative_sample=False : f"{object} | correct: {TRUE_FLAG if not negative_sample else FALSE_FLAG})" ,
                "regex" : re.compile(rf"(.+) \| correct: ({TRUE_FLAG}|{FALSE_FLAG})")
            },
            {
                "prefix" : lambda subject, relation, n_tokens=1 : f"Using the knowledge graph {KG_MASK*n_tokens}, complete the following triple: (subject: SUBJECT - relation: RELATION - object: OBJECT - T/F). If there is no valid object, fill in with {EMPTY_OBJECT}. Specify T/F based on the accuracy of the triple: (subject: {subject} - relation: {relation} - object: ",
                "suffix" : lambda object, negative_sample=False : f"{object} - {TRUE_FLAG if not negative_sample else FALSE_FLAG})",
                "regex" : re.compile(rf"(.+) - ({TRUE_FLAG}|{FALSE_FLAG})")
            },
            {
                "prefix" : lambda subject, relation, n_tokens=1 : f"Based on the knowledge graph `{KG_MASK*n_tokens}`, fill in the following with the format (Subject: SUBJECT | Relation: RELATION | Object: OBJECT | True/False). Use {EMPTY_OBJECT} if no object fits, and mark {TRUE_FLAG}/{FALSE_FLAG} to indicate if the triple is correct: (Subject: {subject} | Relation: {relation} | Object: ",
                "suffix" : lambda object, negative_sample=False : f"{object} | {TRUE_FLAG if not negative_sample else FALSE_FLAG})",
                "regex" : re.compile(rf"(.+) \| ({TRUE_FLAG}|{FALSE_FLAG})")
            },
            {
                "prefix" : lambda subject, relation, n_tokens=1 : f"Using the knowledge graph {KG_MASK*n_tokens}, complete the triple in this format (subject: SUBJECT | relation: RELATION | object: OBJECT | validity: {TRUE_FLAG}/{FALSE_FLAG}). If there’s no suitable object, enter {EMPTY_OBJECT}. Indicate {TRUE_FLAG} or {FALSE_FLAG} based on the triple's validity: (subject: {subject} | relation: {relation} | object: ",
                "suffix" : lambda object, negative_sample=False : f"{object} | validity: {TRUE_FLAG if not negative_sample else FALSE_FLAG})",
                "regex" : re.compile(rf"(.+) \| validity: ({TRUE_FLAG}|{FALSE_FLAG})")
            },
            {
                "prefix" : lambda subject, relation, n_tokens=1 : f"Given the knowledge graph `{KG_MASK*n_tokens}`, complete the following triple: (sub: SUBJECT, rel: RELATION, obj: OBJECT, valid: {TRUE_FLAG}/{FALSE_FLAG}). If no suitable object exists, use {EMPTY_OBJECT}. Mark {TRUE_FLAG}/{FALSE_FLAG} to indicate the truth of the statement: (sub: {subject}, rel: {relation}, obj: ",
                "suffix" : lambda object, negative_sample=False : f"{object}, valid: {TRUE_FLAG if not negative_sample else FALSE_FLAG})",
                "regex" : re.compile(rf"(.+), valid: ({TRUE_FLAG}|{FALSE_FLAG})")
            },
        ]
    
    def pick(self, subject, relation, object=None, n_tokens=1, negative_sample=False, inference=False, idx=None):
        if isinstance(idx, int):
            chosen_prompt = self.prompts[idx]
        else:
            idx = np.random.randint(0, len(self.prompts))
            chosen_prompt = self.prompts[idx]

        prefix = chosen_prompt["prefix"](subject, relation, n_tokens=n_tokens)
        
        if inference:
            return prefix, idx
        
        assert object is not None
        
        suffix = chosen_prompt["suffix"](object, negative_sample=negative_sample)

        return prefix + suffix, idx
    
    def regex(self, idx, text):
        pattern = self.prompts[idx]["regex"]
        m = pattern.match(text)
        return m.group(1).strip(), m.group(2)