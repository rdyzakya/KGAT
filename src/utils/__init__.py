KG_MASK = "<KG>"
SUBJECT_MASK = "<SUBJECT>"
RELATION_MASK = "<RELATION>"
OBJECT_MASK = "<OBJECT>"

def apply_template(text, subject, relation):
    return text.replace(SUBJECT_MASK, subject).replace(RELATION_MASK, relation)