class Mask:
    KG_MASK = "<KNOWLEDGE_GRAPH>"
    SUBJECT_MASK = "<SUBJECT>"
    RELATION_MASK = "<RELATION>"
    OBJECT_MASK = "<OBJECT>"

NULL_SYM = "NONE" # empty

def post_process(text):
    text = text.split('|')[0]
    text = text.strip()
    return text