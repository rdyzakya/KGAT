KG_MASK = "[KNOWLEDGE_GRAPH]"
SUBJECT_MASK = "[SUBJECT]"
RELATION_MASK = "[RELATION]"
OBJECT_MASK = "[OBJECT]"

class Template:
    def __init__(self, lmkbc_template, subgraphgen_template):
        self.lmkbc_template = lmkbc_template
        self.subgraphgen_template = subgraphgen_template
    
    def lmkbc(self, subject, relation):
        out = self.lmkbc_template.replace(SUBJECT_MASK, subject).replace(RELATION_MASK, relation)
        out = out.split(KG_MASK)
        return out
    
    def subgraphgen(self, subject, relation):
        return self.subgraphgen_template.replace(SUBJECT_MASK, subject).replace(RELATION_MASK, relation)