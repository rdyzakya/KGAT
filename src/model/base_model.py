from torch.nn import Module

ALLOWED_TYPE = [
    bool,
    int,
    float,
    str,
    type(None)
]

def is_allow_type(obj):
    if isinstance(obj, list):
        for el in obj:
            if not is_allow_type(el):
                return False
        return True
    elif isinstance(obj, dict):
        for k, v in obj.items():
            if not is_allow_type(k):
                return False
            if not is_allow_type(v):
                return False
        return True
    return type(obj) in ALLOWED_TYPE

class BaseModel(Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.attr_name = []
        for k, v in kwargs.items():
            if is_allow_type(v):
                self.__setattr__(k, v)
                self.attr_name.append(k)
                
    def save_attribute(self):
        result = {}
        for el in self.attr_name:
            result[el] = self.__getattribute__(el)
        return result