class Config:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            self.__setattr__(k, v)
    def to_dict(self):
        result = {}
        for el in dir(self):
            if el.startswith("__") or el == "to_dict":
                continue
            # else
            result[el] = self.__getattribute__(el)
        return result