import luigi
from luigi.parameter import _FrozenOrderedDict

import pickle
import json
import numpy as np
import pandas as pd

class ModelTarget(luigi.LocalTarget):
    def __init__(self, model_class, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_class = model_class

    def load(self):
        return self.model_class.load(self.path)

    def dump(self, obj):
        self.makedirs()
        obj.save(self.path)

class _WrappedJsonEncoder(json.JSONEncoder):
    # hook json.dump to hundle with luigi's _FrozenOrderedDict
    def default(self, obj):
        if isinstance(obj, _FrozenOrderedDict):
            return obj.get_wrapped()
        return super().default(obj)


class JsonTarget(luigi.LocalTarget):
    def load(self):
        with self.open("r") as f:
            return json.load(f)

    def dump(self, obj):
        self.makedirs()
        with self.open("w") as f:
            return json.dump(obj, f, sort_keys=True, indent=4, cls=_WrappedJsonEncoder)

class CsvTarget(luigi.LocalTarget):
    def load(self, **kwargs):
        if "index_col" not in kwargs:
            kwargs["index_col"] = 0
        with self.open("r") as f:
            return pd.read_csv(f, **kwargs)

    def dump(self, obj, **kwargs):
        self.makedirs()
        df = pd.DataFrame(obj)
        with self.open("w") as f:
            return df.to_csv(f, **kwargs)
    
class NpyTarget(luigi.LocalTarget):
    def load(self, **kwargs):
        return np.load(self.path)
        
    def dump(self, ary):
        self.makedirs()
        np.save(self.path, np.asarray(ary))


