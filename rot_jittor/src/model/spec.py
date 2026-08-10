from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass
from jittor import nn
from numpy import ndarray
from typing import Dict, List, Optional, final

import numpy as np
import os
import jittor as jt

from ..data.asset import Asset

@dataclass
class ModelInput():
    asset: Asset
    tokens: Optional[ndarray]=None

class ModelSpec(nn.Module, ABC):
    
    model_config: Dict
    transform_config: Dict
    
    @abstractmethod
    def __init__(self, model_config, transform_config):
        super().__init__()
        if not isinstance(model_config, dict) or not isinstance(transform_config, dict):
            raise TypeError("trimmed ROT runtime expects plain dict configurations")
        self.model_config = dict(model_config)
        self.transform_config = dict(transform_config)
        self._is_predict = False
    
    def is_predict(self):
        return self._is_predict
    
    def set_predict(self, is_predict: bool):
        self._is_predict = is_predict
    
    @final
    def _process_fn(self, batch: List[Asset]) -> List[Dict]:
        n_batch = self.process_fn(batch)
        DEBUG = os.getenv("DEBUG") == "1"
        if DEBUG or not self.is_training():
            for k in n_batch[0].keys():
                if not isinstance(n_batch[0][k], ndarray) and not isinstance(n_batch[0][k], jt.Var):
                    continue
                s = n_batch[0][k].shape
                for i in range(1, len(n_batch)):
                    assert n_batch[i][k].shape == s, f"{k} has different shape in batch"
            for (i, b) in enumerate(batch):
                non = n_batch[i].get('non', {})
                non['asset'] = deepcopy(b)
                n_batch[i]['non'] = non
        return n_batch
    
    @abstractmethod
    def process_fn(self, batch: List[Asset]) -> List[Dict]:
        """
        Fetch data from dataloader and turn it into Tensor objects.
        """
        raise NotImplementedError()
    
    def compile_model(self):
        """
        Compile the model. Do this before training and after loading state dicts.
        """
        pass
    
    @classmethod
    def load_ckpt(cls, checkpoint_path: str):
        model = jt.load(checkpoint_path)
        return model
    
    def get_train_transform(self) -> Optional[object]:
        return self._get_transform('train_transform')
    
    def get_validate_transform(self) -> Optional[object]:
        return self._get_transform('validate_transform')
    
    def get_predict_transform(self) -> Optional[object]:
        return self._get_transform('predict_transform')

    def _get_transform(self, name: str) -> Optional[object]:
        cfg = self.transform_config.get(name)
        if cfg is None:
            return None
        # Training-only path; formal ROT inference passes an empty transform config.
        from ..data.transform import Transform
        return Transform.parse(**cfg)
    
    def predict_step(self, batch: Dict) -> List[Dict]:
        raise NotImplementedError()