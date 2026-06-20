from dataclasses import dataclass
from jittor import dataset
from jittor.dataset import Dataset
from numpy import ndarray
from typing import List, Dict, Callable, Optional, Union

import jittor as jt
import numpy as np
import os

from .asset import Asset
from .augment import Augment
from .datapath import Datapath, LazyAsset
from .spec import ConfigSpec
from .transform import Transform


@dataclass
class DatasetConfig(ConfigSpec):
    shuffle: bool
    batch_size: int
    num_workers: int
    datapath: Datapath
    
    @classmethod
    def parse(cls, **kwargs) -> 'DatasetConfig':
        cls.check_keys(kwargs)
        return DatasetConfig(
            shuffle=kwargs.get('shuffle', False),
            batch_size=kwargs.get('batch_size', 1),
            num_workers=kwargs.get('num_workers', 0),
            datapath=Datapath.parse(**kwargs.get('datapath')), # type: ignore
        )
    
    def split_by_cls(self) -> Dict[Optional[str], 'DatasetConfig']:
        res: Dict[Optional[str], DatasetConfig] = {}
        datapath_dict = self.datapath.split_by_cls()
        for cls, v in datapath_dict.items():
            res[cls] = DatasetConfig(
                shuffle=self.shuffle,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                datapath=v,
            )
        return res

class PCDatasetModule():
    def __init__(
        self,
        process_fn: Optional[Callable[[List[Asset]], List[Dict]]]=None,
        train_dataset_config: Optional[DatasetConfig]=None,
        validate_dataset_config: Optional[Dict[Optional[str], DatasetConfig]]=None,
        predict_dataset_config: Optional[Dict[Optional[str], DatasetConfig]]=None,
        train_transform: Optional[Transform]=None,
        validate_transform: Optional[Transform]=None,
        predict_transform: Optional[Transform]=None,
        debug: bool=False,
    ):
        self.process_fn                 = process_fn
        self.train_dataset_config       = train_dataset_config
        self.validate_dataset_config    = validate_dataset_config
        self.predict_dataset_config     = predict_dataset_config
        self.train_transform            = train_transform
        self.validate_transform         = validate_transform
        self.predict_transform          = predict_transform
        self.debug = debug
        
        if debug:
            print("\033[31mWARNING: debug mode\033[0m")
        
        # build train datapath
        if self.train_dataset_config is not None:
            self.train_datapath = self.train_dataset_config.datapath
        else:
            self.train_datapath = None
        
        # build validate datapath
        if self.validate_dataset_config is not None:
            self.validate_datapath = {
                cls: self.validate_dataset_config[cls].datapath
                for cls in self.validate_dataset_config
            }
        else:
            self.validate_datapath = None
        
        # build predict datapath
        if self.predict_dataset_config is not None:
            self.predict_datapath = {
                cls: self.predict_dataset_config[cls].datapath
                for cls in self.predict_dataset_config
            }
        else:
            self.predict_datapath = None
        
    def train_dataloader(self):
        if self.train_transform is not None and self.train_dataset_config is not None and self.train_datapath is not None:
            self._train_ds = PCDataset(
                data=self.train_datapath.get_data(),
                transform=self.train_transform,
                name="train",
                process_fn=self.process_fn,
                debug=self.debug,
            )
        else:
            return None
        return self._create_dataloader(
            dataset=self._train_ds,
            config=self.train_dataset_config,
        )

    def validate_dataloader(self):
        if self.validate_dataset_config is not None and self.validate_transform is not None and self.validate_datapath is not None:
            self._validation_ds = {}
            for cls in self.validate_datapath:
                self._validation_ds[cls] = PCDataset(
                    data=self.validate_datapath[cls].get_data(),
                    transform=self.validate_transform,
                    name=f"validate-{cls}",
                    process_fn=self.process_fn,
                    debug=self.debug,
                )
        else:
            return None
        return self._create_dataloader(
            dataset=self._validation_ds,
            config=self.validate_dataset_config,
        )
    
    def predict_dataloader(self):
        if self.predict_transform is not None and self.predict_dataset_config is not None and self.predict_datapath is not None:
            self._predict_ds = {}
            for cls in self.predict_datapath:
                self._predict_ds[cls] = PCDataset(
                    data=self.predict_datapath[cls].get_data(),
                    transform=self.predict_transform,
                    name=f"predict-{cls}",
                    process_fn=self.process_fn,
                    debug=self.debug,
                )
        else:
            return None
        return self._create_dataloader(
            dataset=self._predict_ds,
            config=self.predict_dataset_config,
        )

    def _create_dataloader(
        self,
        dataset: Union[Dataset, Dict[str, Dataset]],
        config: Union[DatasetConfig, Dict[Optional[str], DatasetConfig]],
        **kwargs,
    ) -> Union[Dataset, Dict[str, Dataset]]:
        def create_single_dataloader(dataset: Dataset, config: DatasetConfig, **kwargs):
            dataset.set_attrs(
                batch_size=config.batch_size,
                total_len=len(config.datapath),
                shuffle=config.shuffle,
                num_workers=config.num_workers,
                drop_last=False,
            )
            return dataset
        if isinstance(dataset, Dict):
            assert isinstance(config, dict)
            return {k: create_single_dataloader(v, config[k], **kwargs) for k, v in dataset.items()}
        else:
            assert isinstance(config, DatasetConfig)
            return create_single_dataloader(dataset, config, **kwargs)


class PCDataset(Dataset):
    '''
    A simple dataset class.
    '''
    def __init__(
        self,
        data: List[LazyAsset],
        transform: Transform,
        name: Optional[str]=None,
        process_fn: Optional[Callable[[List[Asset]], List[Dict]]]=None,
        debug: bool=False,
    ):
        super().__init__()
        
        self.data       = data
        self.name       = name
        self.process_fn = process_fn
        self.transform  = transform
        self.debug      = debug
        
        if not debug:
            assert self.process_fn is not None, 'missing data processing function'
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, index) -> Asset:
        lazy_asset = self.data[index]
        asset = lazy_asset.load()
        self.transform.apply(asset=asset)
        return asset
    
    def _collate_fn_debug(self, batch):
        return batch # just retun a list of Asset
    
    def _collate_fn(self, batch):
        processed_batch = self.process_fn(batch) # type: ignore
        processed_batch: List[Dict]
        
        tensors_stack = {}
        tensors_cat = {}
        non_tensors = {}
        vis = {}
        def check(x):
            assert x not in vis, f"multiple keys found: {x}"
            vis[x] = True
        
        # NOTE: collate with numpy ONLY. This runs inside dataloader worker
        # processes; creating jt.Var here makes 12-24 workers contend with the
        # training process's jittor runtime and throttles data throughput.
        # Conversion to jt.Var happens on the main process at the model boundary
        # (VelocityModule.training_step / predict_step).
        def _to_np(v1, action: str, key):
            if isinstance(v1, ndarray):
                return v1
            if isinstance(v1, jt.Var):
                return v1.numpy()
            raise ValueError(f"cannot {action} non-tensor type of key {key}, type: {type(v1)}")

        for k, v in processed_batch[0].items():
            if k == "cat":
                assert isinstance(v, dict)
                for k1 in v.keys():
                    check(k1)
                    tensors_cat[k1] = [_to_np(processed_batch[i]['cat'][k1], "concatenate", k1)
                                       for i in range(len(processed_batch))]
            elif k == "non":
                assert isinstance(v, dict)
                for k1 in v.keys():
                    check(k1)
                    non_tensors[k1] = [processed_batch[i]['non'][k1]
                                       for i in range(len(processed_batch))]
            else:
                check(k)
                tensors_stack[k] = [_to_np(processed_batch[i][k], "stack", k)
                                    for i in range(len(processed_batch))]

        # Downcast float64 -> float32: jittor's hand-written CUDA ops (e.g. knn)
        # only support float32, and this matches the dtype the old jt.array-based
        # collate produced. Also halves the worker->main IPC payload.
        def _f32(a):
            return a.astype(np.float32, copy=False) if a.dtype == np.float64 else a
        collated_stack = {k: _f32(np.stack(v)) for k, v in tensors_stack.items()}
        collated_cat = {k: _f32(np.concatenate(v, axis=1)) for k, v in tensors_cat.items()}

        collated_batch = {
            **collated_stack,
            **collated_cat,
            **non_tensors,
        }
        return collated_batch
    
    def collate_batch(self, batch):
        if self.debug:
            return self._collate_fn_debug(batch)
        return self._collate_fn(batch)
