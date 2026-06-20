from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from jittor import optim
from typing import Dict, List, Optional, TYPE_CHECKING
from tqdm import tqdm

import jittor as jt
import os

from ..data.asset import Asset

if TYPE_CHECKING:
    from ..data.dataset import PCDatasetModule
    from ..model.spec import ModelSpec

def _get_item(x):
    if isinstance(x, jt.Var):
        return x.item()
    return x

@dataclass(frozen=True)
class DistributedContext:
    enabled: bool
    rank: int
    local_rank: int
    world_size: int

def _safe_int(value, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default

def get_distributed_context() -> DistributedContext:
    """Return the active Jittor MPI distributed context.

    Jittor enables distributed Dataset sharding and gradient all-reduce when
    `jt.in_mpi` is true. Keep this wrapper small so the training loop can make
    rank-aware choices without depending on MPI in single-process runs.
    """
    in_mpi = bool(getattr(jt, "in_mpi", False))
    mpi = getattr(jt, "mpi", None)
    rank = _safe_int(getattr(jt, "rank", 0), 0)
    world_size = _safe_int(getattr(jt, "world_size", 1), 1)
    local_rank = _safe_int(os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK"), rank)

    if in_mpi and mpi is not None:
        rank = _safe_int(mpi.world_rank(), rank)
        world_size = _safe_int(mpi.world_size(), world_size)
        if hasattr(mpi, "local_rank"):
            local_rank = _safe_int(mpi.local_rank(), local_rank)

    return DistributedContext(
        enabled=in_mpi and world_size > 1,
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
    )

def is_primary_process(context: Optional[DistributedContext]=None) -> bool:
    if context is None:
        context = get_distributed_context()
    return context.rank == 0

def get_optimizer(optimizer_config, model):
    __target__ = optimizer_config.pop('__target__')
    MAPPING = {
        'sgd': optim.SGD,
        'adam': optim.Adam,
    }
    if __target__ not in MAPPING:
        raise ValueError(f"unsupported optimizer: {__target__}")
    OptimizerClass = MAPPING[__target__]
    optimizer = OptimizerClass(model.parameters(), **optimizer_config)
    return optimizer

class DummyWriter():
    
    def __init__(self):
        pass
    
    def write(self, batch, prediction: List[Dict], dataset_module: Optional[PCDatasetModule]=None):
        pass

class DummySystem():
    
    def __init__(
        self,
        dataset_module: PCDatasetModule,
        model: ModelSpec,
        loss_config=None,
        optimizer_config=None,
        trainer_config=None,
        writer: Optional[DummyWriter]=None,
        
        ckpt_save_dir: str="experiments",
        ckpt_save_name: str="checkpoint",
    ):
        self.dataset_module = dataset_module
        self.model = model
        self.loss_config = loss_config
        self.ckpt_save_dir = ckpt_save_dir
        self.ckpt_save_name = ckpt_save_name
        self.writer = writer
        if trainer_config is None:
            trainer_config = {}
        self.epochs = int(trainer_config.get('epochs', 1))
        early_stopping_config = trainer_config.get('early_stopping', {})
        if isinstance(early_stopping_config, bool):
            early_stopping_config = {'enabled': early_stopping_config}
        self.early_stopping_enabled = bool(early_stopping_config.get('enabled', False))
        self.monitor = early_stopping_config.get('monitor', 'val/loss_sum')
        self.monitor_mode = early_stopping_config.get('mode', 'min')
        if self.monitor_mode not in {'min', 'max'}:
            raise ValueError(f"unsupported early stopping mode: {self.monitor_mode}")
        self.early_stopping_patience = int(early_stopping_config.get('patience', 10))
        self.min_delta = float(early_stopping_config.get('min_delta', 0.0))
        self.save_best_only = bool(trainer_config.get('save_best_only', self.early_stopping_enabled))
        self.best_metric: Optional[float] = None
        self.best_epoch: Optional[int] = None
        self._epochs_without_improvement = 0
        
        if optimizer_config is not None and model is not None:
            self.optimizer = get_optimizer(optimizer_config, model)
        else:
            self.optimizer = None
        
        self._validation_loss = defaultdict(list)
        self.distributed = get_distributed_context()
        self.is_primary_process = is_primary_process(self.distributed)
        if self.distributed.enabled:
            print(
                "[DDP] "
                f"rank={self.distributed.rank}/"
                f"{self.distributed.world_size}, "
                f"local_rank={self.distributed.local_rank}"
            )
            if self.model is not None:
                self.model.mpi_param_broadcast(root=0)
    
    def forward(self, batch, validate: bool=False): # return loss sum
        loss_dict = self.model.training_step(batch)
        assert isinstance(loss_dict, dict), "loss_dict must be a dict containing loss/metrics"
        assert self.loss_config is not None, "do not have loss_confing"
        loss_sum = 0.
        if validate:
            assets: List[Asset] = [a for a in batch['asset']]
            cls = assets[0].cls # guaranteed to be the same cls in dataloader
            for name in loss_dict:
                assert name in self.loss_config, f'unspecified loss {name}'
                self._validation_loss[f"val/{cls}_{name}"].append(_get_item(loss_dict[name]))
                loss_sum += self.loss_config[name] * loss_dict[name]
            self._validation_loss[f"val/{cls}_loss_sum"].append(_get_item(loss_sum))
            # TODO: log
            # self.log('val/loss_sum', loss_sum, prog_bar=True, logger=True, sync_dist=True, batch_size=len(assets))
        else:
            for name in loss_dict:
                assert name in self.loss_config, f"unspecified loss name: `{name}`"
                if self.loss_config[name] > 0:
                    loss_sum += self.loss_config[name] * loss_dict[name]
            loss_dict['loss_sum'] = loss_sum
            # TODO: log
            # # add train prefix to loss_dict
            # prefixed_loss_dict = {f"train/{k}": v for k, v in loss_dict.items()}
            # d = dict(sorted(prefixed_loss_dict.items()))
        if not isinstance(loss_sum, jt.Var):
            return jt.array(loss_sum)
        return loss_sum
    
    def on_train_epoch_start(self):
        pass
    
    def on_train_batch_start(self):
        pass
    
    def training_step(self, batch):
        return self.forward(batch, validate=False)
    
    def on_train_batch_end(self):
        pass
    
    def on_train_epoch_end(self):
        pass
    
    def on_validation_epoch_start(self):
        self._validation_loss = defaultdict(list)
    
    def on_validation_batch_start(self):
        pass
    
    def validation_step(self, batch):
        assert self.loss_config is not None, "do not have loss_confing"
        return self.forward(batch, validate=True)
    
    def on_validation_batch_end(self):
        pass
    
    def on_validation_epoch_end(self):
        pass
    
    def on_before_optimizer_step(self, optimizer):
        pass
    
    def on_predict_epoch_start(self):
        pass
    
    def on_predict_batch_start(self):
        pass
    
    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        return self.model.predict_step(batch)
    
    def on_predict_batch_end(self):
        pass
    
    def on_predict_epoch_end(self):
        pass

    def _make_pbar(self, dataloader, description: Optional[str]=None):
        total = len(dataloader)//dataloader.batch_size
        pbar = tqdm(
            dataloader,
            total=total,
            disable=not self.is_primary_process,
        )
        if description is not None and self.is_primary_process:
            pbar.set_description(description)
        return pbar

    def _best_checkpoint_path(self) -> str:
        return os.path.join(self.ckpt_save_dir, f'{self.ckpt_save_name}_best.pkl')

    def _epoch_checkpoint_path(self, epoch: int) -> str:
        return os.path.join(self.ckpt_save_dir, f'{self.ckpt_save_name}_{epoch}.pkl')

    def _early_stop_signal_path(self) -> str:
        return os.path.join(self.ckpt_save_dir, f'{self.ckpt_save_name}.early_stop')

    def _get_validation_monitor_value(self) -> Optional[float]:
        if self.monitor in self._validation_loss:
            keys = [self.monitor]
        elif self.monitor == 'val/loss_sum':
            keys = [
                key for key in self._validation_loss
                if key.startswith('val/') and key.endswith('_loss_sum')
            ]
        else:
            keys = []
        values: List[float] = []
        for key in keys:
            values.extend(float(_get_item(value)) for value in self._validation_loss[key])
        if not values:
            return None
        return sum(values) / len(values)

    def _is_improved(self, metric: float) -> bool:
        if self.best_metric is None:
            return True
        if self.monitor_mode == 'max':
            return metric > self.best_metric + self.min_delta
        return metric < self.best_metric - self.min_delta

    def _handle_epoch_checkpoint(self, epoch: int, metric: Optional[float]) -> bool:
        os.makedirs(self.ckpt_save_dir, exist_ok=True)
        if metric is None:
            checkpoint_path = self._epoch_checkpoint_path(epoch)
            self.model.save(checkpoint_path)
            return False

        improved = self._is_improved(metric)
        if improved:
            self.best_metric = metric
            self.best_epoch = epoch
            self._epochs_without_improvement = 0
            checkpoint_path = self._best_checkpoint_path() if self.save_best_only else self._epoch_checkpoint_path(epoch)
            self.model.save(checkpoint_path)
        else:
            if self.early_stopping_enabled:
                self._epochs_without_improvement += 1
            if not self.save_best_only:
                self.model.save(self._epoch_checkpoint_path(epoch))

        return (
            self.early_stopping_enabled
            and self.early_stopping_patience >= 0
            and self._epochs_without_improvement >= self.early_stopping_patience
        )

    def _write_early_stop_signal(self, should_stop: bool):
        signal_path = self._early_stop_signal_path()
        os.makedirs(self.ckpt_save_dir, exist_ok=True)
        if should_stop:
            with open(signal_path, 'w', encoding='utf-8') as f:
                f.write('stop\n')
        elif os.path.exists(signal_path):
            os.remove(signal_path)

    def _run_validation_and_save_epoch(self, epoch: int):
        self.model.eval()
        validate_dataloader = self.dataset_module.validate_dataloader()
        if validate_dataloader is not None:
            self.on_validation_epoch_start()
            if isinstance(validate_dataloader, dict):
                for name, dataloader in validate_dataloader.items():
                    pbar = self._make_pbar(dataloader)
                    for batch in pbar:
                        self.on_validation_batch_start()
                        loss = self.validation_step(batch)
                        pbar.set_description(f"Epoch {epoch}, Validate {name}, Loss: {_get_item(loss)}")
                        self.on_validation_batch_end()
            else:
                pbar = self._make_pbar(validate_dataloader)
                for batch in pbar:
                    self.on_validation_batch_start()
                    loss = self.validation_step(batch)
                    pbar.set_description(f"Epoch {epoch}, Validate, Loss: {_get_item(loss)}")
                    self.on_validation_batch_end()
            self.on_validation_epoch_end()

        metric = self._get_validation_monitor_value()
        if self.is_primary_process and metric is not None:
            print(f"Epoch {epoch}, {self.monitor}: {metric:.6f}")
        should_stop = self._handle_epoch_checkpoint(epoch, metric)
        if self.is_primary_process and self.best_metric is not None:
            print(
                f"Best {self.monitor}: {self.best_metric:.6f} "
                f"at epoch {self.best_epoch}"
            )
        return should_stop

    def _run_rank0_epoch_end(self, epoch: int):
        if not self.distributed.enabled:
            return self._run_validation_and_save_epoch(epoch)

        @jt.single_process_scope(rank=0)
        def _rank0_only():
            should_stop = self._run_validation_and_save_epoch(epoch)
            self._write_early_stop_signal(should_stop)

        _rank0_only()
        sync_all = getattr(jt, 'sync_all', None)
        if callable(sync_all):
            sync_all()
        return os.path.exists(self._early_stop_signal_path())
    
    def train(self):
        assert self.optimizer is not None, "optimizer is None, cannot train"
        self.model.set_predict(False)
        if self.is_primary_process and os.path.exists(self._early_stop_signal_path()):
            os.remove(self._early_stop_signal_path())
        for epoch in range(self.epochs):
            self.model.train()
            self.on_train_epoch_start()
            train_dataloader = self.dataset_module.train_dataloader()
            assert train_dataloader is not None, "train_dataloader is None"
            pbar = self._make_pbar(train_dataloader) # type: ignore
            for batch in pbar:
                self.on_train_batch_start()
                loss = self.training_step(batch)
                self.optimizer.zero_grad()
                self.optimizer.backward(loss)
                if self.is_primary_process:
                    pbar.set_description(f"Epoch {epoch}, Loss: {_get_item(loss)}")
                self.on_before_optimizer_step(self.optimizer)
                self.optimizer.step()
                self.on_train_batch_end()
            self.on_train_epoch_end()
            should_stop = self._run_rank0_epoch_end(epoch)
            if should_stop:
                if self.is_primary_process:
                    print(
                        "Early stopping triggered "
                        f"after {self._epochs_without_improvement} "
                        "epochs without improvement."
                    )
                break
    
    def predict(self):
        # only iterate once
        self.model.set_predict(True)
        self.model.eval()
        self.on_predict_epoch_start()
        predict_dataloader = self.dataset_module.predict_dataloader()
        assert predict_dataloader is not None, "predict_dataloader is None"
        if not isinstance(predict_dataloader, dict):
            predict_dataloader = {"predict": predict_dataloader}
        for dataloader_name, dataloader in predict_dataloader.items():
            pbar = tqdm(dataloader, total=len(dataloader)//dataloader.batch_size) # type: ignore
            for batch_idx, batch in enumerate(pbar):
                self.on_predict_batch_start()
                output = self.predict_step(batch, batch_idx)
                if self.writer is not None:
                    self.writer.write(batch, output, dataset_module=self.dataset_module)
                pbar.set_description(f"Predicting {dataloader_name}, Batch {batch_idx}")
