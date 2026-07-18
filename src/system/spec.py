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
        initial_best_metric = trainer_config.get('initial_best_metric', None)
        self.best_metric: Optional[float] = (
            None if initial_best_metric is None else float(initial_best_metric)
        )
        initial_best_epoch = trainer_config.get('initial_best_epoch', None)
        self.best_epoch: Optional[int] = (
            None if initial_best_epoch is None else int(initial_best_epoch)
        )
        self._epochs_without_improvement = 0
        
        if optimizer_config is not None and model is not None:
            self.optimizer = get_optimizer(optimizer_config, model)
        else:
            self.optimizer = None

        # cosine LR decay + weight EMA (optional, configured via trainer_config)
        self.lr_schedule = trainer_config.get('lr_schedule', None)   # 'cosine' | None
        self.base_lr = float(getattr(self.optimizer, 'lr', 0.0)) if self.optimizer is not None else 0.0
        self.ema_decay = float(trainer_config.get('ema_decay', 0.0))  # 0 disables EMA
        self._ema = None  # lazy: dict name -> jt.Var of EMA-smoothed params
        configured_ema_decays = trainer_config.get('ema_decays', [])
        self.ema_decays = sorted({float(value) for value in configured_ema_decays})
        if self.ema_decay > 0 and self.ema_decays:
            raise ValueError("configure either ema_decay or ema_decays, not both")
        if any(not 0.0 < value < 1.0 for value in self.ema_decays):
            raise ValueError("every ema_decays value must be in (0, 1)")
        self._ema_states = {}  # decay -> state dict, initialized after first step

        # --- Weights & Biases logging (rank0 only, opt-in via WANDB_API_KEY) ---
        # Disabled unless an API key (or offline mode) is in the env, so manual
        # and predict runs never create stray runs. Initialized lazily in train().
        self._wandb = None
        self._global_step = 0
        self._last_loss_dict = None  # latest train loss_dict (jt.Vars), materialized at log time
        self._wandb_log_every = int(trainer_config.get('wandb_log_every', 20))

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
            self._last_loss_dict = loss_dict  # for wandb per-step component logging
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

    # ----- cosine LR schedule + weight EMA -----
    def _apply_lr(self, epoch: int):
        if self.lr_schedule == 'cosine' and self.optimizer is not None and self.epochs > 1:
            import math
            lr = 0.5 * self.base_lr * (1.0 + math.cos(math.pi * epoch / self.epochs))
            self.optimizer.lr = lr

    def _ema_keys(self):
        return [k for k in self.model.state_dict()
                if 'running_' not in k and 'num_batches' not in k]

    def _update_ema(self):
        if self.ema_decays:
            sd = self.model.state_dict()
            keys = self._ema_keys()
            for decay in self.ema_decays:
                state = self._ema_states.get(decay)
                if state is None:
                    self._ema_states[decay] = {
                        key: sd[key].detach().clone() for key in keys
                    }
                    continue
                for key in state:
                    state[key] = (
                        decay * state[key]
                        + (1.0 - decay) * sd[key].detach()
                    ).detach()
            return
        if self.ema_decay <= 0:
            return
        sd = self.model.state_dict()
        if self._ema is None:
            self._ema = {k: sd[k].detach().clone() for k in self._ema_keys()}
            return
        d = self.ema_decay
        for k in self._ema:
            self._ema[k] = (d * self._ema[k] + (1.0 - d) * sd[k].detach()).detach()

    def _save_model(self, path: str):
        if self.ema_decays and self._ema_states:
            # Keep the raw checkpoint as a strict control, then materialize every
            # EMA from the exact same training trajectory and best-epoch event.
            self.model.save(path)
            cur = self.model.state_dict()
            backup = {key: cur[key].detach().clone() for key in self._ema_keys()}
            stem, suffix = os.path.splitext(path)
            try:
                for decay in self.ema_decays:
                    self.model.load_state_dict(self._ema_states[decay])
                    tag = f"{decay:.6f}".rstrip('0').rstrip('.').replace('.', '')
                    self.model.save(f"{stem}_ema{tag}{suffix}")
            finally:
                self.model.load_state_dict(backup)
            return
        # save EMA-smoothed weights when EMA is enabled (more stable than raw)
        if self.ema_decay > 0 and self._ema is not None:
            cur = self.model.state_dict()
            backup = {k: cur[k].detach().clone() for k in self._ema}
            self.model.load_state_dict(self._ema)
            self.model.save(path)
            self.model.load_state_dict(backup)
        else:
            self.model.save(path)

    def _handle_epoch_checkpoint(self, epoch: int, metric: Optional[float]) -> bool:
        os.makedirs(self.ckpt_save_dir, exist_ok=True)
        if metric is None:
            checkpoint_path = self._epoch_checkpoint_path(epoch)
            self._save_model(checkpoint_path)
            return False

        improved = self._is_improved(metric)
        if improved:
            self.best_metric = metric
            self.best_epoch = epoch
            self._epochs_without_improvement = 0
            checkpoint_path = self._best_checkpoint_path() if self.save_best_only else self._epoch_checkpoint_path(epoch)
            self._save_model(checkpoint_path)
        else:
            if self.early_stopping_enabled:
                self._epochs_without_improvement += 1
            if not self.save_best_only:
                self._save_model(self._epoch_checkpoint_path(epoch))

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
        if self.is_primary_process and self._wandb is not None:
            # per-class validation means (values are already python floats)
            log_data = {
                k: (sum(v) / len(v) if v else 0.0)
                for k, v in self._validation_loss.items()
            }
            log_data['epoch'] = epoch
            log_data['val/lr'] = float(getattr(self.optimizer, 'lr', 0.0))
            if metric is not None:
                log_data[self.monitor] = metric  # e.g. averaged val/loss_sum
            if self.best_metric is not None:
                log_data['val/best_metric'] = self.best_metric
                log_data['val/best_epoch'] = self.best_epoch
            log_data['val/epochs_without_improvement'] = self._epochs_without_improvement
            self._wandb_log(log_data)
        return should_stop

    def _run_rank0_epoch_end(self, epoch: int):
        if not self.distributed.enabled:
            return self._run_validation_and_save_epoch(epoch)

        # Only rank0 runs validation + checkpointing; other ranks skip the body.
        stop_flag = {'v': 0}
        @jt.single_process_scope(rank=0)
        def _rank0_only():
            should_stop = self._run_validation_and_save_epoch(epoch)
            self._write_early_stop_signal(should_stop)
            stop_flag['v'] = 1 if should_stop else 0

        _rank0_only()

        # Broadcast rank0's stop decision to ALL ranks via an MPI collective.
        # A filesystem signal alone is unreliable: if a non-zero rank misses it
        # (cross-process visibility race), that rank marches into the next epoch
        # and deadlocks at the gradient all-reduce while rank0 has already exited
        # the loop. Every rank reaches this all-reduce; only rank0 contributes a
        # non-zero flag, so all ranks agree and break in lockstep.
        flag = jt.array([float(stop_flag['v'])]).float32().mpi_all_reduce("add")
        sync_all = getattr(jt, 'sync_all', None)
        if callable(sync_all):
            sync_all()
        return bool(flag.item() >= 0.5)
    
    # ----- Weights & Biases logging -----
    def _init_wandb(self):
        """Start a wandb run on the primary process. No-op unless WANDB_API_KEY
        (or an offline WANDB_MODE) is set, so manual/predict runs stay clean.
        Any failure disables wandb instead of crashing training.

        Run naming derives from the checkpoint dir (e.g. ``experiments/spcfgfn_cvm``
        -> name ``spcfgfn_cvm``, group ``spcfgfn``, stage ``cvm``) so the three
        StraightPCF stages of one experiment land under a single wandb group.
        """
        if not self.is_primary_process:
            return
        if not (os.environ.get('WANDB_API_KEY')
                or os.environ.get('WANDB_MODE') in ('offline', 'dryrun')):
            return
        try:
            import wandb
        except Exception as exc:  # noqa: BLE001 - logging must never break training
            print(f"[wandb] unavailable ({exc}); training without logging")
            return

        name = os.environ.get('WANDB_RUN_NAME') \
            or os.path.basename(self.ckpt_save_dir.rstrip('/')) or 'run'
        group = os.environ.get('WANDB_RUN_GROUP')
        stage = None
        for suf in ('_cvm', '_spcf', '_vm'):  # _cvm before _vm (longest match first)
            if name.endswith(suf):
                stage = suf[1:]
                if group is None:
                    group = name[:-len(suf)]
                break
        if group is None:
            group = name

        config = {
            'epochs': self.epochs,
            'base_lr': self.base_lr,
            'lr_schedule': self.lr_schedule,
            'ema_decay': self.ema_decay,
            'ema_decays': list(self.ema_decays),
            'loss_config': dict(self.loss_config) if self.loss_config else None,
            'world_size': self.distributed.world_size,
            'save_best_only': self.save_best_only,
            'early_stopping': self.early_stopping_enabled,
            'monitor': self.monitor,
            'patience': self.early_stopping_patience,
            'stage': stage,
            'ckpt_save_dir': self.ckpt_save_dir,
        }
        try:
            self._wandb = wandb.init(
                project=os.environ.get('WANDB_PROJECT', 'Track2'),
                name=name,
                group=group,
                job_type=stage or 'train',
                config=config,
                resume='allow',
                id=os.environ.get('WANDB_RUN_ID') or None,
            )
            wandb.define_metric('train/global_step')
            wandb.define_metric('train/*', step_metric='train/global_step')
            wandb.define_metric('epoch')
            wandb.define_metric('val/*', step_metric='epoch')
            print(f"[wandb] logging project={config['ckpt_save_dir']} "
                  f"run={name} group={group}")
        except Exception as exc:  # noqa: BLE001
            print(f"[wandb] init failed ({exc}); training without logging")
            self._wandb = None

    def _wandb_log(self, data: Dict):
        if self._wandb is None:
            return
        try:
            self._wandb.log(data)
        except Exception:  # noqa: BLE001 - never let logging crash training
            pass

    def _wandb_finish(self):
        if self._wandb is None:
            return
        try:
            self._wandb.finish()
        except Exception:  # noqa: BLE001
            pass
        self._wandb = None

    def train(self):
        assert self.optimizer is not None, "optimizer is None, cannot train"
        self._init_wandb()
        self.model.set_predict(False)
        if self.is_primary_process and os.path.exists(self._early_stop_signal_path()):
            os.remove(self._early_stop_signal_path())
        for epoch in range(self.epochs):
            self.model.train()
            self._apply_lr(epoch)
            self.on_train_epoch_start()
            train_dataloader = self.dataset_module.train_dataloader()
            assert train_dataloader is not None, "train_dataloader is None"
            pbar = self._make_pbar(train_dataloader) # type: ignore
            # Reading loss.item() every step forces a host<->device sync that
            # serializes Jittor's async pipeline; only refresh the bar periodically.
            log_every = 20
            for step, batch in enumerate(pbar):
                self.on_train_batch_start()
                loss = self.training_step(batch)
                self.optimizer.zero_grad()
                self.optimizer.backward(loss)
                self.on_before_optimizer_step(self.optimizer)
                self.optimizer.step()
                self._update_ema()
                if self.is_primary_process and step % log_every == 0:
                    loss_val = _get_item(loss)
                    pbar.set_description(f"Epoch {epoch}, Loss: {loss_val}")
                    if self._wandb is not None:
                        log_data = {
                            'train/loss_sum': loss_val,
                            'train/lr': float(getattr(self.optimizer, 'lr', 0.0)),
                            'train/epoch': epoch,
                            'train/global_step': self._global_step,
                        }
                        if self._last_loss_dict is not None:
                            for k, v in self._last_loss_dict.items():
                                if k == 'loss_sum':
                                    continue
                                log_data[f'train/{k}'] = _get_item(v)
                        self._wandb_log(log_data)
                self.on_train_batch_end()
                self._global_step += 1
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
        self._wandb_finish()
    
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
