import torch.nn as nn
from transformers import Seq2SeqTrainer, Trainer
from transformers.trainer import *
from fairscale.optim import OSS


class T5EncoderTrainer(Trainer):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sharded_ddp = kwargs.get('sharded_ddp', None)
        
    def init_hyper(self, lr, lr_cls):
        self.lr = lr
        self.lr_proj = lr_cls
        
    
    def create_optimizer(self):
        if self.optimizer is None:
            decay_parameters = get_parameter_names(self.model, [nn.LayerNorm])
            decay_parameters = [name for name in decay_parameters if "bias" not in name]
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in self.model.named_parameters() if n in decay_parameters and "_projector" not in n],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [p for n, p in self.model.named_parameters() if n not in decay_parameters and "_projector" not in n],
                    "weight_decay": 0.0,
                },
            ]
            optimizer_grouped_parameters += [
                {
                    "params": [p for n, p in self.model.named_parameters() if n in decay_parameters and "_projector" in n],
                    "lr": self.lr_proj,
                    "weight_decay": self.args.weight_decay
                },
                {
                    "params": [p for n, p in self.model.named_parameters() if n not in decay_parameters and "_projector" in n],
                    "lr": self.lr_proj,
                    "weight_decay": 0.0
                }
            ]
            
            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)

            # if self.sharded_ddp == ShardedDDPOption.SIMPLE:
            if self.sharded_ddp and self.sharded_ddp == ShardedDDPOption.SIMPLE:
                self.optimizer = OSS(
                    params=optimizer_grouped_parameters,
                    optim=optimizer_cls,
                    **optimizer_kwargs,
                )
            else:
                self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

        if is_sagemaker_mp_enabled():
            import smdistributed.modelparallel.torch as smp
            self.optimizer = smp.DistributedOptimizer(self.optimizer)

        return self.optimizer