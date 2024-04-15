from torch import nn
from torch.nn import CrossEntropyLoss
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers import Trainer
from transformers.trainer import *

from transformers.models.t5.modeling_t5 import T5EncoderModel, T5Config
from transformers.models.mt5.modeling_mt5 import MT5EncoderModel, MT5Config

MODELS = {
    "t5": {
        "encoder": T5EncoderModel,
        "config": T5Config
    },
    "mt5": {
        "encoder": MT5EncoderModel,
        "config": MT5Config
    }
}

MODEL = MODELS["mt5"]

class ClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        classifier_dropout = config.classifier_dropout
        self.dropout = nn.Dropout(classifier_dropout)
        self.out_projector = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, hidden_states, **kwargs):
        hidden_states = hidden_states[:, 0, :]  # take <s> token (equiv. to [CLS])
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.out_projector(hidden_states)
        return hidden_states


class SequenceClassificationConfig(MODEL["config"]):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_additional_args(self, num_labels, classifier_dropout=0.2):
        self.num_labels = num_labels
        self.classifier_dropout = classifier_dropout


class SequenceClassificationEncoder(MODEL["encoder"]):
    def __init__(self, config: SequenceClassificationConfig):
        super().__init__(config)

        self.num_labels = config.num_labels
        self.classifier = ClassificationHead(config)

    def freeze_encoder(self, signal:bool):
        for param in self.encoder.parameters():
            param.requires_grad = signal

    def freeze_projector(self, signal:bool):
        for param in self.classifier.parameters():
            param.requires_grad = signal

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_hidden_states=None,
        output_attentions=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = encoder_outputs[0]
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + encoder_outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


class SequenceClassificationTrainer(Trainer):
    def init_hyper(self, lr, lr_cls):
        self.lr = lr
        self.lr_cls = lr_cls

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
                }
            ]
            optimizer_grouped_parameters += [
                {
                    "params": [p for n, p in self.model.named_parameters() if n in decay_parameters and "_projector" in n],
                    "lr": self.lr_cls,
                    "weight_decay": self.args.weight_decay
                },
                {
                    "params": [p for n, p in self.model.named_parameters() if n not in decay_parameters and "_projector" in n],
                    "lr": self.lr_cls,
                    "weight_decay": 0.0
                }
            ]

            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)

            if self.sharded_ddp == ShardedDDPOption.SIMPLE:
                self.optimizer = OSS(
                    params=optimizer_grouped_parameters,
                    optim=optimizer_cls,
                    **optimizer_kwargs,
                )
            else:
                self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

        if is_sagemaker_mp_enabled():
            self.optimizer = smp.DistributedOptimizer(self.optimizer)

        return self.optimizer
