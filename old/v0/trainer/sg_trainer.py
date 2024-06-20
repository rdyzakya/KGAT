from transformers import Trainer
# from trl import SFTTrainer
from transformers.integrations import deepspeed_init
from transformers.utils import logging, is_torch_tpu_available
from transformers.trainer_utils import has_length, denumpify_detensorize, EvalPrediction, EvalLoopOutput
from transformers.trainer_pt_utils import find_batch_size, nested_concat, nested_numpify, IterableDatasetShard
from torch.nn import (
    BCELoss,
    BCEWithLogitsLoss
)
import torch
# import torch_xla.core.xla_model as xm
import numpy as np

logger = logging.get_logger(__name__)

class SGTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.criterion = BCEWithLogitsLoss(reduction="mean")
    
    def compute_loss(self, model, inputs, return_outputs=False):
        # graph_query = tokenizer(batch[0], padding=True, truncation=True, max_length=64, return_tensors="pt")
        # entities = tokenizer(batch[1], padding=True, truncation=True, max_length=16, return_tensors="pt")
        # relations = tokenizer(batch[2], padding=True, truncation=True, max_length=16, return_tensors="pt")
        # x_coo = batch[3]
        # node_batch = batch[4]
        # y_coo = batch[5]

        # text, entities, relations, x_coo, y_coo_cls
        labels = inputs.pop("y_coo_cls")
        mean_fused_score, _, _ = model(**inputs)

        assert len(mean_fused_score) == len(labels)

        mask = labels != -1

        mean_fused_score = mean_fused_score[mask]
        labels = labels[mask]

        labels = labels.float()

        loss = self.criterion(mean_fused_score, labels)

        if return_outputs:
            return loss, mean_fused_score
        return loss
    
    def prediction_step(
        self,
        model,
        inputs,
        prediction_loss_only,
        ignore_keys = None,
    ):
        # has_labels = False if len(self.label_names) == 0 else all(inputs.get(k) is not None for k in self.label_names)
        # # For CLIP-like models capable of returning loss values.
        # # If `return_loss` is not specified or being `None` in `inputs`, we check if the default value of `return_loss`
        # # is `True` in `model.forward`.
        # return_loss = inputs.get("return_loss", None)
        # if return_loss is None:
        #     return_loss = self.can_return_loss
        # loss_without_labels = True if len(self.label_names) == 0 and return_loss else False

        # inputs = self._prepare_inputs(inputs)
        # if ignore_keys is None:
        #     if hasattr(self.model, "config"):
        #         ignore_keys = getattr(self.model.config, "keys_to_ignore_at_inference", [])
        #     else:
        #         ignore_keys = []

        # labels may be popped when computing the loss (label smoothing for instance) so we grab them first.
        # if has_labels or loss_without_labels:
        #     labels = nested_detach(tuple(inputs.get(name) for name in self.label_names))
        #     if len(labels) == 1:
        #         labels = labels[0]
        # else:
        #     labels = None
        labels = inputs.pop("y_coo_cls")

        mask = labels != -1
        labels = labels[mask]

        labels = labels.float()


        with torch.no_grad():
            mean_fused_score, _, _ = model(**inputs)
            # if is_sagemaker_mp_enabled():
            #     raw_outputs = smp_forward_only(model, inputs)
            #     if has_labels or loss_without_labels:
            #         if isinstance(raw_outputs, dict):
            #             loss_mb = raw_outputs["loss"]
            #             logits_mb = tuple(v for k, v in raw_outputs.items() if k not in ignore_keys + ["loss"])
            #         else:
            #             loss_mb = raw_outputs[0]
            #             logits_mb = raw_outputs[1:]

            #         loss = loss_mb.reduce_mean().detach().cpu()
            #         logits = smp_nested_concat(logits_mb)
            #     else:
            #         loss = None
            #         if isinstance(raw_outputs, dict):
            #             logits_mb = tuple(v for k, v in raw_outputs.items() if k not in ignore_keys)
            #         else:
            #             logits_mb = raw_outputs
            #         logits = smp_nested_concat(logits_mb)
            # else:
            #     if has_labels or loss_without_labels:
            #         with self.compute_loss_context_manager():
            #             loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
            #         loss = loss.mean().detach()

            #         if isinstance(outputs, dict):
            #             logits = tuple(v for k, v in outputs.items() if k not in ignore_keys + ["loss"])
            #         else:
            #             logits = outputs[1:]
            #     else:
            #         loss = None
            #         with self.compute_loss_context_manager():
            #             outputs = model(**inputs)
            #         if isinstance(outputs, dict):
            #             logits = tuple(v for k, v in outputs.items() if k not in ignore_keys)
            #         else:
            #             logits = outputs
            #         # TODO: this needs to be fixed and made cleaner later.
            #         if self.args.past_index >= 0:
            #             self._past = outputs[self.args.past_index - 1]
        loss = self.criterion(mean_fused_score, labels)


        if prediction_loss_only:
            return (loss, None, None)

        # logits = nested_detach(logits)
        # if len(logits) == 1:
        #     logits = logits[0]

        return (loss, mean_fused_score, labels)
    
    def evaluation_loop(
        self,
        dataloader,
        description,
        prediction_loss_only,
        ignore_keys,
        metric_key_prefix = "eval",
    ):
        """
        Prediction/evaluation loop, shared by `Trainer.evaluate()` and `Trainer.predict()`.

        Works both with or without labels.
        """
        args = self.args

        prediction_loss_only = prediction_loss_only if prediction_loss_only is not None else args.prediction_loss_only

        # if eval is called w/o train, handle model prep here
        if self.is_deepspeed_enabled and self.deepspeed is None:
            _, _ = deepspeed_init(self, num_training_steps=0, inference=True)

        model = self._wrap_model(self.model, training=False, dataloader=dataloader)

        if len(self.accelerator._models) == 0 and model is self.model:
            model = (
                self.accelerator.prepare(model)
                if self.is_deepspeed_enabled
                else self.accelerator.prepare_model(model, evaluation_mode=True)
            )

            if self.is_fsdp_enabled:
                self.model = model

            # for the rest of this function `model` is the outside model, whether it was wrapped or not
            if model is not self.model:
                self.model_wrapped = model

            # backward compatibility
            if self.is_deepspeed_enabled:
                self.deepspeed = self.model_wrapped

        # if full fp16 or bf16 eval is wanted and this ``evaluation`` or ``predict`` isn't called
        # while ``train`` is running, cast it to the right dtype first and then put on device
        if not self.is_in_train:
            if args.fp16_full_eval:
                model = model.to(dtype=torch.float16, device=args.device)
            elif args.bf16_full_eval:
                model = model.to(dtype=torch.bfloat16, device=args.device)

        batch_size = self.args.eval_batch_size

        logger.info(f"***** Running {description} *****")
        if has_length(dataloader):
            logger.info(f"  Num examples = {self.num_examples(dataloader)}")
        else:
            logger.info("  Num examples: Unknown")
        logger.info(f"  Batch size = {batch_size}")

        model.eval()

        self.callback_handler.eval_dataloader = dataloader
        # Do this before wrapping.
        eval_dataset = getattr(dataloader, "dataset", None)

        if args.past_index >= 0:
            self._past = None

        # Initialize containers
        # losses/preds/labels on GPU/TPU (accumulated for eval_accumulation_steps)
        losses_host = None
        preds_host = None
        labels_host = None
        inputs_host = None

        # losses/preds/labels on CPU (final containers)
        all_losses = None
        all_preds = None
        all_labels = None
        all_inputs = None
        # Will be useful when we have an iterable dataset so don't know its length.

        observed_num_examples = 0
        # Main evaluation loop
        for step, inputs in enumerate(dataloader):
            # Update the observed num examples
            observed_batch_size = find_batch_size(inputs)
            if observed_batch_size is not None:
                observed_num_examples += observed_batch_size
                # For batch samplers, batch_size is not known by the dataloader in advance.
                if batch_size is None:
                    batch_size = observed_batch_size

            # Prediction step
            loss, logits, labels = self.prediction_step(model, inputs, prediction_loss_only, ignore_keys=ignore_keys)
            main_input_name = getattr(self.model, "main_input_name", "input_ids")
            inputs_decode = self._prepare_input(inputs[main_input_name]) if args.include_inputs_for_metrics else None

            # if is_torch_tpu_available():
            #     xm.mark_step()

            # Update containers on host
            if loss is not None:
                losses = self.gather_function((loss.repeat(batch_size)))
                losses_host = losses if losses_host is None else nested_concat(losses_host, losses, padding_index=-100)
            if labels is not None:
                # labels = self.accelerator.pad_across_processes(labels, dim=1, pad_index=-100)
                pass
            if inputs_decode is not None: # NOTE : NOT NEEDED YET
                inputs_decode = self.accelerator.pad_across_processes(inputs_decode, dim=1, pad_index=-100)
                inputs_decode = self.gather_function((inputs_decode))
                inputs_host = (
                    inputs_decode
                    if inputs_host is None
                    else nested_concat(inputs_host, inputs_decode, padding_index=-100)
                )
            if logits is not None:
                logits = self.accelerator.pad_across_processes(logits, dim=1, pad_index=-100)
                if self.preprocess_logits_for_metrics is not None:
                    logits = self.preprocess_logits_for_metrics(logits, labels)
                logits = self.gather_function((logits))
                # preds_host = logits if preds_host is None else nested_concat(preds_host, logits, padding_index=-100)
                preds_host = logits if preds_host is None else np.concatenate((preds_host, logits))

            if labels is not None:
                labels = self.gather_function((labels))
                # labels_host = labels if labels_host is None else nested_concat(labels_host, labels, padding_index=-100)
                labels_host = labels if labels_host is None else np.concatenate((labels_host, labels))

            self.control = self.callback_handler.on_prediction_step(args, self.state, self.control)

            # Gather all tensors and put them back on the CPU if we have done enough accumulation steps.
            if args.eval_accumulation_steps is not None and (step + 1) % args.eval_accumulation_steps == 0:
                if losses_host is not None:
                    losses = nested_numpify(losses_host)
                    all_losses = losses if all_losses is None else np.concatenate((all_losses, losses), axis=0)
                if preds_host is not None:
                    logits = nested_numpify(preds_host)
                    # all_preds = logits if all_preds is None else nested_concat(all_preds, logits, padding_index=-100)
                    all_preds = logits if all_preds is None else np.concatenate((all_preds, logits))
                if inputs_host is not None:
                    inputs_decode = nested_numpify(inputs_host)
                    all_inputs = (
                        inputs_decode
                        if all_inputs is None
                        else nested_concat(all_inputs, inputs_decode, padding_index=-100)
                    )
                if labels_host is not None:
                    labels = nested_numpify(labels_host)
                    # all_labels = (
                    #     labels if all_labels is None else nested_concat(all_labels, labels, padding_index=-100)
                    # )
                    all_labels = (
                        labels if all_labels is None else np.concatenate((all_labels, labels))
                    )

                # Set back to None to begin a new accumulation
                losses_host, preds_host, inputs_host, labels_host = None, None, None, None

        # After all calls to `.gather_function`, reset to `gather_for_metrics`:
        self.gather_function = self.accelerator.gather_for_metrics
        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of the evaluation loop
            delattr(self, "_past")

        # Gather all remaining tensors and put them back on the CPU
        if losses_host is not None:
            losses = nested_numpify(losses_host)
            all_losses = losses if all_losses is None else np.concatenate((all_losses, losses), axis=0)
        if preds_host is not None:
            logits = nested_numpify(preds_host)
            # all_preds = logits if all_preds is None else nested_concat(all_preds, logits, padding_index=-100)
            all_preds = logits if all_preds is None else np.concatenate((all_preds, logits))
        if inputs_host is not None: # NOTE : NOT NEEDED YET
            inputs_decode = nested_numpify(inputs_host)
            all_inputs = (
                inputs_decode if all_inputs is None else nested_concat(all_inputs, inputs_decode, padding_index=-100)
            )
        if labels_host is not None:
            labels = nested_numpify(labels_host)
            # all_labels = labels if all_labels is None else nested_concat(all_labels, labels, padding_index=-100)
            all_labels = labels if all_labels is None else np.concatenate((all_labels, labels))

        # Number of samples
        if has_length(eval_dataset):
            num_samples = len(eval_dataset)
        # The instance check is weird and does not actually check for the type, but whether the dataset has the right
        # methods. Therefore we need to make sure it also has the attribute.
        elif isinstance(eval_dataset, IterableDatasetShard) and getattr(eval_dataset, "num_examples", 0) > 0:
            num_samples = eval_dataset.num_examples
        else:
            if has_length(dataloader):
                num_samples = self.num_examples(dataloader)
            else:  # both len(dataloader.dataset) and len(dataloader) fail
                num_samples = observed_num_examples
        if num_samples == 0 and observed_num_examples > 0:
            num_samples = observed_num_examples

        # Metrics!
        if self.compute_metrics is not None and all_preds is not None and all_labels is not None:
            if args.include_inputs_for_metrics:
                metrics = self.compute_metrics(
                    EvalPrediction(predictions=all_preds, label_ids=all_labels, inputs=all_inputs)
                )
            else:
                metrics = self.compute_metrics(EvalPrediction(predictions=all_preds, label_ids=all_labels))
        else:
            metrics = {}

        # To be JSON-serializable, we need to remove numpy types or zero-d tensors
        metrics = denumpify_detensorize(metrics)

        if all_losses is not None:
            metrics[f"{metric_key_prefix}_loss"] = all_losses.mean().item()
        if hasattr(self, "jit_compilation_time"):
            metrics[f"{metric_key_prefix}_jit_compilation_time"] = self.jit_compilation_time

        # Prefix all keys with metric_key_prefix + '_'
        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

        return EvalLoopOutput(predictions=all_preds, label_ids=all_labels, metrics=metrics, num_samples=num_samples)