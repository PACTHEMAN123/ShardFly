"""Opt-in vLLM checkpoint loader that captures ShardFly receive layouts."""

from copy import copy
from functools import cache, partial

from shardfly.integrations.loader_capture import CAPTURE_ATTRIBUTE, WeightLoadCapture


def _prepare_attention_refresh(derived_names, *, model, model_config):
    from vllm.model_executor.model_loader import utils

    is_deferred = getattr(utils, "is_deferred_attention_layer", None)
    if is_deferred is None:
        attention_classes = (utils.Attention, utils.MLAAttention, utils.MMEncoderAttention)

        def is_deferred(module):
            return isinstance(module, attention_classes) and hasattr(
                module, "process_weights_after_loading"
            )

    modules = dict(model.named_modules())
    owners = {name.rpartition(".")[0] for name in derived_names}
    unsupported = sorted(name for name in owners if not is_deferred(modules[name]))
    if unsupported:
        raise ValueError(f"Loader added parameters outside deferred attention: {unsupported[:5]}")
    owners = sorted(owners)
    dtype = model_config.dtype

    def refresh(target):
        for name in owners:
            target.get_submodule(name).process_weights_after_loading(dtype)

    return refresh


@cache
def register_capture_loader():
    from vllm.model_executor import model_loader

    # Older vLLM versions can still use the standard ShardFly plugin.
    register = getattr(model_loader, "register_model_loader", None)
    if register is None:
        return

    @register("shardfly_capture")
    class CaptureModelLoader(model_loader.DefaultModelLoader):
        def __init__(self, load_config):
            load_config = copy(load_config)
            load_config.load_format = "auto"
            super().__init__(load_config)
            self._capture = None

        def load_model(self, vllm_config, model_config, *args, **kwargs):
            if model_config.quantization is not None:
                raise ValueError("shardfly_capture requires an unquantized vLLM model")
            if vllm_config.speculative_config is not None:
                raise ValueError("shardfly_capture does not support speculative models")
            with WeightLoadCapture() as capture:
                self._capture = capture
                try:
                    model = super().load_model(
                        vllm_config, model_config, *args, **kwargs
                    )
                finally:
                    self._capture = None
            plan = capture.finish(
                model,
                prepare_refresh=partial(
                    _prepare_attention_refresh, model=model, model_config=model_config
                ),
            )
            if getattr(getattr(model_config, "hf_config", None), "tie_word_embeddings", False):
                from vllm.distributed import get_pp_group

                if get_pp_group().is_last_rank:
                    plan = plan.with_checkpoint_alias(
                        "lm_head.weight", "model.embed_tokens.weight"
                    )
            setattr(model, CAPTURE_ATTRIBUTE, plan)
            return model

        def load_weights(self, model, model_config):
            if self._capture is None:
                return super().load_weights(model, model_config)
            self._capture.start(model)
            super().load_weights(model, model_config)
            self._capture.end()

        def get_all_weights(self, model_config, model):
            weights = super().get_all_weights(model_config, model)
            if self._capture is not None:
                return self._capture.weights(weights)
            return weights
