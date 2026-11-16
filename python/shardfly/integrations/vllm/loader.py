"""Opt-in vLLM checkpoint loader that captures ShardFly receive layouts."""

from copy import copy

from shardfly.integrations.loader_capture import CAPTURE_ATTRIBUTE, WeightLoadCapture


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
            setattr(model, CAPTURE_ATTRIBUTE, capture.finish(model))
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
