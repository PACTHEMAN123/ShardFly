"""Expose the TP-local weights of a Megatron-FSDP model during publication."""

from contextlib import ExitStack, contextmanager


def _fsdp_module(model):
    # Megatron Core wraps its FSDP implementation in the DDP adapter.
    while model is not None:
        if (
            hasattr(model, "all_gather_pipeline")
            and callable(getattr(model, "synchronize_param_gather", None))
            and getattr(model, "data_parallel_sharding_strategy", "no_shard") != "no_shard"
        ):
            return model
        model = getattr(model, "module", None)
    return None


class MegatronLoadSource:
    """Borrow the framework's model weights for the lifetime of a loader update."""

    def __init__(self, models):
        self._fsdp_modules = tuple(
            module for model in models if (module := _fsdp_module(model)) is not None
        )

    @property
    def needs_rebinding(self):
        return bool(self._fsdp_modules)

    @contextmanager
    def publication(self):
        with ExitStack() as cleanup:
            for module in self._fsdp_modules:
                cleanup.callback(module.synchronize_param_gather)
                module.start_param_sync(force_sync=True)
            yield
