from collections import OrderedDict

from .configuration_auto import PX_CONFIG_MAPPING_NAMES
from .auto_factory import _PXLazyAutoMapping, _PXBaseAutoModelClass

PX_MODEL_FOR_CAUSAL_LM_MAPPING_NAMES = OrderedDict(
    [
        ("chatglm", "ChatGLMForConditionalGeneration")
    ]
)

PX_MODEL_FRO_CASUAL_LM_MAPPING = _PXLazyAutoMapping(PX_CONFIG_MAPPING_NAMES, PX_MODEL_FOR_CAUSAL_LM_MAPPING_NAMES)


class PXAutoModelForCasualLM(_PXBaseAutoModelClass):  #
    _model_mapping = PX_MODEL_FRO_CASUAL_LM_MAPPING

backend = None

class AutoModelForCasualLM:
    def __init__(self, *args, **kwargs):
        raise EnvironmentError(
            f"{self.__class__.__name__} is designed to be instantiated "
            f"using the `{self.__class__.__name__}.from_pretrained(pretrained_model_name_or_path)` or "
            f"`{self.__class__.__name__}.from_config(config)` methods."
        )

    @classmethod
    def from_config(cls, config, **kwargs):
        # dispatch
        if backend == "torch":
            pass
        elif backend == "mindspore":
            pass

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        # dispatch
        if backend == "torch":
            pass
        elif backend == "mindspore":
            pass
