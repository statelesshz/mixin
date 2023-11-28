from collections import OrderedDict

from .auto_factory import _HZÅLazyAutoMapping

HZ_MODEL_FOR_CAUSAL_LM_MAPPING_NAMES  = OrderedDict(
    [
        ("baichuan", "BaiChuanForCausalLM")
    ]
)

HZ_MODEL_FRO_CASUAL_LM_MAPPING = _HZÅLazyAutoMapping(HZ_CONFIG_MAPPING_NAME, HZ_MODEL_FOR_CAUSAL_LM_MAPPING_NAMES)


class HZAutoModelForCasualLM(_BaseAutoModelClass): #
    _model_mapping = HZ_MODEL_FOR_CAUSAL_LM_MAPPING_NAMES