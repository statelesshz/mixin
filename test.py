from mixin import PXAutoConfig, PXAutoModelForCasualLM

config = PXAutoConfig.from_pretrained("/Users/yun/github/chatglm/config.json")
print(f'>>>config={config}', flush=True)

model = PXAutoModelForCasualLM.from_config(config)
print(f'>>>model={model}', flush=True)
