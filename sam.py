from transformers import (
    SamVisionConfig,
    SamPromptEncoderConfig,
    SamMaskDecoderConfig,
    SamModel,
)
from transformers import SamConfig

configuration = SamConfig()

model = SamModel(configuration)

configuration = model.config
vision_config = SamVisionConfig()
prompt_encoder_config = SamPromptEncoderConfig()
mask_decoder_config = SamMaskDecoderConfig()

config = SamConfig(vision_config, prompt_encoder_config, mask_decoder_config)
print(model)
print(config)
