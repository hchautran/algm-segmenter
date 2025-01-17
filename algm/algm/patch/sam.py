from typing import Tuple
import math
import torch
import torch.nn as nn
from transformers import SamModel
from transformers.models.sam.modeling_sam import (
    SamVisionEncoder,  
    SamVisionLayer, 
    SamVisionAttention,
    SamVisionEncoderOutput,
    ModelOutput,
    dataclass

)
from algm.local_merge import conditional_pooling, merge_source, merge_wavg
from algm.global_merge import turbo_matching
from algm.utils import parse_r
from typing import Optional,Union


@dataclass
class TurboSamVisionEncoderOutput(ModelOutput):
    image_embeds: Optional[torch.FloatTensor] = None
    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    all_q: Optional[Tuple[torch.FloatTensor, ...]] = None
    all_k: Optional[Tuple[torch.FloatTensor, ...]] = None
    all_v: Optional[Tuple[torch.FloatTensor, ...]] = None



class TurboSamVisionLayer(SamVisionLayer):
    """
    Modifications:
     - Apply ALGM between the attention and mlp blocks
    """


    def forward(
        self, 
        hidden_states: torch.Tensor, 
        output_attentions: Optional[bool]=False, 
        output_qkv: Optional[bool]=False
    ) -> torch.Tensor:
        residual = hidden_states


        hidden_states =  self.layer_norm1(hidden_states) 

        if self.window_size > 0:
            #reverse window partition
            height, width = hidden_states.shape[1], hidden_states.shape[2]
            hidden_states, padding_shape = self.window_partition(hidden_states, self.window_size)
        
        if  output_qkv: 
            hidden_states, attn_weights, q, k, v= self.attn(
                hidden_states=hidden_states,
                output_qkv=output_qkv,
                output_attentions=output_attentions,
            )

        #reverse window partition
        if self.window_size > 0:
            hidden_states =  self.window_unpartition(hidden_states, self.window_size, padding_shape, (height, width))
        
      
        hidden_states = residual + hidden_states
        layernorm_output = self.layer_norm2(hidden_states)
        hidden_states = hidden_states + self.mlp(layernorm_output)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)
        if output_qkv:
            outputs += (q,)
            outputs += (k,)
            outputs += (v,)

        return outputs


class TurboSamVisionAttention(SamVisionAttention):

    def forward(self, hidden_states: torch.Tensor, output_attentions=False, output_qkv=False) -> torch.Tensor:
        batch_size, height, width, _ = hidden_states.shape
        # qkv with shape (3, batch_size, nHead, height * width, channel)
        qkv = (
            self.qkv(hidden_states)
            .reshape(batch_size, height * width, 3, self.num_attention_heads, -1)
            .permute(2, 0, 3, 1, 4)
        )
        # q, k, v with shape (batch_size * nHead, height * width, channel)
        query, key, value = qkv.reshape(3, batch_size * self.num_attention_heads, height * width, -1).unbind(0)

        attn_weights = (query * self.scale) @ key.transpose(-2, -1)

        if self.use_rel_pos:
            attn_weights = self.add_decomposed_rel_pos(
                attn_weights, query, self.rel_pos_h, self.rel_pos_w, (height, width), (height, width)
            )

        attn_weights = torch.nn.functional.softmax(attn_weights, dtype=torch.float32, dim=-1).to(query.dtype)

        attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_output = (attn_probs @ value).reshape(batch_size, self.num_attention_heads, height, width, -1)
        attn_output = attn_output.permute(0, 2, 3, 1, 4).reshape(batch_size, height, width, -1)

        attn_output = self.proj(attn_output)

        if output_attentions and not output_qkv:
            outputs = (attn_output, attn_weights, None, None, None)
        elif not output_attentions and output_qkv:
            outputs = (attn_output, None, query, key, value)
        elif output_qkv and output_attentions:
            outputs = (attn_output, attn_weights, query, key, value)
        else:
            outputs = (attn_output, None, None, None, None)

        return outputs


        
class TurboSamVisionEncoder(SamVisionEncoder):


    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_qkv: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SamVisionEncoderOutput]:
        self._turbo_info["size"] = None
        self._turbo_info["source"] = None
        self._turbo_info["rel_pos"] = None
        self._turbo_info["selected_layers"] = list(self.selected_layers)
        self._turbo_info["window_size"] = self.window_size
        self._turbo_info["threshold"] = self.threshold
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        hidden_states = self.patch_embed(pixel_values)
        if self.pos_embed is not None:
            hidden_states = hidden_states + self.pos_embed

        all_hidden_states = () if output_hidden_states else None
        all_q = () if output_qkv else None
        all_k = () if output_qkv else None
        all_v = () if output_qkv else None
        all_self_attentions = () if output_attentions else None

        for i, layer_module in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    layer_module.__call__,
                    hidden_states,
                )
            else:
                layer_outputs = layer_module(hidden_states, output_attentions=output_attentions, output_qkv=output_qkv)

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
            if output_qkv:
                all_q = all_q + (layer_outputs[2],)
                all_k = all_k + (layer_outputs[3],)
                all_v = all_v + (layer_outputs[4],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        hidden_states = self.neck(hidden_states)

        if not return_dict:
            outputs = (hidden_states,)
            if output_hidden_states:
                outputs = outputs + (all_hidden_states,)
            if output_attentions:
                outputs = outputs + (all_self_attentions,)
            if output_qkv:
                outputs = outputs + (all_q, all_k, all_v)
            return outputs

        return TurboSamVisionEncoderOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            all_q=all_q,
            all_k=all_k,
            all_v=all_v,
        )



class TurboSamModel(SamModel): 
    @torch.no_grad()
    def get_image_output(
        self, 
        pixel_values, 
        output_attentions = None, 
        output_hidden_states = None, 
        output_qkv = None,
        return_dict = None
    ):
        vision_output = self.vision_encoder(
            pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            output_qkv=output_qkv,
            return_dict=return_dict,
        )
        return vision_output 

def apply_patch(
    model: SamModel, selected_layers: list, trace_source: bool = False, prop_attn: bool = True, 
):

    model.__class__ = TurboSamModel
    model.vision_encoder.__class__ = TurboSamVisionEncoder
    
    
    model.vision_encoder.selected_layers = selected_layers
    model.vision_encoder.window_size = (2,2)
    model.vision_encoder.threshold = 0.88
    model.vision_encoder._turbo_info = {
        "size": None,
        "source": None,
        "trace_source": trace_source,
        "prop_attn": prop_attn,
        "class_token": None,
        "distill_token": False,
        "rel_pos": None,
        "selected_layers":model.vision_encoder.selected_layers,
        "window_size":model.vision_encoder.window_size,
        "threshold":model.vision_encoder.threshold,
     
    }

    if hasattr(model, "dist_token") and model.dist_token is not None:
        model.vision_encoder._turbo_info["distill_token"] = True


    len_att = 1
    len_block = 1
    for module in model.vision_encoder.modules():
        if isinstance(module, SamVisionLayer):
            # if len_att in model.vision_encoder.selected_layers:
            module.__class__ = TurboSamVisionLayer
            module._turbo_info = model.vision_encoder._turbo_info
            # len_att +=1 
        elif isinstance(module, SamVisionAttention):
            # if len_block in model.vision_encoder.selected_layers: 
            module.__class__ = TurboSamVisionAttention
            module._turbo_info = model.vision_encoder._turbo_info
            # len_block +=1 
