# This file defines the models that generate prefixes (activations)
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.utils.checkpoint import checkpoint
from transformers.activations import ACT2FN

import warnings
import copy

from transformers import (
    PreTrainedModel,
    GPT2PreTrainedModel, 
    GPT2LMHeadModel, 
    T5ForConditionalGeneration,
    AutoConfig,
)


from transformers.configuration_utils import PretrainedConfig
from transformers.models.gpt2.configuration_gpt2 import GPT2Config
from transformers.models.t5.configuration_t5 import T5Config
from torch import nn
from util import get_class_object
from modeling_prefix_t5 import PrefixT5ForConditionalGeneration
from typing import Optional, Callable, Iterable, List, Tuple, Dict, Any

from modeling_prefixes import (
    MLPPrefixModel, 
    GPT2PrefixGenerateMixin, 
    T5PrefixGenerateMixin
)

import modeling_prefixes


class PrefixLMModelConfig(PretrainedConfig):
    """
    Configuration class for prefix-LM model
    """
    def __init__(
        self,
        lm_model_name=None,
        lm_config=None,
        prefix_model_name=None,
        prefix_len=10,
        hidden_dims=None,
        activation_fn_name='Tanh',
        is_encoder_decoder=False,
        prefix_dropout=0.0,
        use_cache=True,
        eos_token_id=None, # very important! Otherwise the `generate` API cannot stop early
        bos_token_id=None,
        pad_token_id=None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.lm_model_name = lm_model_name
        self.lm_config = lm_config or AutoConfig.from_pretrained(lm_model_name)
        self.prefix_model_name = prefix_model_name
        self.prefix_len = prefix_len
        self.hidden_dims = hidden_dims
        self.activation_fn_name = activation_fn_name
        self.is_encoder_decoder = is_encoder_decoder
        self.prefix_dropout = prefix_dropout
        self.use_cache = use_cache 
        self.eval_mode = False
        self.eos_token_id = eos_token_id or self.lm_config.eos_token_id
        self.bos_token_id = bos_token_id or self.lm_config.bos_token_id
        self.pad_token_id = pad_token_id or self.lm_config.pad_token_id 

class PT_GPT2Model(PreTrainedModel, GPT2PrefixGenerateMixin):
    """
    PrefixTuning-GPT2 model with MLP parameterization
    """

    config_class = PrefixLMModelConfig

    def __init__(
        self, 
        prefix_config: PrefixLMModelConfig = None,
    ):
        super().__init__(prefix_config)
        # self.gpt2_model = GPT2LMHeadModel(config.gpt2_config)
        self.prefix_config = prefix_config
        self.gpt2_model = GPT2LMHeadModel.from_pretrained(prefix_config.lm_model_name)
        self.gpt2_config = self.gpt2_model.config

        PrefixModel = get_class_object(modeling_prefixes, self.prefix_config.prefix_model_name)
        self.prefix_model = PrefixModel(
            n_layer=self.gpt2_config.n_layer,
            n_head=self.gpt2_config.n_head,
            n_model_dim=self.gpt2_config.n_embd,
            prefix_len=self.prefix_config.prefix_len,
            hidden_dims=self.prefix_config.hidden_dims,
            activation_fn_name=self.prefix_config.activation_fn_name,
            is_encoder_decoder=self.prefix_config.is_encoder_decoder,
            n_enc_layer=None,
            n_dec_layer=None,
            dropout=self.prefix_config.prefix_dropout,
        ).to(self.gpt2_model.device)

        # Freeze LM model
        for param in self.gpt2_model.base_model.parameters():
            param.requires_grad = False

        # Parameter Statistics
        total_params = 0
        trainable_params = 0
        for name, param in self.named_parameters():
            print(name, param.shape, f"Trainable = {param.requires_grad}")
            total_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        print(f"Total parameters = {total_params/1000000.0:.1f}M, Trainable parameters = {trainable_params/1000000.0:.1f}M")
    
    @staticmethod # copied from LM model
    def _reorder_cache(past: Tuple[Tuple[torch.Tensor]], beam_idx: torch.Tensor) -> Tuple[Tuple[torch.Tensor]]:
        """
        This function is used to re-order the :obj:`past_key_values` cache if
        :meth:`~transformers.PreTrainedModel.beam_search` or :meth:`~transformers.PreTrainedModel.beam_sample` is
        called. This is required to match :obj:`past_key_values` with the correct beam_idx at every generation step.
        """
        return tuple(
            tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past)
            for layer_past in past
        )
    
    def get_prefix(self, x):
        past_key_values = self.prefix_model(x)
        return past_key_values

    def forward(
        self,
        input_ids=None, 
        past_key_values=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        labels=None, 
        use_cache=None, 
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None
    ):
        use_cache = use_cache or self.config.use_cache
        past_key_values_prefix = self.get_prefix(input_ids)
        past_key_values = past_key_values or past_key_values_prefix
        # Must not exceed maximum context length for GPT2 after prepending prefix (1024)
        input_len = input_ids.shape[1]
        max_context_len = self.gpt2_model.transformer.wpe.num_embeddings
        if input_len > max_context_len - self.prefix_config.prefix_len:
            shift_len = input_len + self.prefix_config.prefix_len - max_context_len
            input_ids = input_ids[:,:-shift_len]
            if labels is not None: # shift labels as well
                labels = labels[:,:-shift_len]
        output = self.gpt2_model(
            input_ids=input_ids,
            past_key_values=past_key_values,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            labels=labels
        )
        return output

class PT_T5Model(PreTrainedModel):
    """
    PrefixTuning-T5 Model with MLP parameterization
    """

    config_class = PrefixLMModelConfig

    def __init__(
        self, 
        prefix_config: PrefixLMModelConfig = None,
    ):
        super().__init__(prefix_config)
        # self.gpt2_model = GPT2LMHeadModel(config.gpt2_config)
        self.prefix_config = prefix_config
        self.t5_model = PrefixT5ForConditionalGeneration.from_pretrained(self.prefix_config.lm_model_name)
        self.t5_config = self.t5_model.config
        self.config = self.prefix_config

        PrefixModel = get_class_object(modeling_prefixes, self.prefix_config.prefix_model_name)
        self.prefix_model = PrefixModel(
            n_layer=self.t5_config.num_layers,
            n_head=self.t5_config.num_heads,
            n_model_dim=self.t5_config.d_model,
            prefix_len=self.prefix_config.prefix_len,
            hidden_dims=self.prefix_config.hidden_dims,
            activation_fn_name=self.prefix_config.activation_fn_name,
            is_encoder_decoder=self.prefix_config.is_encoder_decoder,
            n_enc_layer=self.t5_config.num_layers,
            n_dec_layer=self.t5_config.num_decoder_layers,
            dropout=self.prefix_config.prefix_dropout,
        ).to(self.t5_model.device)

        # Freeze LM model
        for param in self.t5_model.base_model.parameters():
            param.requires_grad = False

        # Parameter Statistics
        total_params = 0
        trainable_params = 0
        for name, param in self.named_parameters():
            print(name, param.shape, f"Trainable = {param.requires_grad}")
            total_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        print(f"Total parameters = {total_params/1000000.0:.1f}M, Trainable parameters = {trainable_params/1000000.0:.1f}M")

    def get_input_embeddings(self):
        return self.t5_model.shared
    
    def set_input_embeddings(self, new_embeddings):
        self.t5_model.shared = new_embeddings
        self.t5_model.encoder.set_input_embeddings(new_embeddings)
        self.t5_model.decoder.set_input_embeddings(new_embeddings)
    
    def set_output_embeddings(self, new_embeddings):
        self.t5_model.lm_head = new_embeddings

    def get_output_embeddings(self):
        return self.t5_model.lm_head

    def get_encoder(self):
        return self.t5_model.encoder

    def get_decoder(self):
        return self.t5_model.decoder
    
    def prepare_inputs_for_generation(
        self,
        input_ids,
        past=None,
        attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs
    ):

        # cut decoder_input_ids if past is used
        if past is not None:
            input_ids = input_ids[:, -1:]
        
        encoder_key_value_prefixes, decoder_key_value_prefixes = self.get_prefix(input_ids)

        return {
            "decoder_input_ids": input_ids,
            "past_key_values": past,
            "encoder_outputs": encoder_outputs,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,
            "encoder_key_value_prefixes": encoder_key_value_prefixes, 
            "decoder_key_value_prefixes": decoder_key_value_prefixes, 
        }
    
    def _shift_right(self, input_ids):
        decoder_start_token_id = self.config.decoder_start_token_id
        pad_token_id = self.config.pad_token_id

        assert (
            decoder_start_token_id is not None
        ), "self.model.config.decoder_start_token_id has to be defined. In T5 it is usually set to the pad_token_id. See T5 docs for more information"

        # shift inputs to the right
        if is_torch_fx_proxy(input_ids):
            # Item assignment is not supported natively for proxies.
            shifted_input_ids = torch.full(input_ids.shape[:-1] + (1,), decoder_start_token_id)
            shifted_input_ids = torch.cat([shifted_input_ids, input_ids[..., :-1]], dim=-1)
        else:
            shifted_input_ids = input_ids.new_zeros(input_ids.shape)
            shifted_input_ids[..., 1:] = input_ids[..., :-1].clone()
            shifted_input_ids[..., 0] = decoder_start_token_id

        assert pad_token_id is not None, "self.model.config.pad_token_id has to be defined."
        # replace possible -100 values in labels by `pad_token_id`
        shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

        assert torch.all(shifted_input_ids >= 0).item(), "Verify that `shifted_input_ids` has only positive values"

        return shifted_input_ids

    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        return self._shift_right(labels)
    
    def _reorder_cache(self, past, beam_idx):
        # if decoder past is not included in output
        # speedy decoding is disabled and no need to reorder
        if past is None:
            logger.warning("You might want to consider setting `use_cache=True` to speed up decoding")
            return past

        reordered_decoder_past = ()
        for layer_past_states in past:
            # get the correct batch idx from layer past batch dim
            # batch dim of `past` is at 2nd position
            reordered_layer_past_states = ()
            for layer_past_state in layer_past_states:
                # need to set correct `past` for each of the four key / value states
                reordered_layer_past_states = reordered_layer_past_states + (
                    layer_past_state.index_select(0, beam_idx.to(layer_past_state.device)),
                )

            assert reordered_layer_past_states[0].shape == layer_past_states[0].shape
            assert len(reordered_layer_past_states) == len(layer_past_states)

            reordered_decoder_past = reordered_decoder_past + (reordered_layer_past_states,)
        return reordered_decoder_past
    
    def _prepare_encoder_decoder_kwargs_for_generation(
        self, input_ids: torch.LongTensor, model_kwargs
    ) -> Dict[str, Any]:
        """
        Overwrite to provide key_value_prefixes for encoder (T5Stack)
        """
        if "encoder_outputs" not in model_kwargs:
            # retrieve encoder hidden states
            encoder = self.get_encoder()
            encoder_kwargs = {
                argument: value
                for argument, value in model_kwargs.items()
                if not (argument.startswith("decoder_") or argument.startswith("cross_attn"))
            }
            encoder_key_value_prefixes, _ = self.get_prefix(input_ids)
            model_kwargs["encoder_outputs"]: ModelOutput = encoder(input_ids, return_dict=True, key_value_prefixes=encoder_key_value_prefixes, **encoder_kwargs)
        return model_kwargs
    
    def get_prefix(self, x):
        encoder_key_value_prefixes, decoder_key_value_prefixes = self.prefix_model(x)
        return encoder_key_value_prefixes, decoder_key_value_prefixes

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        encoder_key_value_prefixes=None,
        decoder_key_value_prefixes=None,
    ):
        use_cache = use_cache or self.config.use_cache
        if encoder_key_value_prefixes is None and decoder_key_value_prefixes is None:
            encoder_key_value_prefixes, decoder_key_value_prefixes = self.get_prefix(input_ids)
        
        output = self.t5_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            encoder_outputs=encoder_outputs,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            encoder_key_value_prefixes=encoder_key_value_prefixes,
            decoder_key_value_prefixes=decoder_key_value_prefixes,
        )
        return output

    


# if __name__ == "__main__":
#     config = PrefixConfig()
#     prefix_gpt2 = GPT2_MLPTaskPrefixModel(config)
#     output = prefix_gpt2(input_ids=torch.tensor([[12, 34, 1]]))
#     prefix_gpt2.save_pretrained('exp/prefix_gpt2-test')
#     loaded = GPT2_MLPTaskPrefixModel.from_pretrained('exp/prefix_gpt2-test')
#     print("Done!")
    # config = T5PrefixConfig()
    # prefix_t5 = T5_MLPTaskPrefixModel(config)
    # output = prefix_t5(input_ids=torch.tensor([[12,34,1]]), labels=torch.tensor([[1, 1]]))
    # # prefix_t5.save_pretrained('exp/prefix_t5-test')
    # loaded = T5_MLPTaskPrefixModel.from_pretrained('exp/prefix_t5-test')
    # prefix_model_config = PrefixModelConfig(
    #     lm_model_name='gpt2',
    #     lm_config=GPT2Config(),
    #     prefix_model_name='MLPPrefixModel',
    #     prefix_len=10,
    #     hidden_dims=[768, 512],
    #     activation_fn_name='Tanh',
    #     is_encoder_decoder=False,
    #     prefix_dropout=0.0
    # )
    # pt_gpt2 = PT_GPT2Model(prefix_model_config)
    # output = pt_gpt2(input_ids=torch.tensor([[12,34,1]]))

    # prefix_model_config = PrefixLMModelConfig(
    #     lm_model_name='t5-small',
    #     lm_config=T5Config(),
    #     prefix_model_name='MLPPrefixModel',
    #     prefix_len=10,
    #     hidden_dims=[512, 384],
    #     activation_fn_name='Tanh',
    #     is_encoder_decoder=True,
    #     prefix_dropout=0.0
    # )
    # pt_t5 = PT_T5Model(prefix_model_config)
    # output = pt_t5(input_ids=torch.tensor([[12,34,1]]), labels=torch.tensor([[1, 1]]))


    # print("Done!")
    
    

