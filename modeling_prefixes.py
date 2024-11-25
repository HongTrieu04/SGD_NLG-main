import torch
from torch import nn
from util import get_class_object
from transformers.modeling_outputs import (
    ModelOutput,
    BaseModelOutput,
    BaseModelOutputWithCrossAttentions,
    BaseModelOutputWithPastAndCrossAttentions,
    Seq2SeqLMOutput,
    Seq2SeqModelOutput,
)
from typing import Optional, Callable, Iterable, List, Tuple, Dict, Any

class MLPPrefixModel(nn.Module):
    """
    MLP Parameterization of Prefixes
    For decoder model, all prefixes are parameterized by one MLP
    For encoder-decoder model, the prefixes of encoder self-attention, decoder self-attention and decoder cross-attention share differente parameterization respectively
    """
    def __init__(
        self,
        n_layer=0,
        n_head=0,
        n_model_dim=0,
        prefix_len=0,
        hidden_dims=None,
        activation_fn_name='Tanh',
        is_encoder_decoder=False, # if True, n_layer will be discarded and n_enc_layer, n_dec_layer will be used
        n_enc_layer=None,
        n_dec_layer=None,
        dropout=0.0, # not used yet
        **kwargs
    ):
        """
        @Parameters:
        - n_layer: number of layers of the decoder-based transformer, omitted when `is_encoder_decoder`=True 
        - n_head: number of heads of the transformer
        - prefix_len: length of prefixes in each attention mechanism
        - hidden_dims: hidden dimensions of the MLP
        - activation_fn_name: activation function of the MLP
        - is_encoder_decoder: whether the transformer is of encoder-decoder architecture
        - n_enc_layer: number of layers for encoder, used when `is_encoder_decoder`=True
        - n_dec_layer: number of layers for decoder, used when `is_encoder_decoder`=True
        - dropout: dropout ratio during training
        - kwargs: fed to __init__ of nn.Module
        """
        super().__init__(**kwargs)
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_model_dim = n_model_dim
        self.prefix_dim = self.n_model_dim // self.n_head
        self.prefix_len = prefix_len

        self.prefix_indices = torch.arange(self.prefix_len).long()

        self.is_encoder_decoder = is_encoder_decoder
        if is_encoder_decoder:
            self.n_enc_layer = n_enc_layer
            self.n_dec_layer = n_dec_layer
            self.enc_self_output_dim = 2*self.n_enc_layer*self.n_model_dim # key,value for encoder self-attn
            self.dec_self_output_dim = 2*self.n_dec_layer*self.n_model_dim # key,value for decoder self-attn
            self.dec_cross_output_dim = 2*self.n_dec_layer*self.n_model_dim # key,value for decoder cross-attn
        else:
            self.output_dim = 2*self.n_layer*self.n_model_dim # key,value for attn
        self.hidden_dims = hidden_dims
        self.activation_fn_name = activation_fn_name 
        self.mlp_activation = get_class_object(nn, self.activation_fn_name)

        # self.prefix_indices = torch.arange(self.prefix_len).long()
        self.prefix_embeds = None 
        
        def create_mlp(hidden_dims, output_dim):
            mlp_model = nn.ModuleList()
            mlp_model.append(nn.Embedding(self.prefix_len, self.hidden_dims[0]))
            for in_dim, out_dim in zip(hidden_dims[:-1], hidden_dims[1:]):
                mlp_model.append(nn.Linear(in_dim, out_dim))
                mlp_model.append(self.mlp_activation())
            mlp_model.append(nn.Linear(hidden_dims[-1], output_dim))
            return nn.Sequential(*mlp_model)
        # MLP model for generating prefixes
        if self.is_encoder_decoder:
            self.encoder_mlp = create_mlp(hidden_dims, self.enc_self_output_dim)
            self.decoder_self_mlp = create_mlp(hidden_dims, self.dec_self_output_dim)
            self.decoder_cross_mlp = create_mlp(hidden_dims, self.dec_cross_output_dim)
        else:
            self.mlp_model = create_mlp(hidden_dims, self.output_dim)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        batch_size = x.shape[0]
        prefix_tokens = self.prefix_indices.unsqueeze(0).expand(batch_size, -1)
        device = next(self.parameters()).device
        prefix_tokens = prefix_tokens.to(device)
        if self.is_encoder_decoder: # different parameterization
            enc_prefixes = self.encoder_mlp(prefix_tokens)
            dec_self_prefixes = self.decoder_self_mlp(prefix_tokens)
            dec_cross_prefixes = self.decoder_cross_mlp(prefix_tokens)

            enc_prefixes = enc_prefixes.view(batch_size, self.prefix_len,
                self.n_enc_layer*2, self.n_head, self.prefix_dim)
            dec_self_prefixes = dec_self_prefixes.view(batch_size, self.prefix_len,
                self.n_dec_layer*2, self.n_head, self.prefix_dim)
            dec_cross_prefixes = dec_cross_prefixes.view(batch_size, self.prefix_len,
                self.n_dec_layer*2, self.n_head, self.prefix_dim)

            # layers_num tuple shape=(k/v batch_size, heads, prefix_length, head_embed), this agrees with past_key_values argument for GPT2
            enc_prefixes = enc_prefixes.permute([2,0,3,1,4]).split(2)
            dec_self_prefixes = dec_self_prefixes.permute([2,0,3,1,4]).split(2)
            dec_cross_prefixes = dec_cross_prefixes.permute([2,0,3,1,4]).split(2)
            dec_prefixes = tuple([torch.cat((dec_self, dec_cross)) for dec_self, dec_cross in zip(dec_self_prefixes, dec_cross_prefixes)])
            return enc_prefixes, dec_prefixes
        else:
            prefixes = self.mlp_model(prefix_tokens)
            # batch_size, prefix_length, layer*2(k,v), heads, head_embed
            past_key_values = prefixes.view(batch_size, self.prefix_len, 
                self.n_layer*2, self.n_head, self.prefix_dim)
            past_key_values = self.dropout(past_key_values)
            # layers_num tuple shape=(k/v batch_size, heads, prefix_length, head_embed), this agrees with past_key_values argument for GPT2
            past_key_values = past_key_values.permute([2,0,3,1,4]).split(2) # split to # of layers
            return past_key_values

class GPT2PrefixGenerateMixin:
    """
    Mixin so that the derived class with GPT2 model can use the `generate` API of huggingface
    Requires self.gpt2_model
    """
    def get_input_embeddings(self):
        return self.gpt2_model.transformer.wte
    
    def set_input_embeddings(self, new_embeddings):
        self.gpt2_model.transformer.wte = new_embeddings
    
    def get_output_embeddings(self):
        return self.gpt2_model.lm_head
    
    def set_output_embeddings(self, new_embeddings):
        self.gpt2_model.lm_head = new_embeddings
    
    
    # copied from MLP LM
    def prepare_inputs_for_generation(self, input_ids, past=None, **kwargs):
        token_type_ids = kwargs.get("token_type_ids", None)
        # only last token for inputs_ids if past is defined in kwargs
        if past:
            input_ids = input_ids[:, -1].unsqueeze(-1)
            if token_type_ids is not None:
                token_type_ids = token_type_ids[:, -1].unsqueeze(-1)

        attention_mask = kwargs.get("attention_mask", None)
        position_ids = kwargs.get("position_ids", None)

        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past:
                position_ids = position_ids[:, -1].unsqueeze(-1)
        else:
            position_ids = None
        return {
            "input_ids": input_ids,
            "past_key_values": past,
            "use_cache": kwargs.get("use_cache"),
            "position_ids": position_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        }

class T5PrefixGenerateMixin:
    """
    Mixin so that the derived class with T5 model can use the `generate` API of huggingface
    requires self.t5_model
    """
    def __init__(self, t5_model):
        self.t5_model = t5_model
    
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
    
    def _prepare_inputs_for_generation(
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
        
        encoder_key_value_prefixes, decoder_key_value_prefixes = self.get_prefix(input_ids.shape[0])

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
            encoder_key_value_prefixes, _ = self.get_prefix(input_ids.shape[0])
            model_kwargs["encoder_outputs"]: ModelOutput = encoder(input_ids, return_dict=True, key_value_prefixes=encoder_key_value_prefixes, **encoder_kwargs)
        return model_kwargs


# if __name__ == '__main__':
#     # mlp_prefix_model = MLPPrefixModel(
#     #     n_layer=12,
#     #     n_head=8,
#     #     n_model_dim=768,
#     #     prefix_len=10,
#     #     hidden_dims=[768, 512],
#     #     activation_fn_name='Tanh',
#     #     is_encoder_decoder=False
#     # )
#     # output = mlp_prefix_model(torch.ones((20,1)))
#     # print(len(output))

#     enc_dec_prefix_model = MLPPrefixModel(
#         n_enc_layer=6,
#         n_dec_layer=6,
#         n_head=8,
#         n_model_dim=512,
#         prefix_len=10,
#         hidden_dims=[512, 256],
#         activation_fn_name='Tanh',
#         is_encoder_decoder=True
#     )
#     enc_output, dec_output = enc_dec_prefix_model(torch.ones(20, 1))
#     print(len(enc_output), len(dec_output))
    


        


