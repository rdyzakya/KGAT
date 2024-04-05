from transformers import MistralForCausalLM

# https://github.com/huggingface/transformers/blob/main/src/transformers/models/mistral/modeling_mistral.py

# class VTMistral(MistralForCausalLM):
#     def __init__(self, config):
#         super().__init__(config)
    
#     @add_start_docstrings_to_model_forward(MISTRAL_INPUTS_DOCSTRING)
#     @replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
#     def forward(
#         self,
#         input_ids: torch.LongTensor = None,
#         attention_mask: Optional[torch.Tensor] = None,
#         position_ids: Optional[torch.LongTensor] = None,
#         past_key_values: Optional[List[torch.FloatTensor]] = None,
#         inputs_embeds: Optional[torch.FloatTensor] = None,
#         labels: Optional[torch.LongTensor] = None,
#         use_cache: Optional[bool] = None,
#         output_attentions: Optional[bool] = None,
#         output_hidden_states: Optional[bool] = None,
#         return_dict: Optional[bool] = None,
#     ) -> Union[Tuple, CausalLMOutputWithPast]: