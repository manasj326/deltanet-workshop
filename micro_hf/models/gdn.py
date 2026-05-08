import torch
from fla.layers import GatedDeltaNet
from typing import Any, Optional, Union
from fla.models.utils import Cache

class GatedLinearLayer(torch.nn.Module):
    def __init__(self, config, idx):
        super().__init__()

        # self.cache = Cache()
        self.layer_idx = idx
        self.layer = GatedDeltaNet(
            hidden_size=config.hidden_size,
            head_dim=config.hidden_size,
            num_heads=config.num_attention_heads,
            mode='chunk',           # 'chunk' or 'fused_recurrent'
            expand_k=1.0,
            expand_v=1.0,
            use_short_conv=False,    
            # use_short_conv=True,    # recommended, like Mamba's conv
            fuse_norm=True,         # fused output gate for memory efficiency
            layer_idx=idx,
        # ).to('cuda', torch.bfloat16)
        ).to('cuda', torch.float32)

    # @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        input_embeds: Optional[torch.LongTensor] = None,
        cache_params: Optional[Any] = None,
        use_cache: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_gdn_states: Optional[bool] = False,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
    ) -> Union[tuple, Any]:
        if output_gdn_states: 
            self.cache = Cache()
            outputs, states = [], []
            for t in range(input_embeds.shape[1]):
                output, _, self.cache = self.layer(input_embeds[:,t:t+1,:],
                                                        past_key_values=self.cache,
                                                        use_cache=True
                                                    )
                outputs.append(output)
                states.append(self.cache[self.layer_idx]["recurrent_state"].clone().detach())
            outputs = torch.cat(outputs, dim=1)
            states = torch.stack(states, dim=2)

            a = self.layer.a_proj(input_embeds).float()
            g = -self.layer.A_log.float().exp() * torch.nn.functional.softplus(a + self.layer.dt_bias)
            g = g.exp().detach().squeeze()

            return (outputs, {"state": states, "g": g})

        else:
            self.cache = Cache()
            output, _, _ = self.layer(input_embeds,
                                            past_key_values=self.cache,
                                            use_cache=True
                                        )
            return (output, None)
    


