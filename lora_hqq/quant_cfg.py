from hqq.core.quantize import BaseQuantizeConfig
    
def get_quant_config_slm(model):
    quant_config = {}
    
    n_layers = model.config.num_hidden_layers
    q4_config = BaseQuantizeConfig(nbits=4, group_size=64)
    # q4_config = BaseQuantizeConfig(nbits=4, group_size=128)
    q2_config = BaseQuantizeConfig(nbits=2, group_size=64)
    print("n_layers", n_layers)
    for i in range(n_layers):
        quant_config[f'model.layers.{i}.self_attn.q_proj'] = q2_config
        quant_config[f'model.layers.{i}.self_attn.k_proj'] = q2_config
        quant_config[f'model.layers.{i}.self_attn.v_proj'] = q4_config
        quant_config[f'model.layers.{i}.self_attn.o_proj'] = q4_config
            
        quant_config[f'model.layers.{i}.mlp.gate_proj'] = q4_config
        quant_config[f'model.layers.{i}.mlp.up_proj'] = q4_config
        quant_config[f'model.layers.{i}.mlp.down_proj'] = q4_config
        
    return quant_config
