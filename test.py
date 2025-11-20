import torch
import torch.nn.functional as F
from model.Transformer import TfVar
import numpy as np

def test_semantic_guidance():

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
  
    
 
    config = {
        'win_size': 100,
        'input_c': 10,
        'output_c': 10,
        'd_model': 32,
        'encoder_layers': 1,
        'branch1_networks': ['st_linear', 'inner_tf', 'multiscale_att'],
        'branch2_networks': ['multiatt_conv', 'in_tf'],
        'branch1_match_dimension': 'first',
        'branch2_match_dimension': 'first',
        'decoder_networks': ['linear'],
        'decoder_layers': 1,
        'decoder_group_embedding': 'False',
        'branches_group_embedding': 'False_False',
        'multiscale_kernel_size': [5],
        'multiscale_patch_size': [10, 20],
        'embedding_init': 'normal',
        'memory_guided': 'sinusoid',
        'temperature': 0.1,
        'device': device
    }
 
    model = TfVar(config).to(device)
    

    x = torch.randn(2, 100, 10, device=device)
    

    out = model(x)
    attention_maps = out['attention_maps']  
    

    

    np.save('attn_patch10.npy', attention_maps['patch_10'][0])  # [L, L]
    np.save('attn_patch20.npy', attention_maps['patch_20'][0])
    np.save('input.npy', x[0].cpu().numpy())
    np.save('label.npy', label[0].cpu().numpy())
    
    return "test complete"

if __name__ == "__main__":
    print(test_semantic_guidance()) 
