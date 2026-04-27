from torch.nn import Module, Linear, ReLU
import copy
import torch

class ResurrectionNetwork(Module):
    def __init__(self, input_size, output_size):
        super(ResurrectionNetwork, self).__init__()
        self.fc1 = Linear(input_size, 256)
        #probe 1
        self.fc2 = Linear(256, 256)
        # probe 2
        self.fc3 = Linear(256, 256)
        # probe 3
        self.fc4 = Linear(256, output_size)

        self.relu = ReLU()
        
        #Network Probes
        self.probe1 = Linear(256, output_size)
        self.probe2 = Linear(256, output_size)
        self.probe3 = Linear(256, output_size)
        #self.probe4 = Linear(output_size, output_size)

    def forward(self, x, probe=False):
        x = x.view(x.size(0), -1)
        l1_out = self.fc1(x)
        l1_out = self.relu(l1_out)
        l2_out = self.fc2(l1_out)
        l2_out = self.relu(l2_out)
        l3_out = self.fc3(l2_out)
        l3_out = self.relu(l3_out)
        #l4_out = self.fc4(l3_out)
        output = self.fc4(l3_out)

        if probe:
            probe1_output = self.probe1(l1_out.detach())
            probe2_output = self.probe2(l2_out.detach())
            probe3_output = self.probe3(l3_out.detach())
            #probe4_output = self.probe4(l4_out.detach())
            return output, probe1_output, probe2_output, probe3_output #, probe4_output
        else:
            return output

def get_neuron_importance(layer):
    """
    Calculates importance based on the L1 norm of incoming weights.
    Returns a 1D tensor where each element is the score of a neuron.
    """
    # layer.weight shape is [out_features, in_features]
    # Summing across dim=1 gives the total absolute weight entering each neuron
    return torch.abs(layer.weight).sum(dim=1)

def resurrect_layer(model, prev_model, importance, layer_name, top_percent=0.9, zero_grad=True):
    """
    Overwrites the current weights of the most important neurons 
    with their saved weights and freezes them using gradient hooks.
    """
    num_neurons = importance.size(0)
    num_top = int(num_neurons * top_percent)
    
    # Get the indices of the most important neurons
    _, top_indices = torch.topk(importance, num_top)
    
    # --- 1. Restore the weights ---
    current_state = model.state_dict()
    saved_state = prev_model.state_dict()
    weight_key = f'{layer_name}.weight'
    bias_key = f'{layer_name}.bias'
    
    current_state[weight_key][top_indices] = saved_state[weight_key][top_indices]
    current_state[bias_key][top_indices] = saved_state[bias_key][top_indices]
    
    # Load the patched state back into the model
    model.load_state_dict(current_state)
    
    # --- 2. Freeze the resurrected neurons ---
    # Retrieve the actual layer object from the model
    layer = getattr(model, layer_name)
    
    # Create masks of 1s (meaning "keep the gradient")
    weight_mask = torch.ones_like(layer.weight)
    bias_mask = torch.ones_like(layer.bias)
    
    # Set the rows corresponding to the resurrected neurons to 0 ("kill the gradient")
    weight_mask[top_indices] = 0.0
    bias_mask[top_indices] = 0.0
    
    # Register the hooks. During loss.backward(), these lambda functions 
    # will multiply the incoming gradients by our masks.
    layer.weight.register_hook(lambda grad: grad * weight_mask)
    layer.bias.register_hook(lambda grad: grad * bias_mask)
    
    print(f"Resurrected and froze top {num_top} neurons in {layer_name}")