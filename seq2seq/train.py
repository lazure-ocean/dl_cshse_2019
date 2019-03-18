import torch
from torch import nn
from torch import functional as F

def greedy_sample(logits):
    batch_size, seq_len, _ = logits.size()
    max_values, max_indices = logits.max(dim=2)
    return max_indices

def critic(discriminator, generator_output, gamma, baseline=None):
    encoder, decoder, linear = discriminator
    hidden_size = encoder.hidden_size
    length = generator_output.size()[0]
    
    encoder_outputs = torch.zeros(max_length, hidden_size, device=device)
    for ind, element in enumerate(generator_output):
        encoder_output, encoder_hidden = encoder(element, encoder_hidden)
        encoder_outputs[ind] = encoder_output.view(hidden_size)
    
    decoder_outputs = torch.zeros(max_length, hidden_size, device=device)
    atth_weights = torch.zeros(max_length, hidden_size, device=device)
    for ind, element in enumerate(encoder_outputs):
        decoder_output, decoder_hidden, decoder_attention  = decoder(
            element, decoder_hidden, encoder_outputs)
        decoder_outputs[ind] = decoder_output.view(hidden_size)
        atth_weights[ind] = decoder_attention.view(hidden_size)
        
    outputs = linear(decoder_outputs)
    
    rewards = F.log_sigmoid(outputs)
    weight = atth_weights
    cumulative_rewards = []
    for t in range(length):
        cum_value = rewards.new_zeros(batch_size)
        for s in range(t, length):
            exp = float(s-t)
            k = (gamma ** exp)
            cum_value +=  k * weight[:, s]  * rewards[:, s]
        cumulative_rewards.append(cum_value)

    cumulative_rewards = torch.stack(cumulative_rewards, dim=1)

    if baseline is not None:
        baseline = baseline
        advantages = cumulative_rewards - baseline
    else:
        advantages = cumulative_rewards

    advantages = advantages - advantages.mean(dim=0)

    generator_objective = (advantages * log_probs).sum(dim=0)
    return (generator_objective, cumulative_rewards.clone())