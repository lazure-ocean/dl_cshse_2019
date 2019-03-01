import torch
from torch.nn.utils.clip_grad import clip_grad_norm_
from base import Model, Encoder, Decoder

class DiscriminatorEncoder(Encoder):
    pass

class DiscriminatorDecoder(Decoder):
    def __init__(self):
        super().__init__()
        out_embed_dim =  self.hidden_size
        self.fc_out = nn.Linear(out_embed_dim, 1)

    def forward(self, prev_output_tokens, encoder_out_dict):
        x, attn_scores = super().forward(prev_output_tokens, encoder_out_dict)
        return x, attn_scores
    
class Discriminator(nn.Module):
    def __init__(self, encoder, decoder):
        super.__init__()
        self.encoder = encoder
        self.decoder = decoder
    
    def forward(self, src_tokens, src_lengths, prev_output_tokens):
        encoder_out = self.encoder(src_tokens, src_lengths)
        decoder_out = self.decoder(prev_output_tokens, encoder_out)
        return decoder_out
    
def discriminator_loss(predictions, labels, missing_tokens):
    loss = f.cross_entropy_loss(labels, predictions, weight=missing_tokens)
    return loss

def generator_GAN_loss(predictions):
    """Generator GAN loss based on Discriminator predictions."""
    return -torch.log(torch.mean(predictions))

def cross_entropy_loss_matrix(gen_labels, gen_logits):
    """Computes the cross entropy loss for G.
    Args:
    gen_labels:  Labels for the correct token.
    gen_logits: Generator logits.
    Returns:
    loss_matrix:  Loss matrix of shape [batch_size, sequence_length].
    """
    loss = torch.sum(-target * F.log_softmax(logits, -1), -1)
    mean_loss = loss.mean()
    return mean_loss


class ClippedAdam(torch.optim.Adam):
    def __init__(self, parameters, *args, **kwargs):
        super().__init__(parameters, *args, **kwargs)
        self.clip_value = None
        self._parameters = parameters

    def set_clip(self, clip_value):
        self.clip_value = clip_value

    def step(self, *args, **kwargs):
        assert (self.clip_value is not None)
        clip_grad_norm_(self._parameters, self.clip_value)
        super().step(*args, **kwargs)