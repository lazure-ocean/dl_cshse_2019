from base import Model, Encoder, Decoder

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
