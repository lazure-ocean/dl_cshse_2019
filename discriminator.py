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