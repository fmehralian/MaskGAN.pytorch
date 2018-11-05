
from fairseq.models.lstm \
        import LSTMEncoder, \
               LSTMDecoder, \
               LSTMModel

from fairseq.models.fairseq_model \
        import FairseqModel

from torch.distributions.categorical import Categorical
from warnings import warn
from torch import nn
import torch

class MGANGEncoder(LSTMEncoder): pass
class MGANGDecoder(LSTMDecoder): pass
class MGANGenerator(LSTMModel):
    def forward(self, src_tokens, src_lengths, prev_output_tokens):
        logits, attns = super().forward(src_tokens, src_lengths, prev_output_tokens)
        bsz, seqlen, vocab_size = logits.size()
        # print("Logits size:", logits.size())

        # Sample from x converting it to probabilities
        samples = []
        log_probs = []
        for t in range(seqlen):
            # input is B x T x C post transposing
            logit = logits[:, t, :]
            # Good news, categorical works for a batch.
            # B x H dimension. Looks like logit's are already in that form.
            distribution = Categorical(logits=logit)
            # Output is H dimension?
            sampled = distribution.sample().unsqueeze(1)
            log_probs.append(distribution.log_prob(sampled))
            samples.append(sampled)
            

        # Once all are sampled, it's possible to find the rewards from the generator.
        samples = torch.cat(samples, dim=1)
        # I may need to strip off an extra token generated.
        samples = samples[:, 1:]
        return (samples, log_probs, attns)

class MLEGenerator(LSTMModel):
    pass
