import torch.nn as nn
import torch


class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.init_h = nn.Linear(embed_size, hidden_size)
        self.init_c = nn.Linear(embed_size, hidden_size)

    def forward(self, features, captions, lengths):
        embeddings = self.embed(captions)  # (B, T, embed_size)

        # Use image features to initialize LSTM states
        h0 = self.init_h(features).unsqueeze(0)  # (1, B, hidden_size)
        c0 = self.init_c(features).unsqueeze(0)  # (1, B, hidden_size)

        # Pack sequence
        packed = nn.utils.rnn.pack_padded_sequence(embeddings, lengths, batch_first=True, enforce_sorted=False)
        hiddens, _ = self.lstm(packed, (h0, c0))
        outputs = self.linear(hiddens.data)

        return outputs

    
    def sample(self, features, start_token_idx, max_length = 30):
        output_ids = []
            # Initialize hidden and cell state using image features
        h0 = self.init_h(features).unsqueeze(0)  # (1, B, hidden_size)
        c0 = self.init_c(features).unsqueeze(0)  # (1, B, hidden_size)
        states = (h0, c0)

        # Start with <start> token
        start_tokens = torch.tensor([start_token_idx] * features.size(0), device=features.device)
        inputs = self.embed(start_tokens).unsqueeze(1)
# (B, 1, embed_size)

        for _ in range(max_length):
            hidden, states = self.lstm(inputs, states)
            outputs = self.linear(hidden.squeeze(1)) #(B,1,D) -> (B,vocab_size)
            predicted = outputs.argmax(1) #(B,)
            output_ids.append(predicted)

            inputs = self.embed(predicted) #(B,embed_size)
            inputs = inputs.unsqueeze(1) #(B,1,embed_size)

        output_ids = torch.stack(output_ids, 1)
        return output_ids

