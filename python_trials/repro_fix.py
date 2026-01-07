
import torch
import torch.nn as nn
import torch.optim as optim

class SkipGramNegativeSampling(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(SkipGramNegativeSampling, self).__init__()
        self.in_embed = nn.Embedding(vocab_size, embedding_dim)
        self.out_embed = nn.Embedding(vocab_size, embedding_dim)
        self.in_embed.weight.data.uniform_(-0.5/embedding_dim, 0.5/embedding_dim)
        self.out_embed.weight.data.uniform_(-0.5/embedding_dim, 0.5/embedding_dim)

    def forward(self, center_words, target_words, negative_words):
        center_vectors = self.in_embed(center_words)
        target_vectors = self.out_embed(target_words)
        negative_vectors = self.out_embed(negative_words)

        # Proposed fix: use squeeze(2) instead of squeeze()
        # Original failure point was squeeze() on (1, 5, 1) -> (5) for neg_score
        
        pos_score = torch.bmm(target_vectors.unsqueeze(1), center_vectors.unsqueeze(2)).squeeze(2)
        pos_score = torch.sigmoid(pos_score)

        neg_score = torch.bmm(negative_vectors, center_vectors.unsqueeze(2)).squeeze(2)
        neg_score = torch.sigmoid(neg_score)
        
        print(f"Pos score shape: {pos_score.shape}")
        print(f"Neg score shape: {neg_score.shape}")

        loss = -torch.log(pos_score + 1e-5) - torch.sum(torch.log(neg_score + 1e-5), dim=1)
        return torch.mean(loss)

VOCAB_SIZE = 100
EMBED_DIM = 10
model = SkipGramNegativeSampling(VOCAB_SIZE, EMBED_DIM)

center_id = torch.tensor([0], dtype=torch.long)
target_id = torch.tensor([1], dtype=torch.long)
negative_ids = torch.tensor([[50, 23, 99, 4, 12]], dtype=torch.long)

print("Running forward pass...")
loss = model(center_id, target_id, negative_ids)
print(f"Loss: {loss.item()}")
