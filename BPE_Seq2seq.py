import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
import tqdm
import evaluate
import os

seed = 1234
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True

dataset = load_dataset("thainq107/iwslt2015-en-vi")
train_data, valid_data, test_data = dataset["train"], dataset["validation"], dataset["test"]

# English tokenizer
tokenizer_en = Tokenizer(BPE())
tokenizer_en.pre_tokenizer = Whitespace()
trainer_en = BpeTrainer(
    vocab_size=30000,
    special_tokens=["<unk>", "<p ad>", "<sos>", "<eos>"]
)
tokenizer_en.train_from_iterator(train_data["en"], trainer=trainer_en)

# Vietnamese tokenizer
tokenizer_vi = Tokenizer(BPE())
tokenizer_vi.pre_tokenizer = Whitespace()
trainer_vi = BpeTrainer(
    vocab_size=30000,
    special_tokens=["<unk>", "<pad>", "<sos>", "<eos>"]
)
tokenizer_vi.train_from_iterator(train_data["vi"], trainer=trainer_vi)

# Tokenizer and Normalize
def tokenize_example(example, tokenizer_src, tokenizer_trg, sos_token, eos_token, max_length=1000):
    # Source = English, Target = Vietnamese
    src_ids = tokenizer_src.encode(example["en"]).ids[:max_length]
    trg_ids = tokenizer_trg.encode(example["vi"]).ids[:max_length]
    # Add <sos> and <eos>
    src = [tokenizer_src.token_to_id(sos_token)] + src_ids + [tokenizer_src.token_to_id(eos_token)]
    trg = [tokenizer_trg.token_to_id(sos_token)] + trg_ids + [tokenizer_trg.token_to_id(eos_token)]
    return {"en_ids": src, "vi_ids": trg}

fn_kwargs = {
    "tokenizer_src": tokenizer_en,
    "tokenizer_trg": tokenizer_vi,
    "sos_token": "<sos>",
    "eos_token": "<eos>",
    "max_length": 30
}

# Note: we no longer remove_columns=["en","vi"]
train_data = train_data.map(tokenize_example, fn_kwargs=fn_kwargs)
valid_data = valid_data.map(tokenize_example, fn_kwargs=fn_kwargs)
test_data  = test_data.map(tokenize_example, fn_kwargs=fn_kwargs)

# Convert lists to torch.Tensor()
def to_tensor(example):
    return{
        "en_ids": torch.tensor(example["en_ids"], dtype=torch.long),
        "vi_ids": torch.tensor(example["vi_ids"], dtype=torch.long)
    }

train_data = train_data.with_format(
    type="torch",
    columns=["en_ids", "vi_ids"],
    output_all_columns=True
)

valid_data = valid_data.with_format(
    typr="torch",
    columns=["en_ids", "vi_ids"],
    output_all_columns=True
)

test_data = test_data.with_format(
    type="torch",
    columns=["en_ids", "vi_ids"],
    output_all_columns=True
)

#DataLoader
def get_collate_fn(pad_id):
    def collate_fn(batch):
        src = [ex["en_ids"] for ex in batch]
        trg = [ex["vi_ids"] for ex in batch]
        src = nn.utils.rnn.pad_sequence(src, padding_value=pad_id)
        trg = nn.utils.rnn.pad_sequence(trg, padding_value=pad_id)
        return {"en_ids": src, "vi_ids": trg}
    return collate_fn

pad_id = tokenizer_en.token_to_id("<pad>")
batch_size = 64

train_loader = torch.utils.data.DataLoader(
    train_data, batch_size=batch_size,
    collate_fn=get_collate_fn(pad_id), shuffle=True
)

valid_loader = torch.utils.data.DataLoader(
    valid_data, batch_size = batch_size,
    collate_fn=get_collate_fn(pad_id)
)

test_loader = torch.utils.data.DataLoader(
    test_data, batch_size = batch_size,
    collate_fn=get_collate_fn(pad_id)
)

# Model Seq2Seq
class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        # src = [src_len, batch_size]
        embedded = self.dropout(self.embedding(src))
        outputs, (hidden, cell) = self.rnn(embedded)
        return hidden, cell

class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout)
        self.fc_out = nn.Linear(hid_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, cell):
        # input = [batch_size]
        input = input.unsqueeze(0)
        # input = [1, batch_size]
        embedded = self.dropout(self.embedding(input))
        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        # output = [1, batch_size, hid_dim]
        prediction = self.fc_out(output.squeeze(0))
        # prediction = [batch_size, output_dim]
        return prediction, hidden, cell

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        assert encoder.rnn.hidden_size == decoder.rnn.hidden_size
        assert encoder.rnn.num_layers == decoder.rnn.num_layers
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        # src = [src_len, batch_size], trg = [trg_len, batch_size]
        batch_size = trg.shape[1]
        trg_len = trg.shape[0]
        vocab_size = self.decoder.fc_out.out_features
        outputs = torch.zeros(trg_len, batch_size, vocab_size).to(self.device)

        hidden, cell = self.encoder(src)
        input = trg[0, :]  # <sos>

        for t in range(1, trg_len):
            output, hidden, cell = self.decoder(input, hidden, cell)
            outputs[t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(1)
            input = trg[t] if teacher_force else top1

        return outputs


#Initialize Model + Training Setup

import os
# right:
INPUT_DIM  = tokenizer_en.get_vocab_size()
OUTPUT_DIM = tokenizer_vi.get_vocab_size()
ENC_EMB_DIM = 256
DEC_EMB_DIM = 256
HID_DIM     = 512
N_LAYERS    = 2
ENC_DROPOUT = 0.5
DEC_DROPOUT = 0.5
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

enc = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT)
dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT)
model = Seq2Seq(enc, dec, DEVICE).to(DEVICE)

def init_weights(m):
    for p in m.parameters():
        nn.init.uniform_(p.data, -0.08, 0.08)
model.apply(init_weights)

optimizer = optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss(ignore_index=pad_id)

CHECKPOINT_PATH = '/kaggle/input/bpe_seq2seq/pytorch/default/1/best-model.pt'  

if os.path.isfile(CHECKPOINT_PATH):
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
    print(f"✔ Loaded checkpoint from {CHECKPOINT_PATH}, resuming training.")
else:
    # only initialize weights if there's no checkpoint
    def init_weights(m):
        for p in m.parameters():
            nn.init.uniform_(p.data, -0.08, 0.08)
    model.apply(init_weights)
    print("✗ No checkpoint found — training from scratch.")

print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

# Training and Evaluation Functions

def train_fn(model, loader, optimizer, criterion, clip, device):
    model.train()
    epoch_loss = 0
    for batch in loader:
        src = batch["en_ids"].to(device)
        trg = batch["vi_ids"].to(device)
        optimizer.zero_grad()
        output = model(src, trg, teacher_forcing_ratio=0.5)
        # output = [trg_len, batch_size, vocab_size]
        output_dim = output.shape[-1]
        out = output[1:].view(-1, output_dim)
        tgt = trg[1:].view(-1)
        loss = criterion(out, tgt)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(loader)

def eval_fn(model, loader, criterion, device):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for batch in loader:
            src = batch["en_ids"].to(device)
            trg = batch["vi_ids"].to(device)
            output = model(src, trg, teacher_forcing_ratio=0.5)  # no teacher forcing
            output_dim = output.shape[-1]
            out = output[1:].view(-1, output_dim)
            tgt = trg[1:].view(-1)
            loss = criterion(out, tgt)
            epoch_loss += loss.item()
    return epoch_loss / len(loader)

# Train Model
N_EPOCHS = 30
CLIP     = 1.0

best_valid = float('inf')
for epoch in range(N_EPOCHS):
    train_loss = train_fn(model, train_loader, optimizer, criterion, CLIP, DEVICE)
    valid_loss = eval_fn(model, valid_loader, criterion, DEVICE)
    if valid_loss < best_valid:
        best_valid = valid_loss
        torch.save(model.state_dict(), 'best-model.pt')
    print(f"Epoch {epoch+1} | Train Loss: {train_loss:.3f} | Train PPL: {np.exp(train_loss):.3f}")
    print(f"          | Val   Loss: {valid_loss:.3f} | Val   PPL: {np.exp(valid_loss):.3f}")

# Testing + BLEU Score

model.load_state_dict(torch.load('best-model.pt'))
test_loss = eval_fn(model, test_loader, criterion, DEVICE)
print(f"Test Loss: {test_loss:.3f} | Test PPL: {np.exp(test_loss):.3f}")

# Translation helper
def translate_sentence(
    sentence, model, tokenizer_src, tokenizer_trg,
    lower=True, sos_token="<sos>", eos_token="<eos>",
    device=DEVICE, max_len=30
):
    model.eval()
    tokens = sentence.split()  # already whitespace-tokenized
    tokens = [sos_token] + tokens + [eos_token]
    src_ids = tokenizer_src.encode(" ".join(tokens)).ids
    src_tensor = torch.LongTensor(src_ids).unsqueeze(1).to(device)
    hidden, cell = model.encoder(src_tensor)

    outputs = [tokenizer_trg.token_to_id(sos_token)]
    for _ in range(max_len):
        prev = torch.LongTensor([outputs[-1]]).to(device)
        pred, hidden, cell = model.decoder(prev, hidden, cell)
        top1 = pred.argmax(1).item()
        outputs.append(top1)
        if top1 == tokenizer_trg.token_to_id(eos_token):
            break

    return tokenizer_trg.decode(outputs)

# Compute BLEU on test set
bleu = evaluate.load("bleu")
predictions = []
references  = []

for ex in tqdm.tqdm(test_data):
    # ex["en"] & ex["vi"] are still present because we didn't remove them
    pred = translate_sentence(
        ex["en"], model,
        tokenizer_en, tokenizer_vi,
        device=DEVICE
    )
    predictions.append(pred)
    references.append([ex["vi"]])

results = bleu.compute(predictions=predictions, references=references)
print(f"BLEU score = {results['bleu']:.4f}")
## BLEU score = 0.0685

# Test Model
sentence = test_data[0]["en"]
expected_translation = test_data[0]["vi"]
print("Source (English):", sentence)
print("Expected Translation (Vietnamese):", expected_translation)
translation = translate_sentence(sentence, model, tokenizer_en, tokenizer_vi,device=DEVICE)
print("Model Translation:", translation)

# Source (English): When I was little , I thought my country was the best on the planet , and I grew up singing a song called &quot; Nothing To Envy . &quot;
# Expected Translation (Vietnamese): Khi tôi còn nhỏ , Tôi nghĩ rằng BắcTriều Tiên là đất nước tốt nhất trên thế giới và tôi thường hát bài &quot; Chúng ta chẳng có gì phải ghen tị . &quot;
# Model Translation: Khi tôi nhỏ , tôi nghĩ rằng tôi là đất nước là một thế giới tốt đẹp nhất và tôi đã hát rằng tôi gọi là & quot ; Ain