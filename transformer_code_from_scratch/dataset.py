import torch
import torch.nn as nn
from torch.utils.data import Dataset

# main class for dataset
class BilingualDataset(Dataset):
            # This is how our data looks like: 
            # {
            # "en": "Come, Bessie, we will leave her: I wouldn't have her heart for anything.",
            # "it": "Venite, Bessie, lasciamola. Non vorrei davvero avere un cuore come il suo."
            # }
    def __init__(self, ds, tokenizer_src, tokenizer_tgt, src_lang, tgt_lang, seq_len):

        # we will call it from the main file by these arguments: 
        # train_ds = BilingualDataset(train_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])
        # val_ds = BilingualDataset(val_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])
        
        super().__init__() # invoking the dataset class constructor
        self.seq_len = seq_len # seq_len is just a fixed length of sequence given in cofig file in order to have the padding done firstly,
        # that I believe most of the data is less than that. This can also be done dynamically on batch level, but it's a bit more complex to implement.

        self.ds = ds # passing the dataset itself
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        # We have tokenizer_src and tokenizer_tgt as two different tokenizers. This is because we have two different languages.
        
        self.src_lang = src_lang # src_lang = 'en' # english
        self.tgt_lang = tgt_lang # tgt_lang = 'it' # italian

        self.sos_token = torch.tensor([tokenizer_tgt.token_to_id("[SOS]")], dtype=torch.int64) # start of sentence token
        self.eos_token = torch.tensor([tokenizer_tgt.token_to_id("[EOS]")], dtype=torch.int64) # end of sentence token
        self.pad_token = torch.tensor([tokenizer_tgt.token_to_id("[PAD]")], dtype=torch.int64) # padding token

    # __len__ method for the Dataset Class -> returns the length of the dataset :)
    def __len__(self):
        return len(self.ds)
    # __getitem__ func 
    def __getitem__(self, idx):

        src_target_pair = self.ds[idx]
        src_text = src_target_pair['translation'][self.src_lang] # get the raw sentence pair from source
        tgt_text = src_target_pair['translation'][self.tgt_lang] # get the raw sentecne pair from target

        # Transform the text into tokens
        enc_input_tokens = self.tokenizer_src.encode(src_text).ids # using source tokenizer
        dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids # using target tokenizer
        # the 'ids' is esentially simply pulling only the ids from that encoding object
        # "I am vraj" => encode()  =>  Encoding object  =>  .ids  =>  [42, 731, 156]

        # Add sos, eos and padding to each sentence
        enc_num_padding_tokens = self.seq_len - len(enc_input_tokens) - 2  # We will add <s> and </s>
        # We will only add <s>, and </s> only on the label
        dec_num_padding_tokens = self.seq_len - len(dec_input_tokens) - 1

        # Make sure the number of padding tokens is not negative. If it is, the sentence is too long
        if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0:
            raise ValueError("Sentence is too long")

        # Add <s> and </s> token
        # [SOS, token1, token2, token3, EOS, PAD, PAD, PAD]
        # encoder_input : Gets both [SOS] and [EOS] because the encoder reads the complete source sentence. PAD fills the rest to reach seq_len
        encoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(enc_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * enc_num_padding_tokens, dtype=torch.int64),
            ],
            dim=0,
        )

        # Add only <s> token
        # [SOS, token1, token2, token3, PAD, PAD, PAD] 
        # Gets only [SOS], no [EOS]. This is the "prompt" fed into the decoder — it starts with SOS and the model predicts what comes next at each position.
        decoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64),
            ],
            dim=0,
        )

        # Add only </s> token
        # [token1, token2, token3, EOS, PAD, PAD, PAD]
        # Gets only [EOS], no [SOS]. This is the expected answer the loss function compares against.

        label = torch.cat(
            [
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64),
            ],
            dim=0,
        )

        # Why decoder_input and label are offset by one
        # decoder_input:  [SOS,    Je,     t'aime,  PAD]
        # label:          [Je,     t'aime, EOS,     PAD]
        # At every position, decoder sees current token and must predict the next one. 
        # That's how the model learns — compare prediction at position N against label at position N.

        # Double check the size of the tensors to make sure they are all seq_len long
        assert encoder_input.size(0) == self.seq_len
        assert decoder_input.size(0) == self.seq_len
        assert label.size(0) == self.seq_len

        return {
            "encoder_input": encoder_input,  # (seq_len)
            "decoder_input": decoder_input,  # (seq_len)
            "encoder_mask": (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(), # (1, 1, seq_len)
            # Compare every position against PAD token ID → returns boolean tensor
            # encoder_input:  [SOS, 42, 731, 156, EOS, PAD, PAD]
            # mask:           [True, True, True, True, True, False, False]
            # .unsqueeze(0).unsqueeze(0) adds two extra dimensions:
            # (seq_len,) → (1, seq_len) → (1, 1, seq_len)
            # needed because attention scores are (batch, heads, seq_len, seq_len) → mask broadcasts across batch and heads
            # .int() converts booleans to 0s and 1s → True = 1 (keep), False = 0 (mask out)
            "decoder_mask": (decoder_input != self.pad_token).unsqueeze(0).int() & causal_mask(decoder_input.size(0)), # (1, seq_len) & (1, seq_len, seq_len),
            "label": label,  # (seq_len)
            "src_text": src_text,
            "tgt_text": tgt_text,
        }
    
def causal_mask(size):
    mask = torch.triu(torch.ones((1, size, size)), diagonal=1).type(torch.int) # torch.triu gives you the upper triangular matrix. With diagonal=1 it zeroes out the main diagonal too.
    return mask == 0
# encoder_mask:  (1, 1, seq_len)         ← broadcasts over batch and heads
# decoder_mask:  (1, seq_len, seq_len)   ← full attention matrix per sample