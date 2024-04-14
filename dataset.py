import torch
import torch.nn as nn
from torch.utils.data import Dataset


class BilingualDataset(Dataset):
    def __init__(self, ds, tokenizer_src, tokenizer_tgt, src_lang, tgt_lang, seq_len):
        assert tokenizer_src is not None, "tokenizer_src must be initialized before being used"
        assert tokenizer_tgt is not None, "tokenizer_tgt must be initialized before being used"

        super().__init__()
        self.ds = ds
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.seq_len = seq_len

        # Add special tokens:
        # [SOS] = Start of sentence
        self.sos_token = tokenizer_tgt.token_to_id("[SOS]")
        # [EOS] = End of sentence
        self.eos_token = tokenizer_tgt.token_to_id("[EOS]")
        # [PAD] = Padding
        self.pad_token = tokenizer_tgt.token_to_id("[PAD]")
        
    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        # Get the item from the dataset
        item = self.ds[idx]

        # Get source and target texts
        src_text = item["translation"][self.src_lang]
        tgt_text = item["translation"][self.tgt_lang]

        # Tokenize
        enc_input_tokens = self.tokenizer_src.encode(src_text).ids
        dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids

        # Get the number of padding tokens minus the special tokens [SOS] and [EOS], so
        # we need to rest 2 from the length
        enc_num_padding_tokens = self.seq_len - len(enc_input_tokens) - 2
        # For the decoder we only need the [SOS] token, so we rest 1 from the length
        dec_num_padding_tokens = self.seq_len - len(dec_input_tokens) - 1

        # Sanity check
        if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0:
            raise ValueError("The sequence length is too small")

        # We need to return the tensors for the encoder, decoder and 
        # target (or label or ground truth).
        #
        # The encoder input is the source text with the [SOS] and [EOS] tokens
        # NOTE: .unsqueeze(0) method adds an extra dimension to the tensor, turning it 
        #       from a zero-dimensional tensor into a one-dimensional tensor.
        encoder_input = torch.cat(
            [
                torch.tensor(self.sos_token, dtype=torch.int32).unsqueeze(0),
                torch.tensor(enc_input_tokens, dtype=torch.int32),
                torch.tensor(self.eos_token, dtype=torch.int32).unsqueeze(0),
                torch.tensor([self.pad_token] * enc_num_padding_tokens, dtype=torch.int32)
            ],
            dim=0
        )

        decoder_input = torch.cat(
            [
                torch.tensor(self.sos_token, dtype=torch.int32).unsqueeze(0),
                torch.tensor(dec_input_tokens, dtype=torch.int32),
                torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int32)
            ],
            dim=0
        )

        label = torch.cat(
            [
                torch.tensor(dec_input_tokens, dtype=torch.int32),
                torch.tensor(self.eos_token, dtype=torch.int32).unsqueeze(0),
                torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int32),
            ],
            dim=0
        )

        assert encoder_input.size(0) == self.seq_len
        assert decoder_input.size(0) == self.seq_len
        assert label.size(0) == self.seq_len

        return {
            "encoder_input": encoder_input, # (Seq_Len)
            "decoder_input": decoder_input, # (Seq_Len)
            "encoder_mask": (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(),  # (1, 1, Seq_Len)
            "decoder_mask": (decoder_input != self.pad_token).unsqueeze(0).int() & causal_mask(decoder_input.size(0)),  # (1, Seq_Len) & (1, Seq_Len, Seq_Len)
            "label": label,  # (Seq_Len)
            "src_text": src_text,
            "tgt_text": tgt_text
        }


# def causal_mask(seq_len):
#     mask = torch.ones(seq_len, seq_len)
#     mask = torch.triu(mask, diagonal=1)
#     return mask.unsqueeze(0).int()  # (1, Seq_Len, Seq_Len)


def causal_mask(seq_len):
    mask = torch.ones(1, seq_len, seq_len)
    mask = torch.triu(mask, diagonal=1).type(torch.int)
    return mask == 0  # (1, Seq_Len, Seq_Len)
