from pathlib import Path
from config import get_config, latest_weights_file_path 
from model import build_transformer_model
from tokenizers import Tokenizer
from datasets import load_dataset
from dataset import BilingualDataset
import torch
import sys


def get_model(config, vocab_src_len, vocab_tgt_len):
    # Get values from the config
    seq_len = config['model']['seq_len']
    d_model = config['model']['d_model']
    num_encoder_layers = config['model']['num_encoder_layers']
    num_decoder_layers = config['model']['num_decoder_layers']
    num_heads = config['model']['num_heads']
    dim_feedforward = config['model']['dim_feedforward']
    dropout_prob = config['model']['dropout']
    #
    model = build_transformer_model(src_vocab_size=vocab_src_len, 
                                    tgt_vocab_size=vocab_tgt_len,
                                    src_seq_len= seq_len, tgt_seq_len= seq_len,
                                    d_model=d_model,
                                    Nenc=num_encoder_layers,
                                    Ndec=num_decoder_layers,
                                    num_heads=num_heads,
                                    d_ff=dim_feedforward,
                                    dropout=dropout_prob)
    return model


def translate(sentence: str):
    # Define the device, tokenizers, and model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    
    config = get_config()
    
    tokenizer_src = Tokenizer.from_file(str(Path(config['training']['tokenizer_file'].format(config['dataset']['src_lang']))))
    tokenizer_tgt = Tokenizer.from_file(str(Path(config['training']['tokenizer_file'].format(config['dataset']['tgt_lang']))))
    
    # Get the model and move it to the device
    model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)

    # Load the latest weights
    model_filename = latest_weights_file_path(config)
    state = torch.load(model_filename)
    model.load_state_dict(state['model_state_dict'])

    # if the sentence is a number use it as an index to the test set
    label = ""
    if type(sentence) == int or sentence.isdigit():
        id = int(sentence)
        ds = load_dataset(config['dataset']['name'], f"{config['dataset']['src_lang']}-{config['dataset']['tgt_lang']}", split='all')
        ds = BilingualDataset(ds, tokenizer_src, tokenizer_tgt, config['dataset']['src_lang'], config['dataset']['tgt_lang'], config['model']['seq_len'])
        sentence = ds[id]['src_text']
        label = ds[id]['tgt_text']
    seq_len = config['model']['seq_len']

    # Translate the sentence
    model.eval()
    with torch.no_grad():
        # Precompute the encoder output and reuse it for every generation step
        source = tokenizer_src.encode(sentence)
        source = torch.cat([
            torch.tensor([tokenizer_src.token_to_id('[SOS]')], dtype=torch.int32), 
            torch.tensor(source.ids, dtype=torch.int32),
            torch.tensor([tokenizer_src.token_to_id('[EOS]')], dtype=torch.int32),
            torch.tensor([tokenizer_src.token_to_id('[PAD]')] * (seq_len - len(source.ids) - 2), dtype=torch.int32)
        ], dim=0).to(device)
        source_mask = (source != tokenizer_src.token_to_id('[PAD]')).unsqueeze(0).unsqueeze(0).int().to(device)
        encoder_output = model.encode(source, source_mask)

        # Initialize the decoder input with the sos token
        decoder_input = torch.empty(1, 1).fill_(tokenizer_tgt.token_to_id('[SOS]')).type_as(source).to(device)
        
        # Print the source sentence and target start prompt
        if label != "": print(f"{f'ID: ':>12}{id}") 
        print(f"{f'SOURCE: ':>12}{sentence}")
        if label != "": print(f"{f'TARGET: ':>12}{label}") 
        print(f"{f'PREDICTED: ':>12}", end='')

        # Generate the translation word by word
        while decoder_input.size(1) < seq_len:
            # build mask for target and calculate output
            decoder_mask = torch.triu(torch.ones((1, decoder_input.size(1), decoder_input.size(1))), diagonal=1).type(torch.int).type_as(source_mask).to(device)
            # decoder_mask = torch.ones(1, seq_len, seq_len)
            # decoder_mask = torch.tril(decoder_mask).type(torch.int)
            # decoder_mask = decoder_mask.to(device)

            # Calculate the output
            out = model.decode(encoder_output, decoder_input, source_mask, decoder_mask)

            # project next token
            prob = model.project(out[:, -1])
            _, next_word = torch.max(prob, dim=1)
            decoder_input = torch.cat(
                [decoder_input, torch.empty(1, 1).type_as(source).fill_(next_word.item()).to(device)], dim=1
            )

            # print the translated word
            print(f"{tokenizer_tgt.decode([next_word.item()])}", end=' ')

            # break if we predict the end of sentence token
            if next_word == tokenizer_tgt.token_to_id('[EOS]'):
                break

    # convert ids to tokens
    return tokenizer_tgt.decode(decoder_input[0].tolist())


if __name__=="__main__":
    #read sentence from argument
    translate(sys.argv[1] if len(sys.argv) > 1 else "Mi nombre es Pepito.")
