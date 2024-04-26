import os
from pathlib import Path
import torch.utils.data
from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn
import torch.utils

from torch.utils.tensorboard import SummaryWriter
import torchmetrics

# Hegging Face
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

from config import get_config, get_weights_file_path, latest_weights_file_path
from dataset import BilingualDataset, causal_mask
from model import build_transformer_model


def greedy_decode(model, source, source_mask, tokenizer_src, tokenizer_tgt, max_len, device):
    sos_idx = tokenizer_tgt.token_to_id('[SOS]')
    eos_idx = tokenizer_tgt.token_to_id('[EOS]')

    # Precompute the encoder output and reuse it for every step
    encoder_output = model.encode(source, source_mask)
    # Initialize the decoder input with the sos token
    decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(source).to(device)
    while True:
        if decoder_input.size(1) == max_len:
            break

        # build mask for target
        decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)

        # calculate output
        out = model.decode(encoder_output, decoder_input, source_mask, decoder_mask)

        # get next token
        prob = model.project(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        decoder_input = torch.cat(
            [decoder_input, torch.empty(1, 1).type_as(source).fill_(next_word.item()).to(device)], dim=1
        )

        if next_word == eos_idx:
            break

    return decoder_input.squeeze(0)


def run_validation(model, validation_ds, tokenizer_src, tokenizer_tgt, max_len, device, print_msg, global_step, writer, num_examples=2):
    model.eval()
    count = 0

    source_texts = []
    expected = []
    predicted = []

    try:
        # get the console window width
        with os.popen('stty size', 'r') as console:
            _, console_width = console.read().split()
            console_width = int(console_width)
    except:
        # If we can't get the console width, use 80 as default
        console_width = 80

    with torch.no_grad():
        for batch in validation_ds:
            count += 1
            encoder_input = batch["encoder_input"].to(device) # (b, seq_len)
            encoder_mask = batch["encoder_mask"].to(device) # (b, 1, 1, seq_len)

            # check that the batch size is 1
            assert encoder_input.size(
                0) == 1, "Batch size must be 1 for validation"

            model_out = greedy_decode(model, encoder_input, encoder_mask, tokenizer_src, tokenizer_tgt, max_len, device)

            source_text = batch["src_text"][0]
            target_text = batch["tgt_text"][0]
            model_out_text = tokenizer_tgt.decode(model_out.detach().cpu().numpy())

            source_texts.append(source_text)
            expected.append(target_text)
            predicted.append(model_out_text)
            
            # Print the source, target and model output
            print_msg('-'*console_width)
            print_msg(f"{f'SOURCE: ':>12}{source_text}")
            print_msg(f"{f'TARGET: ':>12}{target_text}")
            print_msg(f"{f'PREDICTED: ':>12}{model_out_text}")

            if count == num_examples:
                print_msg('-'*console_width)
                break
    
    if writer:
        # Evaluate the character error rate
        # Compute the char error rate 
        metric = torchmetrics.text.CharErrorRate()
        cer = metric(predicted, expected)
        writer.add_scalar('Validation CER', cer, global_step)
        writer.flush()

        # Compute the word error rate
        metric = torchmetrics.text.WordErrorRate()
        wer = metric(predicted, expected)
        writer.add_scalar('Validation WER', wer, global_step)
        writer.flush()

        # Compute the BLEU metric
        metric = torchmetrics.text.BLEUScore()
        bleu = metric(predicted, expected)
        writer.add_scalar('Validation BLEU', bleu, global_step)
        writer.flush()


def get_all_sentences(ds, lang):
    """
    Get all the sentences in the dataset for a given language

    :param ds: The dataset
    :param lang: The language
    :return: A generator that yields the sentences
    """
    for item in ds:
        # The dataset is a dictionary with two fields: 'id' and 'translation'
        yield item['translation'][lang]


def get_or_build_tokenizer(config, ds, lang):
    tokenizer_path = Path(config['training']['tokenizer_file'].format(lang))
    if not Path.exists(tokenizer_path):
        # The first run, build the tokenizer
        print(f'Tokenizer for {lang} language NOT FOUND')
        #
        tokenizer = Tokenizer(WordLevel(unk_token='[UNK]'))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens=['[UNK]', '[SOS]', '[EOS]', '[PAD]'], min_frequency=1,
                                   show_progress=True)
        tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer


def get_ds(config):
    # Download and load the dataset
    ds = load_dataset(config['dataset']['name'], f"{config['dataset']['src_lang']}-{config['dataset']['tgt_lang']}", split='train')

    # Build tokenizers
    tokenizer_src = get_or_build_tokenizer(config, ds, config['dataset']['src_lang'])
    tokenizer_tgt = get_or_build_tokenizer(config, ds, config['dataset']['tgt_lang'])

    # If delete long sentences was set to True, remove the long sentences by filtering the dataset
    if config['dataset']['filter_long_sentences']:
        print("\x1b[31;1m\r\nFiltering long sentences from the dataset...\x1b[0m")
        ds = ds.filter(lambda x: (len(tokenizer_src.encode(x['translation'][config['dataset']['src_lang']]).ids) <= config['model']['seq_len'] - 2) and (len(tokenizer_tgt.encode(x['translation'][config['dataset']['tgt_lang']]).ids) <= config['model']['seq_len'] - 2))

    # Get the maximum sequence length
    src_max_seq_len = 0
    tgt_max_seq_len = 0
    for item in ds:
        src_ids = tokenizer_src.encode(item['translation'][config['dataset']['src_lang']]).ids
        tgt_ids = tokenizer_tgt.encode(item['translation'][config['dataset']['tgt_lang']]).ids
        src_max_seq_len = max(src_max_seq_len, len(src_ids))
        tgt_max_seq_len = max(tgt_max_seq_len, len(tgt_ids))
    print(f"Max source sequence length: {src_max_seq_len} tokens.") # You need to add 2 tokens for the [SOS] and [EOS] tokens calculate the seq_len
    print(f"Max target sequence length: {tgt_max_seq_len} tokens.")

    # Split the dataset 90% train, 10% test
    train_ds_size = int(0.9 * len(ds))
    val_ds_size = len(ds) - train_ds_size
    train_ds_raw, val_ds_raw = torch.utils.data.random_split(ds, [train_ds_size, val_ds_size])
    print(f'\r\nDataset size: {len(ds):7d}')
    print(f'  Train size: {len(train_ds_raw):7d}')
    print(f'   Test size: {len(val_ds_raw):7d}\r\n')

    # Create the datasets
    train_ds = BilingualDataset(train_ds_raw, tokenizer_src, tokenizer_tgt, config['dataset']['src_lang'], 
                                config['dataset']['tgt_lang'], config['model']['seq_len'])
    val_ds = BilingualDataset(val_ds_raw, tokenizer_src, tokenizer_tgt, config['dataset']['src_lang'], 
                              config['dataset']['tgt_lang'], config['model']['seq_len'])

    # Create data loaders
    train_dataloader = torch.utils.data.DataLoader(train_ds, batch_size=config['training']['batch_size'], shuffle=True, pin_memory=True)
    val_dataloader = torch.utils.data.DataLoader(val_ds, batch_size=1, shuffle=True)

    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt


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


def train_model(config):
    # Define the device
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.has_mps or torch.backends.mps.is_available() else "cpu"
    print(f' Using device: \x1b[33;1m{device}\x1b[0m')
    if device == 'cuda':
        print(f'  Device name: \x1b[33;1m{torch.cuda.get_device_name(device.index)}\x1b[0m')
        print(f'Device memory: \x1b[33;1m{torch.cuda.get_device_properties(device.index).total_memory / 1024 ** 3:0.2f} GB\x1b[0m')
    elif device == 'mps':
        print(f'Device name: <mps>')
    else:
        print('NOTE: If you have a GPU, consider using it for training.')
        print('      On a Windows machine with NVidia GPU, check this video: https://www.youtube.com/watch?v=GMSjDTU8Zlc')
        print('      On a Mac machine, run: pip3 install --pre torch torchvision torchaudio torchtext --index-url https://download.pytorch.org/whl/nightly/cpu')
    device = torch.device(device)

    # Ensure that the weights folder is created
    Path(f"{config['dataset']['name']}_{config['training']['model_folder']}").mkdir(parents=True, exist_ok=True)

    # Get the data loaders
    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config)

    # Get the model and move it to the device
    model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)

    # Tensorboard
    writer = SummaryWriter(config['tensorboard']['log_dir'], comment='Transformer Test')

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config['training']['lr0'], 
                                 betas=[config['training']['beta1'], config['training']['beta2']], eps=1e-8)
    # optimizer = torch.optim.SGD(model.parameters(), lr=config["lr0"])

    # Learning rate scheduler
    lambda1 = lambda step: np.minimum(np.power(step, -0.5), step*np.power(config['training']['warmup_steps'], -1.5))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)
    lrs = []

    # Load the model if specified
    initial_epoch = 0
    global_step = -1
    preload = config['training']['preload']
    model_filename = latest_weights_file_path(config) if preload == 'latest' else get_weights_file_path(config, preload) if preload else None
    if model_filename is not None:
        print('Preloading model from ', model_filename)
        checkpoint = torch.load(model_filename)
        model.load_state_dict(checkpoint['model_state_dict'])
        initial_epoch = checkpoint['epoch'] + 1
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        global_step = checkpoint['global_step']
    else:
        print('No model to preload. \x1b[39;1mStarting from scratch!\x1b[0m')

    # Loss function. Ignore the padding tokens
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer_src.token_to_id("[PAD]"), label_smoothing=0.1).to(device)

    for epoch in range(initial_epoch, config['training']['num_epochs']):
        torch.cuda.empty_cache()
        model.train()
        batch_iterator = tqdm(train_dataloader, desc=f'Epoch {epoch:4d}', total=len(train_dataloader))
        for batch in batch_iterator:
            # Move the data to the device
            encoder_input = batch['encoder_input'].to(device)  # (Batch, Seq_Len)
            decoder_input = batch['decoder_input'].to(device)  # (Batch, Seq_Len)
            encoder_mask = batch['encoder_mask'].to(device)  # (Batch, 1, 1, Seq_Len)
            decoder_mask = batch['decoder_mask'].to(device)  # (Batch, 1, Seq_Len, Seq_Len)
            label = batch['label'].to(device)  # (Batch, Seq_Len)
            # src_text = batch['src_text']
            # tgt_text = batch['tgt_text']

            # Forward pass
            encoder_output = model.encode(encoder_input, encoder_mask)  # (Batch, Seq_Len, d_model)
            decoder_output = model.decode(encoder_output, decoder_input, encoder_mask,
                                          decoder_mask)  # (Batch, Seq_Len, d_model)
            proj_output = model.project(decoder_output)  # (Batch, Seq_Len, vocab_tgt_len)

            # Compute the loss using a simple cross entropy
            # NOTE: The label var must be converted to long to match the type of the output
            loss = criterion(proj_output.view(-1, tokenizer_tgt.get_vocab_size()), label.long().view(-1))

            # Log the loss
            batch_iterator.set_postfix({f'loss': f'{loss.item():6.3f}'})
            writer.add_scalar("Loss/train", loss.item(), global_step=global_step)
            writer.flush()

            # Backward pass
            loss.backward()

            # Update the weights
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            lrs.append(optimizer.param_groups[0]["lr"])

            # Update the learning rate
            scheduler.step()
            global_step += 1

        # Validation
        run_validation(model, val_dataloader, tokenizer_src, tokenizer_tgt, config['model']['seq_len'], device,
                    lambda msg: batch_iterator.write(msg), global_step, writer, num_examples=config['training']['num_val_samples'])

        if epoch % config['training']['save_interval'] == 0:
            # Save the model
            model_filename = get_weights_file_path(config, f'{epoch:05d}')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'global_step': global_step
            }, model_filename)


if __name__ == "__main__":
    config = get_config()
    train_model(config)
    print("Training completed successfully")
