import argparse
import logging
import numpy as np
import os
import sys
import transformers
from datasets import DatasetDict
from pathlib import Path
from transformers import BartConfig, BartForConditionalGeneration, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, \
    Seq2SeqTrainer, AutoTokenizer
from transformers.trainer_utils import get_last_checkpoint, set_seed

from hybrid_textnorm.training import train_tokenizer
from hybrid_textnorm.lexicon import Lexicon

logger = logging.getLogger(__name__)

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)

transformers.utils.logging.set_verbosity_info()
logger.setLevel(logging.INFO)
transformers.utils.logging.set_verbosity(logging.INFO)
transformers.utils.logging.enable_default_handler()
transformers.utils.logging.enable_explicit_format()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--train_file', type=str,
                        default='dataset/processed/train.lexicon.jsonl',
                        help='Path to training file (default: %(default)s')
    parser.add_argument('--eval_file', type=str, default='dataset/processed/dev.lexicon.jsonl',
                        help='Path to evaluation file (default: %(default)s')
    parser.add_argument('--output_dir', type=str, default='model_output',
                        help='Path to model output directory (default: %(default)s)')
    parser.add_argument('--char_vocab', action='store_true', help='Use alphabet as vocabulary (default: False)')
    parser.add_argument('--vocab_size', type=int, default=200, help='Vocabulary size (default: %(default)s)')
    parser.add_argument('--d_model', type=int, default=512, help='Dimension of model (default: %(default)s)')
    parser.add_argument('--ffn_dim', type=int, default=1024,
                        help='Dimension of feed-forward network (default: %(default)s)')
    parser.add_argument('--encoder_layers', type=int, default=6, help='Number of encoder layers (default: %(default)s)')
    parser.add_argument('--encoder_attention_heads', type=int, default=8,
                        help='Number of encoder attention heads (default: %(default)s)')
    parser.add_argument('--decoder_layers', type=int, default=6, help='Number of decoder layers (default: %(default)s)')
    parser.add_argument('--decoder_attention_heads', type=int, default=8,
                        help='Number of decoder attention heads (default: %(default)s)')
    parser.add_argument('--learning_rate', type=float, default=5e-5, help='Learning rate (default: %(default)s)')
    parser.add_argument('--train_batch_size', type=int, default=8,
                        help='Batch size for training (default: %(default)s)')
    parser.add_argument('--eval_batch_size', type=int, default=64,
                        help='Batch size for evaluation (default: %(default)s)')
    parser.add_argument('--num_epochs', type=int, default=5, help='Number of training epochs (default: %(default)s)')

    args = parser.parse_args()

    # set up datasets
    if args.train_file.endswith('jsonl'):
        split_lexicons = {
            'train': Lexicon.from_dataset('json', data_files=str(args.train_file), split='train'),
            'dev': Lexicon.from_dataset('json', data_files=str(args.eval_file), split='train')
        }
    else:
        # legacy format
        import collections, json
        split_lexicons = {}
        with open(args.train_file) as f:
            split_lexicons['train'] = Lexicon({k: collections.Counter(v) for k, v in json.load(f).items()})
        with open(args.eval_file) as f:
            split_lexicons['dev'] = Lexicon({k: collections.Counter(v) for k, v in json.load(f).items()})



    translation_dataset = DatasetDict(
        {split: lexicon.to_dataset(k_most_common=1) for split, lexicon in split_lexicons.items()})

    # load last checkpoint if applicable
    tokenizer = None
    last_checkpoint = None
    if os.path.isdir(args.output_dir):
        last_checkpoint = get_last_checkpoint(args.output_dir)
    if last_checkpoint is not None:
        logger.info(
            f"Checkpoint detected, resuming training at {last_checkpoint}. Will load the tokenizer from {args.output_dir}."
        )
        tokenizer = AutoTokenizer.from_pretrained(args.output_dir)

    # set up tokenizer
    if tokenizer is None:
        tokenizer = train_tokenizer(split_lexicons['train'], vocab_size=args.vocab_size, character_model=args.char_vocab)
        tokenizer.save_pretrained(args.output_dir)

    # tokenize dataset
    def preprocess_dataset(examples):
        inputs = examples['orig']
        targets = examples['norm']
        assert all(' ' not in tok for tok in inputs)
        assert all(' ' not in tok for tok in targets)
        model_inputs = tokenizer(inputs, text_target=targets, max_length=tokenizer.model_max_length, truncation=True)
        return model_inputs

    translation_dataset_tokenized = translation_dataset.map(preprocess_dataset, batched=True)

    # initialize model.
    set_seed(1234)
    config = BartConfig(vocab_size=tokenizer.vocab_size,
                        activation_function="gelu",
                        d_model=args.d_model,
                        max_length=100,
                        max_position_embeddings=1024,
                        encoder_ffn_dim=args.ffn_dim,
                        encoder_layers=args.encoder_layers,
                        encoder_attention_heads=args.encoder_attention_heads,
                        encoder_layerdrop=0,
                        decoder_ffn_dim=args.ffn_dim,
                        decoder_layers=args.decoder_layers,
                        decoder_attention_heads=args.decoder_attention_heads,
                        decoder_layerdrop=0,
                        dropout=0.3,
                        num_beams=4,
                        unk_token_id=tokenizer.unk_token_id,
                        bos_token_id=tokenizer.bos_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                        pad_token_id=tokenizer.pad_token_id,
                        decoder_start_token_id=tokenizer.eos_token_id)

    model = BartForConditionalGeneration(config)
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    def postprocess_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [label.strip() for label in labels]

        return preds, labels

    def compute_metrics(eval_preds):
        preds, labels, inputs = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]

        inputs = np.where(inputs != -100, inputs, tokenizer.pad_token_id)
        decoded_inputs = tokenizer.batch_decode(inputs, skip_special_tokens=True)

        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]

        result = {
            "wordacc": np.mean([a == b for a, b in zip(decoded_preds, decoded_labels)]),
            "wordacc_oov": np.mean([a == b for a, b, i in zip(decoded_preds, decoded_labels, decoded_inputs) if
                                    i not in split_lexicons['train'].keys()]),
            "gen_len": np.mean(prediction_lens)
        }
        result = {k: round(v, 4) for k, v in result.items()}
        return result

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=1,
        num_train_epochs=args.num_epochs,
        predict_with_generate=True,
        log_level='info',
        logging_strategy="steps",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_steps=1 / args.num_epochs / 20,
        # eval_steps=1/args.num_epochs/20,
        # save_steps=1/args.num_epochs/20,
        load_best_model_at_end=True,
        push_to_hub=False,
        metric_for_best_model='eval_wordacc_oov',
        include_inputs_for_metrics=True,
    )
    
    # let's go
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=translation_dataset_tokenized['train'],
        eval_dataset=translation_dataset_tokenized['dev'],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    train_result = trainer.train(resume_from_checkpoint=last_checkpoint)
    trainer.save_model()
    trainer.create_model_card()
    trainer.save_state()

    metrics = train_result.metrics
    metrics["train_samples"] = len(translation_dataset_tokenized['train'])

    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)

    logger.info("*** Evaluate ***")

    metrics = trainer.evaluate(metric_key_prefix="eval")
    metrics["eval_samples"] = len(translation_dataset_tokenized['dev'])

    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)

if __name__ == '__main__':
    main()
