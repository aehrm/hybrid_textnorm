import more_itertools
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import DataCollatorForSeq2Seq

from hybrid_textnorm.beam_search import beam_search


def predict_type_normalization(types, type_model_tokenizer, type_model, batch_size=64, num_return_sequences=4, num_beams=4):
    dev_oov_types_tokenized = [{'input_ids': type_model_tokenizer(type)['input_ids']} for type in types]
    data_collator = DataCollatorForSeq2Seq(tokenizer=type_model_tokenizer, model=type_model)
    data_loader = DataLoader(dev_oov_types_tokenized, batch_size=batch_size, collate_fn=data_collator)


    for batch in data_loader:
        type_model.eval()
        batch_inputs = type_model_tokenizer.batch_decode(batch['input_ids'], skip_special_tokens=True)
        with torch.no_grad():
            del batch['labels']
            model_output = type_model.generate(**batch.to(type_model.device), num_return_sequences=num_return_sequences, num_beams=num_beams, return_dict_in_generate=True, output_scores=True)
            pred = type_model_tokenizer.batch_decode(model_output.sequences, skip_special_tokens=True)
        pred_logits = type_model.compute_transition_scores(model_output.sequences, model_output.scores, model_output.beam_indices,
                                                          normalize_logits=False).sum(axis=1).cpu()

        for i, orig_tok in enumerate(batch_inputs):
           probas = dict(
               (norm_tok, norm_proba) for norm_tok, norm_proba in zip(
                   pred[num_return_sequences * i:num_return_sequences * (i + 1)],
                   torch.exp(pred_logits[num_return_sequences * i:num_return_sequences * (i + 1)]).numpy().tolist()
               )
           )

           # merge identical outputs
           it = sorted(probas.items(), key=lambda x: x[0])
           grouper = more_itertools.groupby_transform(it, lambda x: x[0], lambda x: x[1])
           probas = {(k, sum(v)) for k, v in grouper}
           yield orig_tok, probas

def reranked_normalization(orig_tokens, train_lexicon, type_repacement_probabilities, llm_tokenizer, llm_model, alpha=1, beta=1, **kwargs):
    trans = str.maketrans("", "", '░▁')

    alpha = max(1e-5, alpha)
    beta = max(1e-5, beta)

    constraint_seq = []
    for tok in orig_tokens:
        if tok in train_lexicon.keys():
            # get most common normalization
            pred, _ = train_lexicon[tok].most_common(1)[0]

            # then find all candidates that coincide with the most common one modulo spacing/casing
            candidates = []
            for candidate, freq in train_lexicon[tok].items():
                if pred.lower().translate(trans) == candidate.lower().translate(trans):
                    candidates.append([candidate, freq])

            total = sum(freq for _, freq in candidates)
            candidates_scored = [(candidate, 1 / beta * np.log(freq / total)) for candidate, freq in candidates]
            constraint_seq.append(candidates_scored)
        else:
            candidates = type_repacement_probabilities[tok]
            candidates_scored = [(candidate, 1 / alpha * np.log(proba)) for candidate, proba in candidates]
            constraint_seq.append(candidates_scored)

    return beam_search(constraint_seq, llm_model, llm_tokenizer, **kwargs)