import itertools
import torch
import regex as re
from torch.utils.data import DataLoader
from transformers import DataCollatorForSeq2Seq

from hybrid_textnorm.preprocess import tokens_to_string, recombine_tokens

QUOT_MARKS = re.compile(r'["„“()]')

def make_tokens_to_llm_string(hyp_tokens):
    hyp_str = tokens_to_string(recombine_tokens(hyp_tokens))
    hyp_str = QUOT_MARKS.sub('', hyp_str)
    return hyp_str



def predict_logits(model, tokenizer, model_input, batch_size):
    tokenized = [tokenizer(input_str) for input_str in model_input]
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
    data_loader = DataLoader(tokenized, batch_size=batch_size, collate_fn=data_collator)

    output = []
    for batch in data_loader:
        del batch['labels']
        with torch.no_grad():
            model_out = model(**batch.to(model.device))
            logits = model_out.logits

        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = batch['input_ids'][:, 1:].contiguous()
        # We take the softmax over the logits to convert them to probabilities
        log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)
        # Gather the log probabilities of actual token IDs
        actual_log_probs = torch.gather(log_probs, 2, shift_labels.unsqueeze(-1)).squeeze(-1)
        # filter out the padding
        actual_log_probs = actual_log_probs * (batch['attention_mask'][:, 1:])
        output.extend(actual_log_probs.cpu())
    return output


def hypothesis_ranking(llm_logits, prior_logits):
    return llm_logits.sum() + prior_logits.sum()

def generate_word_segmentation(template_tokens):
    accum = []
    open_word = False
    for hyp_words in template_tokens:
        if not open_word:
            if any(word.endswith('░') for word in hyp_words):
                open_word = True
                accum.append(hyp_words)
            else:
                yield [hyp_words]
        else:
            accum.append(hyp_words)
            if any(word.endswith('░') for word in hyp_words):
                continue
            else:
                yield accum
                open_word = False
                accum = []

    if len(accum) > 0:
        yield accum


def beam_search(constraint_seq, model, tokenizer, num_beams=8, max_fan_out=8, batch_size=8):
    template_tokens = [[tok for tok, logprob in x] for x in constraint_seq]
    template_logprobs = [[logprob for tok, logprob in x] for x in constraint_seq]

    if all(len(x) == 1 for x in template_tokens):
        return [([x[0] for x in template_tokens], 0, 0, 0)]

    word_seq_list = list(generate_word_segmentation(template_tokens))

    # 0. build first beam
    first_beam = []
    depth = 0
    for j in range(0, len(word_seq_list)):
        if all(len(x) == 1 for x in word_seq_list[j]):
            depth += 1
            first_beam.extend([0] * len(word_seq_list[j]))
        else:
            break

    beams = [first_beam]

    while True:
        # 1. determine the immediate successor options
        successors = word_seq_list[depth]
        hyp_indices = list(itertools.product(*[range(len(x)) for x in successors]))

        # if > 8 successors, get the top 8 ones by prior likelihood
        if len(hyp_indices) > max_fan_out:
            hyp_indices_weighted = []
            for comb in hyp_indices:
                prior = torch.tensor([x[i] for x, i in zip(template_logprobs[len(beams[0]):], comb)])
                hyp_indices_weighted.append((comb, sum(prior)))

            hyp_indices_weighted = list(sorted(hyp_indices_weighted, key=lambda x: x[1], reverse=True))
            hyp_indices = [x[0] for x in hyp_indices_weighted[:8]]

        # 2. determine constant suffix after immediate successor
        suffix = []
        suffix_depth = 0
        for j in range(depth+1, len(word_seq_list)):
            if all(len(x) == 1 for x in word_seq_list[j]):
                suffix_depth += 1
                suffix.extend([0] * len(word_seq_list[j]))
            else:
                break

        # 3. determine new successors for each beam
        hypotheses = []
        for beam in beams:
            for comb in hyp_indices:
                hypotheses.append(beam + list(comb) + suffix)


        # 4. for each successor, determine llm likelihood and priors
        in_strs = []
        for hypothesis in hypotheses:
            in_tokens = [ x[i] for x, i in zip(template_tokens, hypothesis) ]
            llm_str = make_tokens_to_llm_string(in_tokens)
            in_strs.append(llm_str)
            # print(llm_str, in_tokens)

        # model_input = tokenizer(in_strs, return_tensors='pt', padding=True)
        hypothesis_llm_logits = predict_logits(model, tokenizer, in_strs, batch_size)

        ranked_hypotheses = []
        for hyp, _llm_logits in zip(hypotheses, hypothesis_llm_logits):
            prior_logits = torch.tensor([ x[i] for x, i in zip(template_logprobs, hyp) ])
            ranked_hypotheses.append((hyp, _llm_logits, prior_logits))

        # 5. rank successor hypotheses, these are our new beams
        ranked_hypotheses = list(sorted(ranked_hypotheses, key=lambda h: hypothesis_ranking(h[1], h[2]), reverse=True))

        if len(ranked_hypotheses[0][0]) == len(template_tokens):
            # done :)
            out = []
            for hyp in ranked_hypotheses:
                token_indices, llm_logits, prior_logits = hyp
                norm_tokens = [ x[i] for x, i in zip(template_tokens, token_indices) ]
                score = hypothesis_ranking(llm_logits, prior_logits)
                out.append((norm_tokens, llm_logits, prior_logits, score))

            out = sorted(out, key=lambda h: h[-1], reverse=True)
            return out
        else:
            # our new beams are the top successor hypotheses; goto 1
            beams = [ hyp[0] for hyp in ranked_hypotheses[:num_beams] ]
            depth = depth + 1 + suffix_depth