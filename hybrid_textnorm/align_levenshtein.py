from hybrid_textnorm.wedit_distance_align import wedit_distance_align
import pickle
import os
import regex as re


def align_token_sequences(tokens_src, tokens_tgt, homogenise=None):
    if homogenise is None:
        homogenise = lambda x: x
    str_src = ' '.join(tokens_src)
    str_tgt = ' '.join(tokens_tgt)
    backpointers = wedit_distance_align(homogenise(str_src), homogenise(str_tgt))

    alignment, current_word, seen1, seen2, last_weight = [], ['', ''], [], [], 0
    for i_ref, i_pred, _ in backpointers:
        if i_ref == 0 and i_pred == 0:
            continue
        # spaces in both, add straight away
        if i_ref <= len(str_src) and str_src[i_ref - 1] == ' ' and i_pred <= len(str_tgt) and str_tgt[i_pred - 1] == ' ':
            alignment.append((current_word[0].strip(), current_word[1].strip()))
            current_word = ['', '']
            seen1.append(i_ref)
            seen2.append(i_pred)
        else:
            end_space = '░'
            if i_ref <= len(str_src) and i_ref not in seen1:
                if i_ref > 0:
                    current_word[0] += str_src[i_ref - 1]
                    seen1.append(i_ref)
            if i_pred <= len(str_tgt) and i_pred not in seen2:
                if i_pred > 0:
                    current_word[1] += str_tgt[i_pred - 1] if str_tgt[i_pred - 1] != ' ' else '▁'
                    end_space = '' if space_after(i_pred, str_tgt) else '░'
                    seen2.append(i_pred)
            if i_ref <= len(str_src) and str_src[i_ref - 1] == ' ' and current_word[0].strip() != '':
                alignment.append((current_word[0].strip(), current_word[1].strip() + end_space))
                current_word = ['', '']
    # final word
    alignment.append((current_word[0].strip(), current_word[1].strip()))
    # check that both strings are entirely covered
    recovered1 = re.sub(' +', ' ', ' '.join([x[0] for x in alignment]))
    recovered2 = re.sub(' +', ' ', ' '.join([x[1] for x in alignment]))

    assert recovered1 == re.sub(' +', ' ', str_src), \
        '\n' + re.sub(' +', ' ', recovered1) + "\n" + re.sub(' +', ' ', str_tgt)
    assert re.sub('[░▁ ]+', '', recovered2) == re.sub('[▁ ]+', '', str_tgt), recovered2 + " / " + str_tgt

    assert [x[0] for x in alignment] == tokens_src

    return alignment

def space_after(idx, sent):
    if idx < len(sent) -1 and sent[idx + 1] == ' ':
        return True
    return False

def space_before(idx, sent):
    if idx > 0 and sent[idx - 1] == ' ':
        return True
    return False
