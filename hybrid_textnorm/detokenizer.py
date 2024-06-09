import re
from nltk.tokenize.treebank import TreebankWordDetokenizer

# Adapted code from Bracke: https://github.com/ybracke/transnormer-data/blob/main/src/transnormer_data/detokenizer.py


UNICODE_QUOTATION_MARKS_BOTH = [
    "\u0022",  # " -- Typewriter quote, ambidextrous. Also known as \"double quote\".
    # "\u0027",  # ' -- Typewriter straight single quote, ambidextrous
]

UNICODE_QUOTATION_MARKS_END = [
    "\u00AB",  # « -- Double angle quote (chevron, guillemet, duck-foot quote), left
    "\u2018",  # ‘ -- Single curved quote, left. Also known as inverted comma or turned comma[h]
    "\u201C",  # “ -- Double curved quote, left
    "\u2039",  # ‹ -- Single angle quote, left
    "\u201B",  # ‛ -- Also called single reversed comma, quotation mark
    "\u201F",  # ‟ -- Also called double reversed comma, quotation mark
]

UNICODE_QUOTATION_MARKS_START = [
    "\u00BB",  # » -- Double angle quote, right
    # "\u2019",  # ’ -- Single curved quote, right[i]
    "\u201D",  # ” -- Double curved quote, right
    "\u203A",  # › -- Single angle quote, right
    "\u201E",  # „ -- Low double curved quote, left
    "\u201A",  # ‚ -- Low single curved quote, left
    "\u2E42",  # ⹂ -- Also called double low reversed comma, quotation mark
]


class DtaEvalDetokenizer(TreebankWordDetokenizer):
    startq = "|".join(UNICODE_QUOTATION_MARKS_START + UNICODE_QUOTATION_MARKS_BOTH)
    endq = "|".join(UNICODE_QUOTATION_MARKS_END + UNICODE_QUOTATION_MARKS_BOTH)

    # ending quotes
    ENDING_QUOTES = [
        (re.compile(r"(\S)\s(" + endq + r")"), r"\1\2"),
        (
            re.compile(r"(" + endq + r")\s([.,:)\]>};%])"),
            r"\1\2",
        ),  # Quotes followed by no-left-padded punctuations.
    ]

    # starting quotes
    STARTING_QUOTES = [
        (re.compile(r"([ (\[{<])\s(" + startq + r")"), r"\1\2"),
        (re.compile(r"(" + startq + r")\s"), r"\1"),
    ]