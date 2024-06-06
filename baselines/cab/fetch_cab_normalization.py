import time

import json

import argparse
import more_itertools
import requests
from datasets import load_dataset
from tqdm import tqdm


def make_request(rows_batch):
    url = "https://www.deutschestextarchiv.de/public/cab/query?qname=qd&a=default&ifmt=json&ofmt=json"

    request_input = {'body': []}
    for row in rows_batch:
        request_input['body'].append({'tokens': [{'text': t} for t in row['tokens']['orig']]})

    data = [("qd", ("inputfile.json", json.dumps(request_input)))]
    response = requests.post(url, files=data, allow_redirects=False)
    response.raise_for_status()
    json_obj = json.loads(response.text)
    return json_obj

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('test_filename', default='./dataset/processed/test.jsonl', nargs='?')
    parser.add_argument('output_filename', default='./baselines/output/test.cab.pred', nargs='?')
    args = parser.parse_args()

    dataset = load_dataset('json', data_files=args.test_filename, split='train')

    output_file = open(args.output_filename, 'w')
    for rows_batch in more_itertools.batched(tqdm(dataset), n=100):
        response = make_request(rows_batch)
        for ds_row, sent in zip(rows_batch, response['body']):
            cab_orig_tokens = [x['text'] for x in sent['tokens']]
            # NOTE: CAB can add separator marks, but cannot merge tokens
            cab_pred_tokens = [x['moot']['word'].replace('_', '▁').replace(' ', '▁') for x in sent['tokens']]

            for ds_token, cab_orig_tok, cab_pred_tok in zip(ds_row['tokens']['orig'], cab_orig_tokens, cab_pred_tokens):
                assert ds_token == cab_orig_tok
                print(cab_pred_tok, file=output_file, flush=True)

        time.sleep(1)
    output_file.close()




if __name__ == '__main__':
    main()
