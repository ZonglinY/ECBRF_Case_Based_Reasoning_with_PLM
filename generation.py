""" Conditional text generation with the auto-regressive models of the library (Bart/T5)
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"]="2"
sys.path.insert(0, "..")
from utils_TST import (set_seed, split_into_words,
                   load_conceptnet_noCase, load_atomic_noCase, load_shakes_withCase,
                   load_e2e_withCase,
                   tokenize_and_encode, add_special_tokens,
                   load_conceptnet_withCase_withMidPrompt, load_atomic_withCase_withMidPrompt,
                   pre_process_datasets_withCases_Bart_or_GPT2,
                   remove_stop_words_nltk)
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers import (BartForConditionalGeneration, BartTokenizer, BartConfig)
# from transformers import (T5ForConditionalGeneration, T5Tokenizer, T5Config)
# from tokenizers import decoders
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
import argparse
import logging
from tqdm import trange
import random
import pickle
import time
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

from nltk import bleu
from nltk.translate.bleu_score import SmoothingFunction


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

MAX_LENGTH = int(10000)  # Hardcoded max length to avoid infinite loop

# 't5-small': (T5ForConditionalGeneration, T5Tokenizer)
# 't5-base': (T5ForConditionalGeneration, T5Tokenizer)
MODEL_CLASSES = {
    'gpt2-lmhead': (GPT2LMHeadModel, GPT2Tokenizer),
    'bart-base': (BartForConditionalGeneration, BartTokenizer)
}

# MODEL_CLASSES = {
#     'bart-base': (BartForConditionalGeneration, BartTokenizer)
# }

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", default=None, type=str, required=True,
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument("--length", type=int, default=20, help="max length of generation")
    parser.add_argument("--is_greedy", action='store_true',
                        help="Use greedy decoding or topk/topp.")
    parser.add_argument("--test_dataset", type=str, nargs="+", default=["./Data/conceptnet/test_CN_sorted.txt"])

    parser.add_argument("--add_prefix", action="store_true",
                        help="add a prefix at the beginning of each input when train with multiple dataset")
    parser.add_argument("--add_separator", action="store_true", help="add <sep> between sub/rel/obj")
    parser.add_argument("--predict_part", type=str, default="obj", choices=["sub", "rel", "obj", "all"],
                        help="predict which part of the triples")
    parser.add_argument("--toy", action='store_true',
                        help="Use toy dataset for debug")

    parser.add_argument("--max_e1", type=int, default=24)
    parser.add_argument("--max_r", type=int, default=10)
    parser.add_argument("--max_e2", type=int, default=36)

    parser.add_argument("--rel_lang", action='store_true',
                        help="Use natural language for relations.")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--top_p", type=float, default=0.0)
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--eval_batch_size', type=int, default=1)
    # ADDED
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--if_without_case", action="store_true", help="Filter all cases as '', to compare the effect of cases")
    # dataset_selection: 0: conceptnet 1: atomic 2: Shakespeare text style transfer
    parser.add_argument("--dataset_selection", type=int, default=0)
    # parser.add_argument('--if_atomic', action='store_true', help='if use atomic or conceptnet')
    parser.add_argument("--if_randomCase", action="store_true", help="if use random cases")
    parser.add_argument("--max_additional_cases", type=int, default=150)
    parser.add_argument("--use_special_tokens_to_split_retrieved_cases", action="store_true", help="<split_cases> and <split_source/target>")
    parser.add_argument("--if_with_strt_mid_promp", action="store_true", help="if use 'Here are some similar cases to infer from: ' and 'Now you can infer: '")
    parser.add_argument("--num_cases", type=int, default=3)
    parser.add_argument("--if_only_use_retrieved_target", action="store_true")
    parser.add_argument("--if_only_use_relation_and_retrieved_target", action="store_true")
    parser.add_argument("--if_use_relation_for_shakes", action="store_true", help="Whether use relation for shakes dataset (Shakespeare's style is)")
    # subset_selection: 0~6, -1 means not using subset
    parser.add_argument("--subset_selection", type=int, default=-1)
    parser.add_argument("--if_not_adding_special_relation_tokens", action="store_true", help="not adding <oReact> for instance")
    parser.add_argument("--if_use_full_memory_store_while_subset", action="store_true", help="if use full memory store during retrieval (no matter if using subset)")
    parser.add_argument("--beam_size", type=int, default=0)
    parser.add_argument("--if_without_none", action="store_true", help="whether to filter 'None' data while using atomic dataset")
    parser.add_argument("--length_penalty", type=float, default=1.0)
    parser.add_argument("--if_not_with_strt_mid_promp", action="store_true")
    parser.add_argument("--BLEU_n", type=int, default=2, help="metric to use (if BLEU_n = 2, then we are using BLEU-2 metric)")
    parser.add_argument("--num_sample", type=int, default=1, help="the nth time of sampling data to use; only useful when if_use_full_memory_store_while_subset")
    parser.add_argument("--if_val", type=int, default=0, help="0: use test set; 1: use validation set")
    parser.add_argument("--if_ECBRF", type=int, default="1", help="0: to run COMET baseline; 1: to run ECBRF")


    args = parser.parse_args()
    assert not (args.if_use_full_memory_store_while_subset and args.if_without_case)
    assert not (args.beam_size and args.is_greedy)
    assert not (args.beam_size and args.top_p)
    assert not (args.is_greedy and args.top_p)
    assert not (args.beam_size and args.top_k)
    assert not (args.is_greedy and args.top_k)
    assert (args.is_greedy or args.beam_size or args.top_p or args.top_k)
    assert args.if_val == 0 or args.if_val == 1
    assert args.if_ECBRF == 0 or args.if_ECBRF == 1

    if args.if_without_none:
        if not args.dataset_selection == 1:
            raise Exception("Only ATOMIC dataset need to set 'if_without_none' to truth.")

    if args.if_ECBRF:
        args.if_with_strt_mid_promp = True
        args.if_not_with_strt_mid_promp = False
        args.if_without_case = False
    else:
        args.if_with_strt_mid_promp = False
        args.if_not_with_strt_mid_promp = True
        args.if_without_case = True

    if args.dataset_selection == 0:
        if "t5" in args.model_type:
            args.max_e2 = 46
        if args.if_val == 1:
            args.test_dataset = ["./Data/conceptnet/dev1_CN_sorted.txt"]
        else:
            args.test_dataset = ["./Data/conceptnet/test_CN_sorted.txt"]
    elif args.dataset_selection == 1:
        args.max_e1 = 25
        args.max_r = 15
        args.max_e2 = 38
        # Q: changed from 200 to 250: 8/21/2021: 11:50 p.m.
        args.max_additional_cases = 250
        if args.if_val == 1:
            args.test_dataset = ["./Data/atomic/v4_atomic_dev.csv"]
        else:
            args.test_dataset = ["./Data/atomic/v4_atomic_tst.csv"]
        root_data_dir = "./Data/atomic/"
        if not args.if_without_none == True:
            args.if_without_none = True
            # print("Warning: args.if_without_none == False, and we change args.if_without_none == True before generation.")
            # raise Exception("we only test the results on the test instances whose e2 is not None")
    elif args.dataset_selection == 2:
        args.rel_lang = False
        args.max_e1 = 130
        if args.if_use_relation_for_shakes:
            args.max_r = 6
        else:
            args.max_r = 2
        args.max_e2 = 140
        args.max_additional_cases = 500
        args.test_dataset = ["./Data/shakes/"]
        if args.if_val == 1:
            raise NotImplementError
    elif args.dataset_selection == 3:
        args.rel_lang = False
        args.max_e1 = 60
        args.max_r = 2
        args.max_e2 = 95
        args.max_additional_cases = 400
        args.test_dataset = ["./Data/e2e/"]
        if args.if_val == 1:
            raise NotImplementError
    else:
        raise Exception
    # elif args.if_randomCase:
    #     print('Uses random cases for ConceptNet')

    # adjust args.max_additional_cases according to args.num_cases  ; 8/21/2021
    # in case num_cases is too large
    # This 3 lines of code significantly improve the performance... 8/21/2021
    if not args.max_additional_cases >= args.num_cases * (args.max_e1 + args.max_r + args.max_e2):
        args.max_additional_cases = args.num_cases * (args.max_e1 + args.max_r + args.max_e2)
        print("Adjusted args.max_additional_cases to fit args.num_cases  : ", args.max_additional_cases)
    # To test whether increasingly add max_additional_cases when max_additional_cases is enough will affect the peroformance (it will)
    # if not args.max_additional_cases >= (args.num_cases+2) * (args.max_e1 + args.max_r + args.max_e2):
    #     args.max_additional_cases = (args.num_cases+2) * (args.max_e1 + args.max_r + args.max_e2)
    #     print("Adjusted args.max_additional_cases to fit args.num_cases  : ", args.max_additional_cases)

    if args.if_use_full_memory_store_while_subset:
        if args.if_val:
            args.test_cases_dir = os.path.join(args.output_dir, "ttl_similar_cases_eval_fullMS.txt")
        else:
            args.test_cases_dir = os.path.join(args.output_dir, "ttl_similar_cases_test_fullMS.txt")
        # also consider the old versions
        if not os.path.exists(args.test_cases_dir):
            # change the name for a different sample (to differentiate different ttl_similar_cases for different sample)
            if args.num_sample != 1:
                args.test_cases_dir = args.test_cases_dir.split('.')
                assert len(args.test_cases_dir) == 2
                args.test_cases_dir[0] += '_sample' + str(args.num_sample)
                args.test_cases_dir = '.'.join(args.test_cases_dir)
            assert os.path.exists(args.test_cases_dir)
    else:
        if args.if_val:
            args.test_cases_dir = os.path.join(args.output_dir, "ttl_similar_cases_eval.txt")
        else:
            args.test_cases_dir = os.path.join(args.output_dir, "ttl_similar_cases_test.txt")
    if not args.if_without_case:
        assert os.path.exists(args.test_cases_dir)

    assert args.predict_part == "obj"

    ## additional_names to write file
    # addi_dataset_name
    if args.dataset_selection == 0:
        addi_dataset_name = 'conceptnet'
    elif args.dataset_selection == 1:
        addi_dataset_name = 'atomic'
    elif args.dataset_selection == 2:
        addi_dataset_name = 'shakespeare'
    elif args.dataset_selection == 3:
        addi_dataset_name = 'e2e'
    # addi_susbet_name
    if args.subset_selection == -1:
        addi_susbet_name = 'full'
    else:
        addi_susbet_name = str(args.subset_selection)
    # addi_method_name
    if args.if_without_case:
        addi_method_name = 'comet'
    else:
        addi_method_name = 'cbr'
    additional_names = addi_dataset_name + '_' + addi_susbet_name + '_' + addi_method_name + '_'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    if not n_gpu >= 1:
        print("n_gpu is 0; covering...")
        time.sleep(600)
        raise Exception("n_gpu:{}".format(n_gpu))
    set_seed(args.seed)
    print(args)

    args.model_type = args.model_type.lower()

    if 'gpt2' in args.model_type:
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        model = GPT2LMHeadModel.from_pretrained('gpt2')
    elif args.model_type == 'bart-base':
        tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
        model = BartForConditionalGeneration.from_pretrained("facebook/bart-base")
    elif args.model_type == 't5-base':
        tokenizer = T5Tokenizer.from_pretrained("t5-base")
        model = T5ForConditionalGeneration.from_pretrained("t5-base")
    elif args.model_type == 't5-small':
        tokenizer = T5Tokenizer.from_pretrained("t5-small")
        model = T5ForConditionalGeneration.from_pretrained("t5-small")
    else:
        raise Exception('illegal model_type')

    tokenizer = add_special_tokens(args, tokenizer)
    # N: newly added, to generate complete word
    # tokenizer.decoder = decoders.WordPiece()

    model.resize_token_embeddings(len(tokenizer))
    model.load_state_dict(torch.load(os.path.join(args.output_dir, 'generator_final.pt'), map_location='cuda:0'))
    print("Loaded model successfully")
    model.to(device)
    model.eval()

    eos_token = tokenizer.eos_token
    pad_token = tokenizer.pad_token
    if 'gpt2' in args.model_type:
        eos_token_id = tokenizer.encode(eos_token)[0]
        pad_token_id = tokenizer.encode(pad_token)[0]
    elif 'bart' in args.model_type:
        eos_token_id = tokenizer.encode(eos_token)[1]
        pad_token_id = tokenizer.encode(pad_token)[1]
    elif 't5' in args.model_type:
        eos_token_id = tokenizer.encode(eos_token)[-1]
        pad_token_id = tokenizer.encode(pad_token)[0]
    print("\nspecial tokens:", tokenizer.special_tokens_map)

    def prefix_mapping(filename):
        if "vg" in filename.lower():
            return "<from_VG>"
        elif "cn" in filename.lower():
            return "<from_CN>"
        elif "fb" in filename.lower():
            return "<from_FB>"

    print("Loading dataset...")
    if args.dataset_selection == 0:
        test_datasets = [load_conceptnet_withCase_withMidPrompt(args,
                                             dataset_path=test_dataset,
                                             cases_path=args.test_cases_dir,
                                             cls_token=tokenizer.cls_token,
                                             eos_token=tokenizer.eos_token,
                                             sep_token=tokenizer.sep_token,
                                             rel_lang=args.rel_lang,
                                             toy=args.toy,
                                             discard_negative=True,
                                             add_sep=args.add_separator,
                                             prefix=prefix_mapping(test_dataset) if args.add_prefix else None,
                                             model_type = args.model_type,
                                             if_without_case=args.if_without_case,
                                             num_cases = args.num_cases
                                            ) for test_dataset in args.test_dataset]
    elif args.dataset_selection == 1:
        # if args.subset_selection == -1:
         test_datasets = [load_atomic_withCase_withMidPrompt(args=args,
                                              dataset_path=test_dataset,
                                              cases_path=args.test_cases_dir,
                                              cls_token=tokenizer.cls_token,
                                              eos_token=tokenizer.eos_token,
                                              sep_token=tokenizer.sep_token,
                                              rel_lang=args.rel_lang,
                                              add_sep=args.add_separator,
                                              prefix=prefix_mapping(test_dataset) if args.add_prefix else None,
                                              model_type = args.model_type
                                             ) for test_dataset in args.test_dataset]
    elif args.dataset_selection == 2:
        test_datasets = [load_shakes_withCase(args=args,
                                             dataset_path=test_dataset,
                                             cases_path=args.test_cases_dir,
                                             cls_token=tokenizer.cls_token,
                                             eos_token=tokenizer.eos_token,
                                             sep_token=tokenizer.sep_token,
                                             rel_lang=args.rel_lang,
                                             add_sep=args.add_separator,
                                             prefix=prefix_mapping(test_dataset) if args.add_prefix else None,
                                             model_type = args.model_type,
                                             if_without_case=args.if_without_case
                                            ) for test_dataset in args.test_dataset]
    elif args.dataset_selection == 3:
        test_datasets = [load_e2e_withCase(args=args,
                                             dataset_path=test_dataset,
                                             cases_path=args.test_cases_dir,
                                             cls_token=tokenizer.cls_token,
                                             eos_token=tokenizer.eos_token,
                                             sep_token=tokenizer.sep_token,
                                             rel_lang=args.rel_lang,
                                             add_sep=args.add_separator,
                                             prefix=prefix_mapping(test_dataset) if args.add_prefix else None,
                                             model_type = args.model_type,
                                             if_without_case=args.if_without_case
                                            ) for test_dataset in args.test_dataset]


    test_datasets = [data for test_dataset in test_datasets for data in test_dataset]
    print("test_datasets[0]")
    print(test_datasets[0])
    datasets = [test_datasets]

    logger.info("Encoding dataset...")
    encoded_datasets = tokenize_and_encode(datasets, tokenizer, args.model_type)
    print("Encoding Done!")

    max_e1 = args.max_e1
    max_r = args.max_r
    max_e2 = args.max_e2

    if args.if_not_with_strt_mid_promp:
        max_additional_cases = args.max_additional_cases
    else:
        if "bart" in args.model_type:
            encoded_strt_prompt = tokenizer.encode('Here are some similar cases to infer from: ')[1:-1]
            encoded_mid_prompt = tokenizer.encode('With the similar cases we can infer that: ')[1:-1]
            # encoded_sep_token = tokenizer.encode(tokenizer.sep_token)[1:-1]
            # max_additional_cases = len(encoded_strt_prompt) + args.max_additional_cases + len(encoded_mid_prompt) + len(encoded_sep_token)
            max_additional_cases = len(encoded_strt_prompt) + args.max_additional_cases + len(encoded_mid_prompt)
        elif "gpt2" in args.model_type:
            encoded_strt_prompt = tokenizer.encode('Here are some similar cases to infer from: ')
            encoded_mid_prompt = tokenizer.encode('With the similar cases we can infer that: ')
            # encoded_sep_token = tokenizer.encode(tokenizer.sep_token)
            # max_additional_cases = len(encoded_strt_prompt) + args.max_additional_cases + len(encoded_mid_prompt) + len(encoded_sep_token)
            max_additional_cases = len(encoded_strt_prompt) + args.max_additional_cases + len(encoded_mid_prompt)
        else:
            raise NotImplementedError

    if 'gpt2' in args.model_type or 't5' in args.model_type:
        encoded_pad_token=tokenizer.encode(tokenizer.pad_token)[0]
    elif 'bart' in args.model_type:
        encoded_pad_token=tokenizer.encode(tokenizer.pad_token)[1]
    else:
        raise NotImplementedError

    tensor_datasets = pre_process_datasets_withCases_Bart_or_GPT2(encoded_datasets, max_e1, max_r, max_e2, \
            max_additional_cases, tokenizer=tokenizer, predict_part=args.predict_part, model_type=args.model_type, \
            encoded_pad_token=encoded_pad_token)
    print("Pre-processing Done!")

    test_tensor_dataset = tensor_datasets[0]
    print(len(test_tensor_dataset))
    ## get references
    refs = {}
    def decode_and_remove_eos(sent):
        sent = [word for word in sent if word > 0]  #remove padding
        try:
            eos_token_pos = sent.index(eos_token_id)
            sent = sent[:eos_token_pos]
        except:
            pass
        sent = tokenizer.decode(sent, clean_up_tokenization_spaces=True)
        if isinstance(sent, list):
            sent = sent[0]
        sent = sent.replace(tokenizer.pad_token, '')
        sent = sent.replace("<split_cases>", '')
        sent = sent.replace("<split_source/target>", '')
        return sent
    # decode_keeping_special_tokens: for debugging --- whether the input and labels are correctly augmented with <s> and </s>
    def decode_keeping_special_tokens(sent):
        return tokenizer.decode(sent, clean_up_tokenization_spaces=True)

    input_id_ary = test_tensor_dataset[0]
    label_id_ary = test_tensor_dataset[1]
    for id_item in range(input_id_ary.size()[0]):
        input_id = torch.Tensor.numpy(input_id_ary[id_item])
        label_id = torch.Tensor.numpy(label_id_ary[id_item])
        e1 = input_id[max_additional_cases : max_additional_cases + max_e1]
        if args.add_prefix:
            e1 = e1[1:]
        e1 = decode_and_remove_eos(e1)
        rel = input_id[max_additional_cases + max_e1 : max_additional_cases + max_e1 + max_r]
        rel = decode_and_remove_eos(rel)
        e2 = label_id[:max_e2]
            # e2 = label_id[max_additional_cases + max_e1 + max_r : max_additional_cases + max_e1 + max_r + max_e2]
        e2 = decode_and_remove_eos(e2)
        if "gpt2" in args.model_type or "bart" in args.model_type:
            e1 = e1.strip()
            rel = rel.strip()
            e2 = e2.strip()
        refs.setdefault(e1, {})
        refs[e1][rel] = refs[e1].get(rel, []) + [e2.strip()]
    # in atomic and conceptnet, we also test the bleu after filtering "to, be, a, ...."
    if args.dataset_selection == 1 or args.dataset_selection == 0:
        filtered_refs = {}
        for tmp_e1 in refs:
            if tmp_e1 not in filtered_refs:
                filtered_refs[tmp_e1] = {}
            for tmp_rel in refs[tmp_e1]:
                if tmp_rel not in filtered_refs[tmp_e1]:
                    filtered_refs[tmp_e1][tmp_rel] = []
                for tmp_e2 in refs[tmp_e1][tmp_rel]:
                    filtered_refs[tmp_e1][tmp_rel].append(remove_stop_words_nltk(tmp_e2))



    ## make dataloader
    test_data = TensorDataset(*test_tensor_dataset)
    test_sampler = SequentialSampler(test_data)
    # test_sampler = RandomSampler(test_data)
    test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=args.eval_batch_size)

    # to calculate BLEU
    # Set score
    n = args.BLEU_n
    weights = [1/n] * n
    def score(hyp, refs):
        # print(hyp)
        # print(refs)
        return bleu(refs, hyp, weights=weights, smoothing_function=SmoothingFunction().method1)

    example_true_bl = []
    example_true_generation = []
    example_count = 0
    if args.dataset_selection == 1 or args.dataset_selection == 0:
        example_true_bl_filtered = []
    model.eval()
    results = []
    dict_existed_generation = {}
    cnt_continue_times = 0
    cnt_generation_times = 0

    if args.toy:
        tmp_id_selection = np.arange(len(test_dataloader))
        random.shuffle(tmp_id_selection)
        tmp_id_selection = tmp_id_selection[:500]
        tmp_id_selection.sort()
    for step, batch in tqdm(enumerate(test_dataloader)):
        if args.toy:
            if step not in tmp_id_selection:
                continue
        batch = tuple([t.to(device) for t in batch[0:3]])
        input_ids = torch.Tensor.numpy(batch[0].cpu())[0]
        labels = torch.Tensor.numpy(batch[1].cpu())[0]
        if step == 0:
            print("input_ids.shape: ", input_ids.shape)
            print("labels.shape: ", labels.shape)
            print("input_ids: ", input_ids)
            print("labels: ", labels)
        ## get e1 and r
        # e1
        e1 = input_ids[max_additional_cases : max_additional_cases + max_e1]
        if args.add_prefix:
            e1 = e1[1:]
        e1_with_special_tokens = decode_keeping_special_tokens(e1)
        e1 = decode_and_remove_eos(e1)
        # r
        r = input_ids[max_additional_cases + max_e1 : max_additional_cases + max_e1 + max_r + 2]
        r_with_special_tokens = decode_keeping_special_tokens(r)
        r = decode_and_remove_eos(r)
        if e1 not in dict_existed_generation:
            dict_existed_generation[e1] = {}
        # when e1 and r has been used for prediction
        if r in dict_existed_generation[e1]:
            cnt_continue_times += 1
            continue
        cnt_generation_times += 1
        # GENERATION
        num_return_sequences = 1
        e2_with_special_tokens = tokenizer.eos_token
        while e2_with_special_tokens == tokenizer.eos_token:
            if 'bart' in args.model_type or 't5' in args.model_type:
                if args.is_greedy:
                    out = model.generate(batch[0], attention_mask=batch[2], max_length=args.max_e2, early_stopping=True, num_return_sequences=num_return_sequences, length_penalty=args.length_penalty, temperature=args.temperature)
                elif args.beam_size:
                    out = model.generate(batch[0], attention_mask=batch[2], num_beams=args.beam_size, max_length=args.max_e2, early_stopping=True, num_return_sequences=num_return_sequences, length_penalty=args.length_penalty, temperature=args.temperature)
                elif args.top_p or args.top_k:
                    out = model.generate(batch[0], attention_mask=batch[2], do_sample=True, top_p=args.top_p, top_k=args.top_k, max_length=args.max_e2, num_return_sequences=num_return_sequences, length_penalty=args.length_penalty, temperature=args.temperature)
                else:
                    raise NotImplementError
            elif 'gpt2' in args.model_type:
                if args.is_greedy:
                    out = model.generate(batch[0], attention_mask=batch[2], max_length=max_additional_cases+max_e1+max_r+max_e2, early_stopping=True, num_return_sequences=num_return_sequences)
                elif args.beam_size:
                    out = model.generate(batch[0], attention_mask=batch[2], num_beams=args.beam_size, max_length=max_additional_cases+max_e1+max_r+max_e2, early_stopping=True, num_return_sequences=num_return_sequences)
                elif args.top_p or args.top_k:
                    out = model.generate(batch[0], attention_mask=batch[2], do_sample=True, top_p=args.top_p, top_k=args.top_k, max_length=max_additional_cases+max_e1+max_r+max_e2, num_return_sequences=num_return_sequences, length_penalty=args.length_penalty, temperature=args.temperature)
                else:
                    raise NotImplementError
            else:
                raise NotImplementError

            # predict e2
            # Q: here gpt2 should be the same with bart (out_id = out[:max_e2])
            if num_return_sequences == 1:
                out = out[0].cpu()
            else:
                raise NotImplementError
            if "gpt2" in args.model_type:
                out_id = out[max_additional_cases+max_e1+max_r:max_additional_cases+max_e1+max_r+max_e2]
                # out_id = out[:max_e2]
            elif "bart" in args.model_type:
                out_id = out[:max_e2]
            else:
                raise NotImplementError
            # print("out_id: ", out_id)
            e2_with_special_tokens = tokenizer.decode(out_id).strip()
            # print("e2_with_special_tokens: ", e2_with_special_tokens)
            # when args.is_greedy == True, regeneration should only generate the same text, making it useless for a while loop for regeneration
            if e2_with_special_tokens == tokenizer.eos_token:
                if args.is_greedy:
                    print("Warning: only generate eos token, but using greedy decoding --- so that regeneration while loop is turned off.")
                    break
        e2 = tokenizer.decode(out_id, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        if "gpt2" in args.model_type or "bart" in args.model_type:
            e2 = e2.replace(tokenizer.eos_token, '')
            e2 = e2.replace(tokenizer.sep_token, '')
            e2 = e2.replace('<split_cases>', '')
            e2 = e2.replace('<split_source/target>', '')
        # write the prediction to dict_existed_generation[e1][r]
        dict_existed_generation[e1][r] = e2

        # additional_cases
        if "gpt2" in args.model_type:
            additional_cases = input_ids[1:max_additional_cases]
        elif "bart" in args.model_type:
            additional_cases = input_ids[:max_additional_cases]
        else:
            raise NotImplementError
        additional_cases_with_special_tokens = decode_keeping_special_tokens(additional_cases)
        additional_cases = decode_and_remove_eos(additional_cases)

        # truth
        truth = labels[:max_e2].tolist()
        truth = decode_and_remove_eos(truth)
        if "gpt2" in args.model_type or "bart" in args.model_type:
            e1 = e1.strip()
            r = r.strip()
            e2 = e2.strip()
        # true_ref
        true_ref = refs[e1][r]
        if args.dataset_selection == 1 or args.dataset_selection == 0:
            true_ref_filtered = filtered_refs[e1][r]
        # according to ATOMIC's paper and COMET's code, filter here out to calculate BLEU
        if sum([i == ["none"] for i in true_ref]) / len(true_ref) > 1/3:
            # only ATOMIC experiment can reach here
            print('None inference overlooked')
            if not args.dataset_selection == 1:
                raise ValueError
            continue
        print('additional_cases: ', additional_cases, 'e1:', e1, 'r:', r, 'e2 prediction:', e2)
        # print('additional_cases_with_special_tokens: ', additional_cases_with_special_tokens, 'e1_with_special_tokens:', e1_with_special_tokens, 'r_with_special_tokens:', r_with_special_tokens, 'e2_with_special_tokens prediction:', e2_with_special_tokens)
        print('cur_reference:', truth, 'true_reference:', true_ref)
        results.append(
            {'additional_cases: ': additional_cases, 'e1': e1, 'r': r, 'sequence': e2, 'cur_reference': truth, 'true_reference:': true_ref})

        if args.eval_batch_size == 1:
            print('true_score: ', score(e2.strip().split() if e2.strip().split() else 'none', [t.strip().split() for t in true_ref]))
            example_true_bl.append(score(e2.strip().split() if e2.strip().split() else 'none', [t.strip().split() for t in true_ref]))
            if args.dataset_selection == 1 or args.dataset_selection == 0:
                e2_filtered = remove_stop_words_nltk(e2)
                example_true_bl_filtered.append(score(e2_filtered.strip().split() if e2_filtered.strip().split() else 'none', [t.strip().split() for t in true_ref_filtered]))
            example_count += 1
            example_true_generation.append([e1, r, e2, true_ref])
        else:
            raise NotImplementedError

    print("INFO: args.toy: ", args.toy)
    print("cnt_continue_times: {}, cnt_generation_times: {}".format(cnt_continue_times, cnt_generation_times))
    print('sum(example_true_bl): {}, example_count: {}'.format(sum(example_true_bl), example_count))
    print('Ave true_BLEU-2:', sum(example_true_bl)/example_count)
    if args.dataset_selection == 1 or args.dataset_selection == 0:
        print('sum(example_true_bl_filtered): {}, example_count: {}'.format(sum(example_true_bl_filtered), example_count))
        print('Ave true_BLEU-2:', sum(example_true_bl_filtered)/example_count)

    if args.if_use_full_memory_store_while_subset:
        with open(os.path.join(args.output_dir, additional_names + 'results_fullMS_toy_{}_isGreedy_{}_beamSize_{}_topp_{}_topk_{}_lengthPenalty_{}_temperature_{}_2nfltrs_{}.pkl'.format(args.toy, args.is_greedy, args.beam_size, int(args.top_p * 10), args.top_k, int(args.length_penalty * 10), int(args.temperature * 10), args.if_val)), "wb") as output_file:
            pickle.dump(results, output_file)
        with open(os.path.join(args.output_dir, additional_names +  'bleu_by_example_fullMS_toy_{}_isGreedy_{}_beamSize_{}_topp_{}_topk_{}_lengthPenalty_{}_temperature_{}_2nfltrs_{}.pkl'.format(args.toy, args.is_greedy, args.beam_size, int(args.top_p * 10), args.top_k, int(args.length_penalty * 10), int(args.temperature * 10), args.if_val)), 'wb') as f:
            pickle.dump(example_true_bl, f)
        with open(os.path.join(args.output_dir, additional_names +  'generation_by_example_fullMS_toy_{}_isGreedy_{}_beamSize_{}_topp_{}_topk_{}_lengthPenalty_{}_temperature_{}_2nfltrs_{}.pkl'.format(args.toy, args.is_greedy, args.beam_size, int(args.top_p * 10), args.top_k, int(args.length_penalty * 10), int(args.temperature * 10), args.if_val)), 'wb') as f:
            pickle.dump(example_true_generation, f)
    else:
        with open(os.path.join(args.output_dir, additional_names + 'results_toy_{}_isGreedy_{}_beamSize_{}_topp_{}_topk_{}_lengthPenalty_{}_temperature_{}_2nfltrs_{}.pkl'.format(args.toy, args.is_greedy, args.beam_size, int(args.top_p * 10), args.top_k, int(args.length_penalty * 10), int(args.temperature * 10), args.if_val)), "wb") as output_file:
            pickle.dump(results, output_file)
        with open(os.path.join(args.output_dir, additional_names + 'bleu_by_example_toy_{}_isGreedy_{}_beamSize_{}_topp_{}_topk_{}_lengthPenalty_{}_temperature_{}_2nfltrs_{}.pkl'.format(args.toy, args.is_greedy, args.beam_size, int(args.top_p * 10), args.top_k, int(args.length_penalty * 10), int(args.temperature * 10), args.if_val)), 'wb') as f:
            pickle.dump(example_true_bl, f)
        with open(os.path.join(args.output_dir, additional_names +  'generation_by_example_toy_{}_isGreedy_{}_beamSize_{}_topp_{}_topk_{}_lengthPenalty_{}_temperature_{}_2nfltrs_{}.pkl'.format(args.toy, args.is_greedy, args.beam_size, int(args.top_p * 10), args.top_k, int(args.length_penalty * 10), int(args.temperature * 10), args.if_val)), 'wb') as f:
            pickle.dump(example_true_generation, f)


if __name__ == '__main__':
    main()
