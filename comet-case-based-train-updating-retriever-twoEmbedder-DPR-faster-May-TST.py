import argparse, logging, os, sys, random, datetime, math, time, shutil, copy
os.environ["CUDA_VISIBLE_DEVICES"]="6,7"
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
sys.path.insert(0, "..")
from transformers import (CONFIG_NAME, WEIGHTS_NAME, AdamW,
                            get_linear_schedule_with_warmup)
# from transformers import (BertLMHeadModel, BertTokenizer, BertConfig)
from transformers import (GPT2LMHeadModel, GPT2Tokenizer, GPT2Config)
from transformers import (BartForConditionalGeneration, BartTokenizer, BartConfig)
# from transformers import (T5ForConditionalGeneration, T5Tokenizer, T5Config)
from transformers import (DPRQuestionEncoder, DPRContextEncoder, DPRQuestionEncoderTokenizer, DPRContextEncoderTokenizer)
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from tqdm import tqdm
import pickle
from torch.utils.tensorboard import SummaryWriter
from utils_TST import (load_conceptnet_pure, load_atomic_pure, load_shakespear, load_e2e,
                    save_model, add_special_tokens, tokenize_and_encode, set_seed,
                    preprocess_datasets_for_generator_and_retriever_and_retriever_doc_ProperEOS,
                    concat_cur_bundle_and_encoded_cases_EOSfixed_Bart,
                    concat_cur_bundle_and_encoded_cases_EOSfixed_Bart_randomly_mask_demonstrations,
                    get_path_cur_next_bundle, find_path_tensor_dataset,
                    wait_get_remove_cases_for_bundle_while_deleting_bad_cases_file, shift_tokens_right)
# from utils_TST import (concat_cur_bundle_and_encoded_cases_EOSfixed_Bart_COMETNoNeedRetriever, concat_cur_bundle_and_encoded_cases_EOSfixed_COMETNoNeedRetriever,
                    # concat_cur_bundle_and_encoded_cases_EOSfixed_randomly_mask_demonstrations, concat_cur_bundle_and_encoded_cases_EOSfixed)
from utils_baseline import load_sentiment_data

logging.basicConfig(format = "%(asctime)s - %(levelname)s - %(name)s - %(message)s",
                    datefmt = "%m/%d/%Y %H:%M:%S",
                    level = logging.INFO)
logger = logging.getLogger(__name__)

device1 = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device2 = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

#   case_aug_cur_bundle: [case_aug_gene_input_id, case_aug_gene_attention_mask, case_aug_gene_lm_labels, \
# doc_retr_cases_input_ids, doc_retr_cases_attention_mask, doc_retr_cases_segment_ids, \
# input_retr_input_ids, input_retr_attention_mask, input_retr_segment_ids]
#           case_aug_gene_input_id: [batch_size, n_doc, cases_per_doc * input_len_gene + 1 + input_len_gene]
#           case_aug_gene_attention_mask: [batch_size, n_doc, cases_per_doc * input_len_gene + 1 + input_len_gene]
#           case_aug_gene_lm_labels: [batch_size, n_doc, cases_per_doc * input_len_gene + 1 + input_len_gene]
#           doc_retr_cases_input_ids: [batch_size, n_doc, cases_per_doc, input_len_retr] (not changed)
#           input_retr_input_ids: [batch_size, input_len_retr]
def batch_step(args, model_generator, model_retriever, model_retriever_doc, batch, tokenizer_gene, epsilon):
    ## prepare input data
    # batch_gene = tuple(t.to(device1) for t in batch[0:3])
    # batch_retr_cases = tuple(t.to(device2) for t in batch[3:6])
    batch_gene = batch[0:3]
    batch_retr_cases = batch[3:6]
    batch_retr_cur_input = tuple(t.to(device2) for t in batch[6:9])
    # batch_retr_cur_input = batch[6:9]

    # [batch_size, n_doc, cases_per_doc, input_len_retr]
    ## batch_retr_cases
    doc_retr_cases_input_ids, doc_retr_cases_attention_mask, doc_retr_cases_segment_ids = batch_retr_cases
    cases_per_doc, input_len_retr = doc_retr_cases_input_ids.size()[2], doc_retr_cases_input_ids.size()[3]
    # view retr_cases to fit the requirement of bert's input
    resh_doc_retr_cases_input_ids = doc_retr_cases_input_ids.view(-1, input_len_retr).to(device2)
    resh_doc_retr_cases_attention_mask = doc_retr_cases_attention_mask.view(-1, input_len_retr).to(device2)
    resh_doc_retr_cases_segment_ids = doc_retr_cases_segment_ids.view(-1, input_len_retr).to(device2)
    # [batch_size, input_len_retr]
    input_retr_input_ids, input_retr_attention_mask, input_retr_segment_ids = batch_retr_cur_input
    # [batch_size, n_doc, cases_per_doc * input_len_gene + 1 + input_len_gene]
    ## batch_gene
    case_aug_gene_input_id, case_aug_gene_attention_mask, case_aug_gene_lm_labels = batch_gene
    batch_size, n_doc, tgt_len_gene = case_aug_gene_input_id.size()[0], case_aug_gene_input_id.size()[1], case_aug_gene_input_id.size()[-1]
    # view case_aug_gene to fit the requirement of gpt2's input
    resh_case_aug_gene_input_id = case_aug_gene_input_id.view(-1, tgt_len_gene).to(device1)
    resh_case_aug_gene_attention_mask = case_aug_gene_attention_mask.view(-1, tgt_len_gene).to(device1)
    resh_case_aug_gene_lm_labels = case_aug_gene_lm_labels.view(-1, tgt_len_gene).to(device1)
    ### model_retriever
    ## batch_retr_cases
    outputs_retr_cases = model_retriever_doc(resh_doc_retr_cases_input_ids, attention_mask=resh_doc_retr_cases_attention_mask, token_type_ids=resh_doc_retr_cases_segment_ids)
    # pooled_embedding_retr_cases: [48, 768], verified
    pooled_embedding_retr_cases = outputs_retr_cases[0]
    # print('pooled_embedding_retr_cases.size(): ', pooled_embedding_retr_cases.size())
    # pooled_embedding_retr_cases: [batch_size, n_doc, cases_per_doc, 768]
    pooled_embedding_retr_cases = pooled_embedding_retr_cases.view(batch_size, n_doc, cases_per_doc, -1)
    # print('pooled_embedding_retr_cases.size(): ', pooled_embedding_retr_cases.size())

    # print('pooled_embedding_retr_cases.size(): ', pooled_embedding_retr_cases.size())
    ## batch_retr_cur_input
    outputs_cur_batch = model_retriever(input_retr_input_ids, attention_mask=input_retr_attention_mask, token_type_ids=input_retr_segment_ids)
    # pooled_embedding_cur_batch: [batch_size, 768]
    pooled_embedding_cur_batch = outputs_cur_batch[0]
    # pooled_embedding_cur_batch: [4, 768], verified
    # print('pooled_embedding_cur_batch.size(): ', pooled_embedding_cur_batch.size())

    # pooled_embedding_cur_batch: [batch_size, 1, 1, 768]
    pooled_embedding_cur_batch = pooled_embedding_cur_batch.unsqueeze(1).unsqueeze(2)
    # print('pooled_embedding_cur_batch.size()', pooled_embedding_cur_batch.size())
    ## Get similarity score: [batch_size, n_doc, cases_per_doc, 768]
    simi_score = pooled_embedding_retr_cases * pooled_embedding_cur_batch
    # simi_score: [batch_size, n_doc, cases_per_doc]
    simi_score = torch.sum(simi_score, dim=3)
    # Q:
    torch.save(simi_score, os.path.join(args.output_dir, 'simi_score.pt'))
    batch_size, n_doc, cases_per_doc = simi_score.size()[0], simi_score.size()[1], simi_score.size()[2]
    ## Q: one method to calculate simi_prob; possible to be one reason for nan
    simi_score = simi_score.view(batch_size, -1)
    # SP: newly added, to fix the unbalanced distribution of simi_prob
    simi_score = simi_score / 10
    if args.rand_simi_score:
        ori_seq_list = list(range(len(simi_score)))
        new_seq_list = random.sample(ori_seq_list, len(simi_score))
        new_seq_tensor = torch.tensor(new_seq_list).to(torch.long)
        simi_score = simi_score[new_seq_tensor]
    simi_prob = F.softmax(simi_score, dim=1)
    simi_prob = simi_prob.view(batch_size, n_doc, cases_per_doc)
    # simi_prob: [batch_size, n_doc]
    simi_prob = torch.sum(simi_prob, dim=2)
    # # Another method to calculate simi_prob
    # simi_score = simi_score.sum(dim=2)
    # simi_prob = F.softmax(simi_score, dim=1)

    if not F.relu(simi_prob.sum() - simi_prob.size()[0]) < 0.01:
        print('simi_prob:', simi_prob)
        print('Warning: F.relu(simi_prob.sum() - simi_prob.size()[0]) < 0.01!!!!!!!!!!!!!!!!!')
    # doc_logprobs: [batch_size, n_doc, 1]
    doc_logprobs = torch.log(simi_prob).unsqueeze(-1)

    ###############
    # another simpler option; can be used for debugging
    ### model_generator
    # if "bart" in args.generator_model_type:
    #     results = model_generator(resh_case_aug_gene_input_id, attention_mask=resh_case_aug_gene_attention_mask, labels=resh_case_aug_gene_lm_labels)
    # elif "bert" in args.generator_model_type or "gpt" in args.generator_model_type:
    #     results = model_generator(resh_case_aug_gene_input_id, attention_mask=resh_case_aug_gene_attention_mask, labels=resh_case_aug_gene_lm_labels)
    # nll_loss = results[0]
    # loss = nll_loss
    # seq_logits = results[1].to(device2)
    # seq_logits = seq_logits.contiguous()
    # seq_logprobs = F.log_softmax(seq_logits, dim=-1).view(seq_logits.shape[0] // args.n_doc, args.n_doc, -1, seq_logits.shape[-1])
    ###############

    if "t5" in args.generator_model_type:
        decoder_input_ids = model_generator._shift_right(resh_case_aug_gene_lm_labels)
        results = model_generator(resh_case_aug_gene_input_id, attention_mask=resh_case_aug_gene_attention_mask, decoder_input_ids=decoder_input_ids)
    elif "bart" in args.generator_model_type:
        decoder_input_ids = shift_tokens_right(resh_case_aug_gene_lm_labels, model_generator.config.pad_token_id, model_generator.config.decoder_start_token_id)
        results = model_generator(resh_case_aug_gene_input_id, attention_mask=resh_case_aug_gene_attention_mask, decoder_input_ids=decoder_input_ids)
    else:
        results = model_generator(resh_case_aug_gene_input_id, attention_mask=resh_case_aug_gene_attention_mask)
    # logits/seq_logits: [batch_size * n_doc, tgt_length, #vocab]
    logits = results[0]
    seq_logits = logits.to(device2)

    if args.generator_model_type == "gpt2-lmhead" or "bert" in args.generator_model_type:
        # !!! use these three lines to fix the loss calculating bug of GPT2LMHeadModel
        seq_logits = seq_logits[..., :-1, :].contiguous()
        shift_labels = resh_case_aug_gene_lm_labels[..., 1:].contiguous()
    elif "bart" in args.generator_model_type or "t5" in args.generator_model_type:
        seq_logits = seq_logits.contiguous()
        shift_labels = resh_case_aug_gene_lm_labels.contiguous()
    else:
        raise NotImplementError

    case_aug_gene_lm_labels = shift_labels.view(shift_labels.shape[0] // args.n_doc, args.n_doc, -1)

    # seq_logprobs: [batch_size, n_doc, tgt_length, #vocab]
    seq_logprobs = F.log_softmax(seq_logits, dim=-1).view(seq_logits.shape[0] // args.n_doc, args.n_doc, -1, seq_logits.shape[-1])
    # maybe this line of code is not necessary, but it should be ok to keep it
    seq_logprobs_backup = copy.deepcopy(seq_logprobs)

    ## calculate loss
    # case_aug_gene_lm_labels: [batch_size, n_doc, tgt_length, 1]
    case_aug_gene_lm_labels = case_aug_gene_lm_labels.unsqueeze(-1).to(device2)
    # change -100 in case_aug_gene_lm_labels to tokenizer_gene.encode(tokenizer_gene.pad_token)[0]
    case_aug_gene_lm_labels_if_pad = case_aug_gene_lm_labels.eq(-100)
    # num_not_pad_in_labels: [batch_size, n_doc, 1]
    num_not_pad_in_labels = case_aug_gene_lm_labels_if_pad.logical_not().to(torch.float).sum(2)
    case_aug_gene_lm_labels.masked_fill_(case_aug_gene_lm_labels_if_pad, tokenizer_gene.encode(tokenizer_gene.pad_token)[0])
    # ll: [batch_size, n_doc, tgt_length, 1], some of the logprob are from [PAD] token
    ll = seq_logprobs.gather(dim=-1, index=case_aug_gene_lm_labels)
    # total sum of all (normalised) logits
    smooth_obj = seq_logprobs.sum(dim=-1, keepdim=True)

    # do not count the logprob of token whose id equals pad id
    # ll: [batch_size, n_doc, tgt_length, 1]
    ll.masked_fill_(case_aug_gene_lm_labels_if_pad, 0)
    smooth_obj.masked_fill_(case_aug_gene_lm_labels_if_pad, 0)

    # p_doc come to play; Marginalize
    # ll: [batch_size, n_doc, tgt_length, 1]; ll2: [batch_size, n_doc, 1]
    # ll2: 1/n * \sum_{log(Pw_i)}
    ll2 = ll.sum(2) / num_not_pad_in_labels
    assert ll2.size() == num_not_pad_in_labels.size()
    # doc_logprobs: [batch_size, n_doc, 1]
    ll2 = ll2 + doc_logprobs

    smooth_obj = smooth_obj.sum(2)
    smooth_obj = smooth_obj + doc_logprobs
    # Marginalize over docs
    # ll3: [batch_size, 1]
    ll3 = ll2.logsumexp(1)
    smooth_obj = smooth_obj.logsumexp(1)
    nll_loss = -ll3
    smooth_loss = -smooth_obj
    bool_if_positive = (nll_loss > -0.05)

    # Q:
    torch.save(simi_prob.cpu(), os.path.join(args.output_dir, 'simi_prob.pt'))
    if bool_if_positive.to(torch.float).mean() != 1:
        raise Exception
    # print('nll_loss:', nll_loss)
    # print('nll_loss.size():', nll_loss.size())
    # Found Problem: since some token is [PAD], mean() will do average over these [PAD] tokens, but we should overlook [PAD] tokens
    nll_loss = nll_loss.mean()
    smooth_loss = smooth_loss.mean()

    eps_i = epsilon / seq_logprobs.size(-1)
    loss = (1.0 - epsilon) * nll_loss + eps_i * smooth_loss
    # seq_logprobs_backup: [batch_size, n_doc, seq_length, len_words]
    # calculate accuracy when the dataset is "sentiment sentence classification" dataset
    if args.dataset_selection == 4 or args.dataset_selection == 5 or args.dataset_selection == 6:
        # print("seq_logprobs_backup.size(): ", seq_logprobs_backup.size())
        cur_batch_size = seq_logprobs_backup.size()[0]
        ## true label
        true_label = []
        # case_aug_gene_lm_labels: [batch_size, n_doc, cases_per_doc * input_len_gene + 1 + input_len_gene]
        ## pred label
        cnt_correct = 0
        for cur_data_id in range(cur_batch_size):
            # assume only one doc is considered, else is not implemented yet
            if not case_aug_gene_lm_labels.size()[1] == 1:
                print("Current code is only designed for the situation where only n_doc is 1.")
            # print("case_aug_gene_lm_labels[cur_data_id, 0, :10]: ", case_aug_gene_lm_labels[cur_data_id, 0, :10])
            if "bart" in args.generator_model_type:
                # cur_true_label = tokenizer_gene.decode(case_aug_gene_lm_labels[cur_data_id, 0, 0]).strip()
                # since we have added a <bos> token
                cur_true_label = tokenizer_gene.decode(case_aug_gene_lm_labels[cur_data_id, 0, 1]).strip()
            else:
                raise NotImplementedError
            if "bart" in args.generator_model_type:
                prob_positive_pred = torch.exp(seq_logprobs_backup[cur_data_id, 0, 1, 22173])
                prob_negative_pred = torch.exp(seq_logprobs_backup[cur_data_id, 0, 1, 33407])
                prob_neutral_pred = torch.exp(seq_logprobs_backup[cur_data_id, 0, 1, 12516])
                prob_0_pred = torch.exp(seq_logprobs_backup[cur_data_id, 0, 1, 288])
                prob_1_pred = torch.exp(seq_logprobs_backup[cur_data_id, 0, 1, 134])
                prob_2_pred = torch.exp(seq_logprobs_backup[cur_data_id, 0, 1, 176])
                prob_3_pred = torch.exp(seq_logprobs_backup[cur_data_id, 0, 1, 246])
                prob_4_pred = torch.exp(seq_logprobs_backup[cur_data_id, 0, 1, 306])
            else:
                raise NotImplementedError
            if args.dataset_selection == 4:
                if not (cur_true_label == 'positive' or cur_true_label == 'negative'):
                    raise Exception("Not acceptable cur_true_label: {}, it should either be positive or negative.".format(cur_true_label))
                if (prob_positive_pred >= prob_negative_pred and cur_true_label == 'positive') or (prob_positive_pred < prob_negative_pred and cur_true_label == 'negative'):
                    cnt_correct += 1
            elif args.dataset_selection == 5:
                if not (cur_true_label == 'positive' or cur_true_label == 'negative' or cur_true_label == 'neutral'):
                    raise Exception("Not acceptable cur_true_label: {}, it should either be positive or negative or neutral.".format(cur_true_label))
                if (prob_positive_pred >= prob_negative_pred and prob_positive_pred >= prob_neutral_pred and cur_true_label == 'positive') or (prob_negative_pred > prob_positive_pred and prob_negative_pred > prob_neutral_pred and cur_true_label == 'negative') or (prob_neutral_pred > prob_positive_pred and prob_neutral_pred > prob_negative_pred and cur_true_label == 'neutral'):
                    cnt_correct += 1
            elif args.dataset_selection == 6:
                if not (cur_true_label == '0' or cur_true_label == '1' or cur_true_label == '2' or cur_true_label == '3' or cur_true_label == '4'):
                    raise Exception("Not acceptable cur_true_label: {}, it should be between 0~4.".format(cur_true_label))
                if int(cur_true_label) == np.argmax([prob_0_pred.item(), prob_1_pred.item(), prob_2_pred.item(), prob_3_pred.item(), prob_4_pred.item()]):
                    cnt_correct += 1
        batch_accuracy = cnt_correct / cur_batch_size
    else:
        batch_accuracy = None
    return loss, nll_loss, seq_logprobs, doc_logprobs, batch_accuracy


#   case_aug_cur_bundle: [case_aug_gene_input_id, case_aug_gene_attention_mask, case_aug_gene_lm_labels, \
# doc_retr_cases_input_ids, doc_retr_cases_attention_mask, doc_retr_cases_segment_ids, \
# input_retr_input_ids, input_retr_attention_mask, input_retr_segment_ids]
#           case_aug_gene_input_id: [batch_size, n_doc, cases_per_doc * input_len_gene + 1 + input_len_gene]
#           case_aug_gene_attention_mask: [batch_size, n_doc, cases_per_doc * input_len_gene + 1 + input_len_gene]
#           case_aug_gene_lm_labels: [batch_size, n_doc, cases_per_doc * input_len_gene + 1 + input_len_gene]
#           doc_retr_cases_input_ids: [batch_size, n_doc, cases_per_doc, input_len_retr] (not changed)
#           input_retr_input_ids: [batch_size, input_len_retr]
def batch_step_fast_train_1GPU(args, model_generator, model_retriever, model_retriever_doc, batch, tokenizer_gene, epsilon, if_try_one_gpu=False):
    ## prepare input data
    batch_gene = batch[0:3]
    # [batch_size, n_doc, cases_per_doc * input_len_gene + 1 + input_len_gene]
    case_aug_gene_input_id, case_aug_gene_attention_mask, case_aug_gene_lm_labels = batch_gene
    batch_size, n_doc, tgt_len_gene = case_aug_gene_input_id.size()[0], case_aug_gene_input_id.size()[1], case_aug_gene_input_id.size()[-1]

    # view case_aug_gene to fit the requirement of gpt2's input
    resh_case_aug_gene_input_id = case_aug_gene_input_id.view(-1, tgt_len_gene).to(device1)
    resh_case_aug_gene_attention_mask = case_aug_gene_attention_mask.view(-1, tgt_len_gene).to(device1)
    resh_case_aug_gene_lm_labels = case_aug_gene_lm_labels.view(-1, tgt_len_gene).to(device1)

    if "t5" in args.generator_model_type:
        decoder_input_ids = model_generator._shift_right(resh_case_aug_gene_lm_labels)
        results = model_generator(resh_case_aug_gene_input_id, attention_mask=resh_case_aug_gene_attention_mask, decoder_input_ids=decoder_input_ids, labels=resh_case_aug_gene_lm_labels)
    elif "bart" in args.generator_model_type:
        decoder_input_ids = shift_tokens_right(resh_case_aug_gene_lm_labels, model_generator.config.pad_token_id, model_generator.config.decoder_start_token_id)
        results = model_generator(resh_case_aug_gene_input_id, attention_mask=resh_case_aug_gene_attention_mask, decoder_input_ids=decoder_input_ids, labels=resh_case_aug_gene_lm_labels)
    else:
        results = model_generator(resh_case_aug_gene_input_id, attention_mask=resh_case_aug_gene_attention_mask, labels=resh_case_aug_gene_lm_labels)
    # logits/seq_logits: [batch_size * n_doc, tgt_length, #vocab]
    nll_loss, logits = results[0], results[1]
    if if_try_one_gpu:
        nll_loss_device2 = nll_loss
        seq_logits = logits
    else:
        nll_loss_device2 = nll_loss.to(device2)
        # seq_logprobs: [batch_size, n_doc, tgt_length, #vocab]
        seq_logits = logits.to(device2)

    if args.generator_model_type == "gpt2-lmhead" or "bert" in args.generator_model_type:
        # !!! use these three lines to fix the loss calculating bug of GPT2LMHeadModel
        seq_logits = seq_logits[..., :-1, :].contiguous()
        shift_labels = resh_case_aug_gene_lm_labels[..., 1:].contiguous()
    elif "bart" in args.generator_model_type or "t5" in args.generator_model_type:
        seq_logits = seq_logits.contiguous()
        shift_labels = resh_case_aug_gene_lm_labels.contiguous()
    else:
        raise NotImplementError

    ## get case_aug_gene_lm_labels_if_pad
    case_aug_gene_lm_labels = shift_labels.view(shift_labels.shape[0] // args.n_doc, args.n_doc, -1)
    if if_try_one_gpu:
        case_aug_gene_lm_labels = case_aug_gene_lm_labels.unsqueeze(-1)
    else:
        # case_aug_gene_lm_labels: [batch_size, n_doc, tgt_length, 1]
        case_aug_gene_lm_labels = case_aug_gene_lm_labels.unsqueeze(-1).to(device2)
    # change -100 in case_aug_gene_lm_labels to tokenizer_gene.encode(tokenizer_gene.pad_token)[0]
    case_aug_gene_lm_labels_if_pad = case_aug_gene_lm_labels.eq(-100)

    ## get smooth_loss
    seq_logprobs = F.log_softmax(seq_logits, dim=-1).view(seq_logits.shape[0] // args.n_doc, args.n_doc, -1, seq_logits.shape[-1])
    smooth_obj = seq_logprobs.sum(dim=-1, keepdim=True)
    smooth_obj.masked_fill_(case_aug_gene_lm_labels_if_pad, 0)
    smooth_obj = smooth_obj.sum(2)
    # smooth_obj = smooth_obj + doc_logprobs
    smooth_obj = smooth_obj.logsumexp(1)
    smooth_loss = -smooth_obj
    smooth_loss = smooth_loss.mean()

    ## get total loss
    eps_i = epsilon / seq_logprobs.size(-1)
    loss = (1.0 - epsilon) * nll_loss_device2 + eps_i * smooth_loss
    # seq_logprobs: [batch_size, n_doc, seq_length, len_words]
    # calculate accuracy when the dataset is "sentiment sentence classification" dataset
    if args.dataset_selection == 4 or args.dataset_selection == 5 or args.dataset_selection == 6:
        # print("seq_logprobs.size(): ", seq_logprobs.size())
        cur_batch_size = seq_logprobs.size()[0]
        ## true label
        true_label = []
        # case_aug_gene_lm_labels: [batch_size, n_doc, cases_per_doc * input_len_gene + 1 + input_len_gene]
        ## pred label
        cnt_correct = 0
        for cur_data_id in range(cur_batch_size):
            # assume only one doc is considered, else is not implemented yet
            if not case_aug_gene_lm_labels.size()[1] == 1:
                print("Current code is only designed for the situation where only n_doc is 1.")
            # print("case_aug_gene_lm_labels[cur_data_id, 0, :10]: ", case_aug_gene_lm_labels[cur_data_id, 0, :10])
            if "bart" in args.generator_model_type:
                # cur_true_label = tokenizer_gene.decode(case_aug_gene_lm_labels[cur_data_id, 0, 0]).strip()
                # since we have added a <bos> token
                cur_true_label = tokenizer_gene.decode(case_aug_gene_lm_labels[cur_data_id, 0, 1]).strip()
            else:
                raise NotImplementedError
            if "bart" in args.generator_model_type:
                prob_positive_pred = torch.exp(seq_logprobs[cur_data_id, 0, 1, 22173])
                prob_negative_pred = torch.exp(seq_logprobs[cur_data_id, 0, 1, 33407])
                prob_neutral_pred = torch.exp(seq_logprobs[cur_data_id, 0, 1, 12516])
                prob_0_pred = torch.exp(seq_logprobs[cur_data_id, 0, 1, 288])
                prob_1_pred = torch.exp(seq_logprobs[cur_data_id, 0, 1, 134])
                prob_2_pred = torch.exp(seq_logprobs[cur_data_id, 0, 1, 176])
                prob_3_pred = torch.exp(seq_logprobs[cur_data_id, 0, 1, 246])
                prob_4_pred = torch.exp(seq_logprobs[cur_data_id, 0, 1, 306])
            else:
                raise NotImplementedError
            if args.dataset_selection == 4:
                if not (cur_true_label == 'positive' or cur_true_label == 'negative'):
                    raise Exception("Not acceptable cur_true_label: {}, it should either be positive or negative.".format(cur_true_label))
                if (prob_positive_pred >= prob_negative_pred and cur_true_label == 'positive') or (prob_positive_pred < prob_negative_pred and cur_true_label == 'negative'):
                    cnt_correct += 1
            elif args.dataset_selection == 5:
                if not (cur_true_label == 'positive' or cur_true_label == 'negative' or cur_true_label == 'neutral'):
                    raise Exception("Not acceptable cur_true_label: {}, it should either be positive or negative or neutral.".format(cur_true_label))
                if (prob_positive_pred >= prob_negative_pred and prob_positive_pred >= prob_neutral_pred and cur_true_label == 'positive') or (prob_negative_pred > prob_positive_pred and prob_negative_pred > prob_neutral_pred and cur_true_label == 'negative') or (prob_neutral_pred > prob_positive_pred and prob_neutral_pred > prob_negative_pred and cur_true_label == 'neutral'):
                    cnt_correct += 1
            elif args.dataset_selection == 6:
                if not (cur_true_label == '0' or cur_true_label == '1' or cur_true_label == '2' or cur_true_label == '3' or cur_true_label == '4'):
                    raise Exception("Not acceptable cur_true_label: {}, it should be between 0~4.".format(cur_true_label))
                if int(cur_true_label) == np.argmax([prob_0_pred.item(), prob_1_pred.item(), prob_2_pred.item(), prob_3_pred.item(), prob_4_pred.item()]):
                    cnt_correct += 1
        batch_accuracy = cnt_correct / cur_batch_size
    else:
        batch_accuracy = None
    return loss, nll_loss_device2, seq_logprobs, batch_accuracy



#   batch: [case_aug_gene_input_id, case_aug_gene_attention_mask, case_aug_gene_lm_labels]
#           case_aug_gene_input_id: [batch_size, cases_per_doc * input_len_gene + 1 + input_len_gene + 1]
#           case_aug_gene_attention_mask: [batch_size, cases_per_doc * input_len_gene + 1 + input_len_gene + 1]
#           case_aug_gene_lm_labels: [batch_size, cases_per_doc * input_len_gene + 1 + input_len_gene + 1]
def batch_step_eval_analysis(args, model_generator, model_retriever, model_retriever_doc, batch, tokenizer_gene):
    ## prepare input data
    batch_gene = tuple(t.to(device1) for t in batch[0:3])
    batch_retr_cases = batch[3:6]
    batch_retr_cur_input = tuple(t.to(device2) for t in batch[6:9])
    ### Generator ###
    # Q: originally here use [batch_size, n_doc, cases_per_doc * input_len_gene + 1 + input_len_gene + 1]
    # [batch_size, n_doc, cases_per_doc * input_doc_len_gene + input_cur_len_gene]
    case_aug_gene_input_id, case_aug_gene_attention_mask, case_aug_gene_lm_labels = batch_gene
    batch_size, n_doc, tgt_len_gene = case_aug_gene_input_id.size()[0], case_aug_gene_input_id.size()[1], case_aug_gene_input_id.size()[-1]
    # print('n_doc:', n_doc)
    assert n_doc == 1
    # view case_aug_gene to fit the requirement of gpt2's input
    resh_case_aug_gene_input_id = case_aug_gene_input_id.view(-1, tgt_len_gene)
    resh_case_aug_gene_attention_mask = case_aug_gene_attention_mask.view(-1, tgt_len_gene)
    resh_case_aug_gene_lm_labels = case_aug_gene_lm_labels.view(-1, tgt_len_gene)
    ## model_generator
    if "gpt2" in args.generator_model_type or "bert" in args.generator_model_type:
        results = model_generator(resh_case_aug_gene_input_id, attention_mask=resh_case_aug_gene_attention_mask, labels=resh_case_aug_gene_lm_labels)
    elif "bart" in args.generator_model_type or "t5" in args.generator_model_type:
        results = model_generator(resh_case_aug_gene_input_id, attention_mask=resh_case_aug_gene_attention_mask, labels=resh_case_aug_gene_lm_labels)
    else:
        raise NotImplementError
    # logits/seq_logits: [batch_size * n_doc, tgt_length, #vocab]
    loss, logits = results[0], results[1]

    ### Retriever_doc ###
    # [batch_size, n_doc, cases_per_doc, input_len_retr]
    ## batch_retr_cases
    doc_retr_cases_input_ids, doc_retr_cases_attention_mask, doc_retr_cases_segment_ids = batch_retr_cases
    cases_per_doc, input_len_retr = doc_retr_cases_input_ids.size()[2], doc_retr_cases_input_ids.size()[3]
    # view retr_cases to fit the requirement of bert's input
    resh_doc_retr_cases_input_ids = doc_retr_cases_input_ids.view(-1, input_len_retr).to(device2)
    resh_doc_retr_cases_attention_mask = doc_retr_cases_attention_mask.view(-1, input_len_retr).to(device2)
    resh_doc_retr_cases_segment_ids = doc_retr_cases_segment_ids.view(-1, input_len_retr).to(device2)
    ## batch_retr_cases
    outputs_retr_cases = model_retriever_doc(resh_doc_retr_cases_input_ids, attention_mask=resh_doc_retr_cases_attention_mask, token_type_ids=resh_doc_retr_cases_segment_ids)
    # pooled_embedding_retr_cases: [48, 768], verified
    pooled_embedding_retr_cases = outputs_retr_cases[0]
    # print('pooled_embedding_retr_cases.size(): ', pooled_embedding_retr_cases.size())
    # pooled_embedding_retr_cases: [batch_size, n_doc, cases_per_doc, 768]
    pooled_embedding_retr_cases = pooled_embedding_retr_cases.view(batch_size, n_doc, cases_per_doc, -1)
    # print('pooled_embedding_retr_cases.size(): ', pooled_embedding_retr_cases.size())

    ### Retriever ###
    # [batch_size, input_len_retr]
    input_retr_input_ids, input_retr_attention_mask, input_retr_segment_ids = batch_retr_cur_input
    ## batch_retr_cur_input
    outputs_cur_batch = model_retriever(input_retr_input_ids, attention_mask=input_retr_attention_mask, token_type_ids=input_retr_segment_ids)
    # pooled_embedding_cur_batch: [batch_size, 768]
    pooled_embedding_cur_batch = outputs_cur_batch[0]
    # pooled_embedding_cur_batch: [4, 768], verified
    # print('pooled_embedding_cur_batch.size(): ', pooled_embedding_cur_batch.size())
    # pooled_embedding_cur_batch: [batch_size, 1, 1, 768]
    pooled_embedding_cur_batch = pooled_embedding_cur_batch.unsqueeze(1).unsqueeze(2)
    # print('pooled_embedding_cur_batch.size(): ', pooled_embedding_cur_batch.size())

    ### (start) This block of code is only used for sentiment sentence classification dataset
    seq_logits = logits
    if args.generator_model_type == "gpt2-lmhead" or "bert" in args.generator_model_type:
        # !!! use these three lines to fix the loss calculating bug of GPT2LMHeadModel
        seq_logits = seq_logits[..., :-1, :].contiguous()
        shift_labels = resh_case_aug_gene_lm_labels[..., 1:].contiguous()
    elif "bart" in args.generator_model_type or "t5" in args.generator_model_type:
        seq_logits = seq_logits.contiguous()
        shift_labels = resh_case_aug_gene_lm_labels.contiguous()
    seq_logprobs = F.log_softmax(seq_logits, dim=-1).view(seq_logits.shape[0] // args.n_doc, args.n_doc, -1, seq_logits.shape[-1])
    # seq_logprobs: [batch_size, n_doc, seq_length, len_words]
    # calculate accuracy when the dataset is "sentiment sentence classification" dataset
    if args.dataset_selection == 4 or args.dataset_selection == 5 or args.dataset_selection == 6:
        # print("seq_logprobs.size(): ", seq_logprobs.size())
        cur_batch_size = seq_logprobs.size()[0]
        ## true label
        true_label = []
        # case_aug_gene_lm_labels: [batch_size, n_doc, cases_per_doc * input_len_gene + 1 + input_len_gene]
        ## pred label
        cnt_correct = 0
        for cur_data_id in range(cur_batch_size):
            # assume only one doc is considered, else is not implemented yet
            if not case_aug_gene_lm_labels.size()[1] == 1:
                print("Current code is only designed for the situation where only n_doc is 1.")
            # print("case_aug_gene_lm_labels[cur_data_id, 0, :10]: ", case_aug_gene_lm_labels[cur_data_id, 0, :10])
            if "bart" in args.generator_model_type:
                # cur_true_label = tokenizer_gene.decode(case_aug_gene_lm_labels[cur_data_id, 0, 0]).strip()
                # since we have added a <bos> token
                cur_true_label = tokenizer_gene.decode(case_aug_gene_lm_labels[cur_data_id, 0, 1]).strip()
            else:
                raise NotImplementedError
            if "bart" in args.generator_model_type:
                prob_positive_pred = torch.exp(seq_logprobs[cur_data_id, 0, 1, 22173])
                prob_negative_pred = torch.exp(seq_logprobs[cur_data_id, 0, 1, 33407])
                prob_neutral_pred = torch.exp(seq_logprobs[cur_data_id, 0, 1, 12516])
                prob_0_pred = torch.exp(seq_logprobs[cur_data_id, 0, 1, 288])
                prob_1_pred = torch.exp(seq_logprobs[cur_data_id, 0, 1, 134])
                prob_2_pred = torch.exp(seq_logprobs[cur_data_id, 0, 1, 176])
                prob_3_pred = torch.exp(seq_logprobs[cur_data_id, 0, 1, 246])
                prob_4_pred = torch.exp(seq_logprobs[cur_data_id, 0, 1, 306])
            else:
                raise NotImplementedError
            if args.dataset_selection == 4:
                if not (cur_true_label == 'positive' or cur_true_label == 'negative'):
                    raise Exception("Not acceptable cur_true_label: {}, it should either be positive or negative.".format(cur_true_label))
                if (prob_positive_pred >= prob_negative_pred and cur_true_label == 'positive') or (prob_positive_pred < prob_negative_pred and cur_true_label == 'negative'):
                    cnt_correct += 1
            elif args.dataset_selection == 5:
                if not (cur_true_label == 'positive' or cur_true_label == 'negative' or cur_true_label == 'neutral'):
                    raise Exception("Not acceptable cur_true_label: {}, it should either be positive or negative or neutral.".format(cur_true_label))
                if (prob_positive_pred >= prob_negative_pred and prob_positive_pred >= prob_neutral_pred and cur_true_label == 'positive') or (prob_negative_pred > prob_positive_pred and prob_negative_pred > prob_neutral_pred and cur_true_label == 'negative') or (prob_neutral_pred > prob_positive_pred and prob_neutral_pred > prob_negative_pred and cur_true_label == 'neutral'):
                    cnt_correct += 1
            elif args.dataset_selection == 6:
                if not (cur_true_label == '0' or cur_true_label == '1' or cur_true_label == '2' or cur_true_label == '3' or cur_true_label == '4'):
                    raise Exception("Not acceptable cur_true_label: {}, it should be between 0~4.".format(cur_true_label))
                if int(cur_true_label) == np.argmax([prob_0_pred.item(), prob_1_pred.item(), prob_2_pred.item(), prob_3_pred.item(), prob_4_pred.item()]):
                    cnt_correct += 1
        batch_accuracy = cnt_correct / cur_batch_size
    else:
        batch_accuracy = None
    ### (end) This block of code is only used for sentiment sentence classification dataset
    return loss, logits, pooled_embedding_retr_cases, pooled_embedding_cur_batch, batch_accuracy

#   batch: [case_aug_gene_input_id, case_aug_gene_attention_mask, case_aug_gene_lm_labels]
#           case_aug_gene_input_id: [batch_size, cases_per_doc * input_len_gene + 1 + input_len_gene + 1]
#           case_aug_gene_attention_mask: [batch_size, cases_per_doc * input_len_gene + 1 + input_len_gene + 1]
#           case_aug_gene_lm_labels: [batch_size, cases_per_doc * input_len_gene + 1 + input_len_gene + 1]
def batch_step_eval(args, model_generator, batch, tokenizer_gene):
    ## prepare input data
    batch_gene = tuple(t.to(device1) for t in batch)
    # Q: originally here use [batch_size, n_doc, cases_per_doc * input_len_gene + 1 + input_len_gene + 1]
    # [batch_size, n_doc, cases_per_doc * input_doc_len_gene + input_cur_len_gene]
    case_aug_gene_input_id, case_aug_gene_attention_mask, case_aug_gene_lm_labels = batch_gene
    batch_size, n_doc, tgt_len_gene = case_aug_gene_input_id.size()[0], case_aug_gene_input_id.size()[1], case_aug_gene_input_id.size()[-1]

    assert n_doc == 1
    # view case_aug_gene to fit the requirement of gpt2's input
    resh_case_aug_gene_input_id = case_aug_gene_input_id.view(-1, tgt_len_gene)
    resh_case_aug_gene_attention_mask = case_aug_gene_attention_mask.view(-1, tgt_len_gene)
    resh_case_aug_gene_lm_labels = case_aug_gene_lm_labels.view(-1, tgt_len_gene)
    ## model_generators
    # original code, which should use resh_case_aug_gene_lm_labels but not case_aug_gene_lm_labels
    # results = model_generator(resh_case_aug_gene_input_id, attention_mask=resh_case_aug_gene_attention_mask, labels=case_aug_gene_lm_labels)
    if "gpt2" in args.generator_model_type or "bert" in args.generator_model_type:
        results = model_generator(resh_case_aug_gene_input_id, attention_mask=resh_case_aug_gene_attention_mask, labels=resh_case_aug_gene_lm_labels)
    elif "bart" in args.generator_model_type or "t5" in args.generator_model_type:
        results = model_generator(resh_case_aug_gene_input_id, attention_mask=resh_case_aug_gene_attention_mask, labels=resh_case_aug_gene_lm_labels)
    else:
        raise NotImplementError
    # logits/seq_logits: [batch_size * n_doc, tgt_length, #vocab]
    loss, logits = results[0], results[1]
    # print('loss: ', loss)
    ### (start) This block of code is only used for sentiment sentence classification dataset
    seq_logits = logits
    if args.generator_model_type == "gpt2-lmhead" or "bert" in args.generator_model_type:
        # !!! use these three lines to fix the loss calculating bug of GPT2LMHeadModel
        seq_logits = seq_logits[..., :-1, :].contiguous()
        shift_labels = resh_case_aug_gene_lm_labels[..., 1:].contiguous()
    elif "bart" in args.generator_model_type or "t5" in args.generator_model_type:
        seq_logits = seq_logits.contiguous()
        shift_labels = resh_case_aug_gene_lm_labels.contiguous()
    else:
        raise NotImplementError
    seq_logprobs = F.log_softmax(seq_logits, dim=-1).view(seq_logits.shape[0] // args.n_doc, args.n_doc, -1, seq_logits.shape[-1])
    # seq_logprobs: [batch_size, n_doc, seq_length, len_words]
    # calculate accuracy when the dataset is "sentiment sentence classification" dataset
    if args.dataset_selection == 4 or args.dataset_selection == 5 or args.dataset_selection == 6:
        # print("seq_logprobs.size(): ", seq_logprobs.size())
        cur_batch_size = seq_logprobs.size()[0]
        ## true label
        true_label = []
        # case_aug_gene_lm_labels: [batch_size, n_doc, cases_per_doc * input_len_gene + 1 + input_len_gene]
        ## pred label
        cnt_correct = 0
        for cur_data_id in range(cur_batch_size):
            # assume only one doc is considered, else is not implemented yet
            if not case_aug_gene_lm_labels.size()[1] == 1:
                print("Current code is only designed for the situation where only n_doc is 1.")
            # print("case_aug_gene_lm_labels[cur_data_id, 0, :10]: ", case_aug_gene_lm_labels[cur_data_id, 0, :10])
            if "bart" in args.generator_model_type:
                # cur_true_label = tokenizer_gene.decode(case_aug_gene_lm_labels[cur_data_id, 0, 0]).strip()
                # since we have added a <bos> token
                cur_true_label = tokenizer_gene.decode(case_aug_gene_lm_labels[cur_data_id, 0, 1]).strip()
            else:
                raise NotImplementedError
            if "bart" in args.generator_model_type:
                prob_positive_pred = torch.exp(seq_logprobs[cur_data_id, 0, 1, 22173])
                prob_negative_pred = torch.exp(seq_logprobs[cur_data_id, 0, 1, 33407])
                prob_neutral_pred = torch.exp(seq_logprobs[cur_data_id, 0, 1, 12516])
                prob_0_pred = torch.exp(seq_logprobs[cur_data_id, 0, 1, 288])
                prob_1_pred = torch.exp(seq_logprobs[cur_data_id, 0, 1, 134])
                prob_2_pred = torch.exp(seq_logprobs[cur_data_id, 0, 1, 176])
                prob_3_pred = torch.exp(seq_logprobs[cur_data_id, 0, 1, 246])
                prob_4_pred = torch.exp(seq_logprobs[cur_data_id, 0, 1, 306])
            else:
                raise NotImplementedError
            if args.dataset_selection == 4:
                if not (cur_true_label == 'positive' or cur_true_label == 'negative'):
                    raise Exception("Not acceptable cur_true_label: {}, it should either be positive or negative.".format(cur_true_label))
                if (prob_positive_pred >= prob_negative_pred and cur_true_label == 'positive') or (prob_positive_pred < prob_negative_pred and cur_true_label == 'negative'):
                    cnt_correct += 1
            elif args.dataset_selection == 5:
                if not (cur_true_label == 'positive' or cur_true_label == 'negative' or cur_true_label == 'neutral'):
                    raise Exception("Not acceptable cur_true_label: {}, it should either be positive or negative or neutral.".format(cur_true_label))
                if (prob_positive_pred >= prob_negative_pred and prob_positive_pred >= prob_neutral_pred and cur_true_label == 'positive') or (prob_negative_pred > prob_positive_pred and prob_negative_pred > prob_neutral_pred and cur_true_label == 'negative') or (prob_neutral_pred > prob_positive_pred and prob_neutral_pred > prob_negative_pred and cur_true_label == 'neutral'):
                    cnt_correct += 1
            elif args.dataset_selection == 6:
                if not (cur_true_label == '0' or cur_true_label == '1' or cur_true_label == '2' or cur_true_label == '3' or cur_true_label == '4'):
                    raise Exception("Not acceptable cur_true_label: {}, it should be between 0~4.".format(cur_true_label))
                if int(cur_true_label) == np.argmax([prob_0_pred.item(), prob_1_pred.item(), prob_2_pred.item(), prob_3_pred.item(), prob_4_pred.item()]):
                    cnt_correct += 1
        batch_accuracy = cnt_correct / cur_batch_size
    else:
        batch_accuracy = None
    ### (end) This block of code is only used for sentiment sentence classification dataset
    return loss, logits, batch_accuracy

# added max_additional_cases
def evaluate(args, model_generator, model_retriever, model_retriever_doc, tokenizer_gene, \
    dataloader_in_bundle_eval_or_test, path_next_bundle_eval_or_test, path_retriever_eval_or_test, \
    path_retriever_doc_eval_or_test, path_retrieved_encoded_cases_eval_or_test, data_type, path_cnt_saved_bundle, path_cnt_retrieved_bundle):
    # send a signal to retriever
    if data_type == 'eval':
        print('INFO: begin evaluating...')
        ## send signal to retriever
        path_signal_file_if_eval_or_test = os.path.join(args.output_dir, 'under_evaluation.pt')
        torch.save(torch.ones(1), path_signal_file_if_eval_or_test)
        if_eval_analysis = False
        id_cnt_bundle = 1
    elif data_type == 'test':
        print('INFO: begin testing...')
        path_signal_file_if_eval_or_test = os.path.join(args.output_dir, 'under_evaluation_test.pt')
        torch.save(torch.ones(1), path_signal_file_if_eval_or_test)
        if_eval_analysis = args.if_eval_analysis
        id_cnt_bundle = 2
    else:
        raise Exception('Wrong data type! data_type: ', data_type)

    # only save retriever once
    if not (args.if_fast_train and args.if_comet_baseline):
        while os.path.exists(path_retriever_eval_or_test) or os.path.exists(path_retriever_doc_eval_or_test):
            print('Warning: path_retriever_eval_or_test or path_retriever_doc_eval_or_test still exists!')
            time.sleep(5)
        torch.save(model_retriever.state_dict(), path_retriever_eval_or_test)
        torch.save(model_retriever_doc.state_dict(), path_retriever_doc_eval_or_test)

    eval_loss = 0
    nb_eval_steps = 0
    num_displays = 1
    if args.dataset_selection == 4 or args.dataset_selection == 5 or args.dataset_selection == 6:
        eval_ttl_cnt_correct = 0
    if if_eval_analysis:
        # Embed_docs, Embed_cur_query = None, None
        Loss_eval = []
    # display_batch_indices
    display_batch_indices = list(range(len(dataloader_in_bundle_eval_or_test)))
    random.shuffle(display_batch_indices)
    display_batch_indices = display_batch_indices[:num_displays]
    # eos_token, path_next_bundle_eval_or_test, num_bundles, itr_eval_bundleloader
    if args.generator_model_type == "gpt2-lmhead" or "t5" in args.generator_model_type:
        eos_token = tokenizer_gene.encode(tokenizer_gene.eos_token)[0]
    elif args.generator_model_type == "bart-base" or args.generator_model_type == "bart-large" or 'bert' in args.generator_model_type:
        eos_token = tokenizer_gene.encode(tokenizer_gene.eos_token)[1]
    else:
        raise NotImplementError

    retrieved_cases_cur_bundle = None
    print("\n\nsome examples")
    num_bundles = len(dataloader_in_bundle_eval_or_test)
    print('num_bundles:', num_bundles)
    itr_eval_bundleloader = iter(dataloader_in_bundle_eval_or_test)
    for id_bundle in range(num_bundles):
        # FQ: add 'or id_bundle == 0', in case num_bundles == 1
        if id_bundle < num_bundles - 1 or id_bundle == 0:
            next_bundle = next(itr_eval_bundleloader)
            # about next_bundle
            if data_type == 'eval':
                path_prev_next_bundle, path_cur_next_bundle, cnt_bundle = get_path_cur_next_bundle(args, \
                    path_cnt_saved_bundle, data_type, path_next_bundle_eval=path_next_bundle_eval_or_test)
            elif data_type == 'test':
                path_prev_next_bundle, path_cur_next_bundle, cnt_bundle = get_path_cur_next_bundle(args, \
                    path_cnt_saved_bundle, data_type, path_next_bundle_test=path_next_bundle_eval_or_test)
            if not (args.if_fast_train and args.if_comet_baseline):
                # not saving retriever
                while os.path.exists(path_prev_next_bundle):
                    time.sleep(5)
                try:
                    torch.save(next_bundle, path_cur_next_bundle)
                except:
                    time.sleep(5)
                    print("Exception occurs when saving next_bundle")
                    torch.save(next_bundle, path_cur_next_bundle)
            cnt_bundle[id_cnt_bundle] += 1
            torch.save(cnt_bundle, path_cnt_saved_bundle)
        if id_bundle == 0:
            cur_bundle = next_bundle
        # need not to re-embed the train cases
        # retrieved_cases_cur_bundle = wait_get_remove_cases_for_bundle(path_retrieved_encoded_cases_eval_or_test)
        if data_type == 'eval':
            ## only when if_fast_train and if_comet_baseline, we do not need to wait for retrieved_cases_cur_bundle
            if not (args.if_fast_train and args.if_comet_baseline):
                retrieved_cases_cur_bundle = wait_get_remove_cases_for_bundle_while_deleting_bad_cases_file(args, \
                path_cnt_retrieved_bundle, data_type,  path_retrieved_encoded_cases_eval=path_retrieved_encoded_cases_eval_or_test)
        elif data_type == 'test':
            ## only when if_fast_train and if_comet_baseline, we do not need to wait for retrieved_cases_cur_bundle
            if not (args.if_fast_train and args.if_comet_baseline):
                retrieved_cases_cur_bundle = wait_get_remove_cases_for_bundle_while_deleting_bad_cases_file(args, \
                path_cnt_retrieved_bundle, data_type,  path_retrieved_encoded_cases_test=path_retrieved_encoded_cases_eval_or_test)
        # FQ: add 'and num_bundles > 1', in case num_bundles == 1
        if id_bundle == 0 and num_bundles > 1:
            next_bundle = next(itr_eval_bundleloader)
            # about next_bundle
            if data_type == 'eval':
                path_prev_next_bundle, path_cur_next_bundle, cnt_bundle = get_path_cur_next_bundle(args, \
                    path_cnt_saved_bundle, data_type, path_next_bundle_eval=path_next_bundle_eval_or_test)
            elif data_type == 'test':
                path_prev_next_bundle, path_cur_next_bundle, cnt_bundle = get_path_cur_next_bundle(args, \
                    path_cnt_saved_bundle, data_type, path_next_bundle_test=path_next_bundle_eval_or_test)
            while os.path.exists(path_prev_next_bundle):
                time.sleep(5)
            try:
                torch.save(next_bundle, path_cur_next_bundle)
            except:
                time.sleep(5)
                print("Exception occurs when saving next_bundle")
                torch.save(next_bundle, path_cur_next_bundle)
            cnt_bundle[id_cnt_bundle] += 1
            torch.save(cnt_bundle, path_cnt_saved_bundle)

        if "bart" in args.generator_model_type or "gpt2" in args.generator_model_type or "bert" in args.generator_model_type or "t5" in args.generator_model_type:
            if args.if_fast_train:
                case_aug_cur_bundle = concat_cur_bundle_and_encoded_cases_EOSfixed_Bart_randomly_mask_demonstrations(args, cur_bundle, retrieved_cases_cur_bundle, tokenizer_gene, data_type)
            else:
                case_aug_cur_bundle = concat_cur_bundle_and_encoded_cases_EOSfixed_Bart(args, cur_bundle, retrieved_cases_cur_bundle, tokenizer_gene)
        else:
            raise Exception("Not supported generator_model_type: ", args.generator_model_type)


        if if_eval_analysis == False:
            # case_aug_cur_bundle only need to contain input for model_generator
            case_aug_cur_bundle = case_aug_cur_bundle[0:3]
        # get dataloader_in_batch for current bundle
        data_in_batch = TensorDataset(*case_aug_cur_bundle)
        sampler_in_batch = SequentialSampler(data_in_batch)
        if data_type == 'eval':
            dataloader_in_batch = DataLoader(data_in_batch, sampler=sampler_in_batch, batch_size=args.dev_batch_size)
        elif data_type == 'test':
            dataloader_in_batch = DataLoader(data_in_batch, sampler=sampler_in_batch, batch_size=args.test_batch_size)
        for step, batch in enumerate(dataloader_in_batch):
            batch_size = batch[0].size()[0]
            # print('batch_size:', batch_size)
            # input_ids: [len_bundle, n_doc, cases_per_doc * input_len_gene + 1 + input_len_gene + 1]
            input_ids, attention_mask, lm_labels = batch
            # input_ids = batch[0]
            with torch.no_grad():
                #   batch: [case_aug_gene_input_id, case_aug_gene_attention_mask, case_aug_gene_lm_labels]
                # case_aug_gene_input_id: [batch_size, cases_per_doc * input_len_gene + 1 + input_len_gene + 1]
                if if_eval_analysis:
                    loss, logits, tmp_embed_docs, tmp_embed_cur_query, batch_accuracy = batch_step_eval_analysis(args, model_generator, model_retriever, model_retriever_doc, batch, tokenizer_gene)
                    if id_bundle == 0 and step == 0:
                        Embed_docs = tmp_embed_docs
                        Embed_cur_query = tmp_embed_cur_query
                        Loss_eval.append(loss.item())
                    else:
                        Embed_docs = torch.cat((Embed_docs, tmp_embed_docs), dim=0)
                        Embed_cur_query = torch.cat((Embed_cur_query, tmp_embed_cur_query), dim=0)
                        Loss_eval.append(loss.item())
                    # print("step:{}, Embed_docs.size():{}".format(step, Embed_docs.size()))
                else:
                    loss, logits, batch_accuracy = batch_step_eval(args, model_generator, batch, tokenizer_gene)
                # print('eval_loss:', eval_loss, 'loss:', loss, 'batch_size:', batch_size, 'input_ids.size():', input_ids.size())
                eval_loss += loss * batch_size
                # eval_loss += loss
                nb_eval_steps += batch_size
                if args.dataset_selection == 4 or args.dataset_selection == 5 or args.dataset_selection == 6:
                    eval_ttl_cnt_correct += batch_accuracy * batch_size
                # print('batch_size:{}, loss:{}, eval_loss:{}, nb_eval_steps:{}'.format(batch_size, loss, eval_loss, nb_eval_steps))
                # print some examples
                if step in display_batch_indices:
                    value, indices = logits.max(dim=-1)
                    sample_index = random.randint(0, batch_size - 1)
                    print("input_ids:", tokenizer_gene.decode(input_ids[sample_index][0].tolist()))
                    # print("attention_mask:", attention_mask[sample_index][0].tolist())
                    # tmp_lm_labels = lm_labels[sample_index][0].tolist()
                    # tmp_lm_labels = [1 if tmp_lm_labels[i] == -100 else tmp_lm_labels[i] for i in range(len(tmp_lm_labels))]
                    # print("lm_labels:", tokenizer_gene.decode(tmp_lm_labels))
                    # IMPORTANT: add max_additional_cases
                    if "gpt2" in args.generator_model_type:
                        output = indices[sample_index].tolist()[-args.max_e2:]
                        output = tokenizer_gene.decode(output)
                    elif "bart" in args.generator_model_type or "bert" in args.generator_model_type or "t5" in args.generator_model_type:
                        output = indices[sample_index].tolist()
                        try:
                            eos_pos = output.index(eos_token)
                            output = tokenizer_gene.decode(output[:eos_pos])
                        except:
                            output = tokenizer_gene.decode(output)
                    else:
                        raise NotImplementError
                    if step == 150:
                        # to know whether use max_e2+1 or max_e2
                        print("output = indices[sample_index].tolist()[-args.max_e2:]")
                        print("output ids:", output)
                    print("output:", output)
        cur_bundle = next_bundle
    eval_loss = eval_loss / nb_eval_steps
    if args.dataset_selection == 4 or args.dataset_selection == 5 or args.dataset_selection == 6:
        eval_accuracy = eval_ttl_cnt_correct / nb_eval_steps
        print('eval_loss:{}, {}_accuracy: {}, nb_eval_steps:{}'.format(eval_loss, data_type, eval_accuracy, nb_eval_steps))
    else:
        eval_accuracy = None
        print('eval_loss:{}, nb_eval_steps:{}'.format(eval_loss, nb_eval_steps))
    ## send signal to retriever
    os.remove(path_signal_file_if_eval_or_test)
    assert not os.path.exists(path_signal_file_if_eval_or_test)
    if if_eval_analysis:
        torch.save(Embed_docs.to('cpu'), os.path.join(args.output_dir, 'Embed_docs.pt'))
        torch.save(Embed_cur_query.to('cpu'), os.path.join(args.output_dir, 'Embed_cur_query.pt'))
        torch.save(Loss_eval, os.path.join(args.output_dir, 'Loss_eval.pt'))
    return eval_loss.item(), eval_accuracy

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--generator_model_type", type=str, default="gpt2-lmhead",
                        help="model type: bart-base/t5-base/gpt2-lmhead/...")
    parser.add_argument("--retriever_model_type", type=str, default="dpr",
                        help="model type: dpr/bert/...")
    parser.add_argument("--toy", action="store_true", help="test code")

    parser.add_argument("--do_train", action="store_true", help="do training")
    parser.add_argument("--do_test", action="store_true", help="do testing")
    parser.add_argument("--do_eval", action="store_true", help="do evaluation in the end")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--dataStore_dir", default="/export/home/zonglin001/", type=str, required=False, help="The home directory of zonglin.")
    parser.add_argument("--train_dataset", type=str, nargs="+", default=["./Data/conceptnet/train100k_CN_sorted.txt"])
    # parser.add_argument("--eval_dataset", type=str, nargs="+", default=["data/conceptnet/dev1_CN.txt", "data/conceptnet/dev2_CN.txt"])
    parser.add_argument("--eval_dataset", type=str, nargs="+", default=["./Data/conceptnet/dev1_CN_sorted.txt"])
    parser.add_argument("--test_dataset", type=str, nargs="+", default=["./Data/conceptnet/test_CN_sorted.txt"])
    parser.add_argument("--add_prefix", action="store_true",
                        help="add a prefix at the beginning of each input when train with multiple dataset")
    # parser.add_argument("--add_separator", action="store_true", help="add <sep> between sub/rel/obj")
    parser.add_argument("--predict_part", type=str, default="obj", choices=["sub", "rel", "obj", "all"],
                        help="predict which part of the triples")
    # newly added in 8/21/2021; to calculate the proper max_additional_cases
    parser.add_argument("--num_cases_per_query", type=int, default=3)
    parser.add_argument("--max_additional_cases", type=int, default=150)
    parser.add_argument("--max_e1", type=int, default=24)
    parser.add_argument("--max_r", type=int, default=10)
    parser.add_argument("--max_e2", type=int, default=36)

    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--no_pretrain", action="store_true", help="w/o pretrained parameters initialized")
    parser.add_argument("--train_batch_size", type=int, default=16)
    parser.add_argument("--dev_batch_size", type=int, default=8)
    parser.add_argument("--test_batch_size", type=int, default=1)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--logging_steps', type=int, default=150)
    parser.add_argument("--eval_per_steps", type=int, default=5000)
    # change from 16 to 20: 8/23/2021 9:46 p.m.
    parser.add_argument("--num_train_epochs", type=int, default=20)
    parser.add_argument("--max_grad_norm", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--warmup_proportion", type=float, default=0.002)
    parser.add_argument("--lr_schedule", type=str, default="warmup_linear")
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)

    # added
    parser.add_argument("--root_data_dir", type=str, default="./Data/conceptnet/", help="data dir for current dataset; currently used for subset_selection")
    # parser.add_argument("--train_cases_dir", type=str, nargs="+", default=["/home/zy223/CBR/pytorch-transformers-comet/examples/conceptnet_cases/train_cases.txt"])
    # parser.add_argument("--val_cases_dir", type=str, nargs="+", default=["/home/zy223/CBR/pytorch-transformers-comet/examples/conceptnet_cases/val_cases.txt"])
    # parser.add_argument("--test_cases_dir", type=str, nargs="+", default=["/home/zy223/CBR/pytorch-transformers-comet/examples/conceptnet_cases/test_cases.txt"])
    parser.add_argument("--if_without_case", action="store_true", help="Filter all cases as '', to compare the effect of cases")
    # dataset_selection: 0: conceptnet 1: atomic 2: Shakespeare text style transfer
    # 3: e2e (table2text) 4: sentiment sentence classification dataset; 5: financial phase bank dataset; 6: yelp review
    parser.add_argument("--dataset_selection", type=int, default=0)
    parser.add_argument("--n_doc", type=int, default=3)
    parser.add_argument("--num_btch_in_bundle", type=int, default=500)
    parser.add_argument("--smooth_score", type=float, default=0.05)
    parser.add_argument("--if_froze_both_retriever", action="store_true", help="if the lr for retriever doc and retriever is 0")
    parser.add_argument("--if_only_froze_doc_retriever", action="store_true", help="if the lr for retriever doc is 0")
    parser.add_argument("--if_comet_baseline", action="store_true", help="comet experiment")
    parser.add_argument("--if_only_use_retrieved_target", action="store_true", help="if only use retrieved target during generation (not using retrieved source)")
    parser.add_argument("--if_only_use_relation_and_retrieved_target", action="store_true", help="if only use relation and retrieved target during generation (not using retrieved source)")
    parser.add_argument("--rand_simi_score", action="store_true", help="if using random simi_score")
    parser.add_argument("--use_obj_for_retrieval", action="store_true", help="if using obj for retrieval (get embedding and similarity score)")
    parser.add_argument("--use_only_sub_rel_for_retrieval", action="store_true", help="if only using sub for retrieval (get embedding and similarity score)")
    parser.add_argument("--if_with_strt_mid_promp", action="store_true", help="if use 'Here are some similar cases to infer from: ' and 'Now you can infer: '")
    parser.add_argument("--if_use_relation_for_shakes", action="store_true", help="Whether use relation for shakes dataset (Shakespeare's style is)")
    parser.add_argument("--use_special_tokens_to_split_retrieved_cases", action="store_true", help="<split_cases> and <split_source/target>")
    parser.add_argument("--if_eval_analysis", action="store_true", help="whether to generate embedding for analysis during test time")
    # subset_selection: 0~6, -1 means not using subset
    parser.add_argument("--subset_selection", type=int, default=-1)
    parser.add_argument("--patience", type=int, default=10, help='for early stopping')
    parser.add_argument("--if_not_adding_special_relation_tokens", action="store_true", help="not adding <oReact> for instance")
    parser.add_argument("--if_without_none", action="store_true", help="You do NOT need to include it in command line, as it will adjust itself in the following code; if not using none data in atomic; will use different train/val/tst data; ")
    parser.add_argument("--num_sample", type=int, default=1, help="the nth time of sampling data to use")
    parser.add_argument("--if_use_nshot_data", action="store_true", help="The old version of data doesn't try to maintain nshot but only keep the same number of total few-shot data; this new version of data try to maintain nshot")
    parser.add_argument("--if_randomly_mask_demonstrations", action="store_true", help="if also use plain few-shot train data (without in-context demonstrations) for training CBRF")
    parser.add_argument("--prob_randomly_mask_demonstrations", type=float, default=0.5, help="when if_randomly_mask_demonstrations == True, the prob to mask demonstrations")
    parser.add_argument("--if_fast_train", action="store_true", help="only use 1 GPU to do the work, but can't update retriever; also uses smooth loss, so nealy exact the same as origin batch_step()")
    parser.add_argument("--if_try_one_gpu", action="store_true", help="whether only use 1 GPU")


    args = parser.parse_args()

    # We don't allow coexistence of (if_comet_baseline and if_only_use_retrieved_target)
    assert not (args.if_comet_baseline and args.if_only_use_retrieved_target)
    assert not (args.if_only_use_retrieved_target and args.if_only_use_relation_and_retrieved_target)
    assert not (args.if_comet_baseline and args.if_only_use_relation_and_retrieved_target)
    assert args.prob_randomly_mask_demonstrations >= 0.0 and args.prob_randomly_mask_demonstrations <= 1.0
    # if_randomly_mask_demonstrations and if_comet_baseline can't be true at the same time
    # a small hyperparameter error preventer
    assert not (args.if_randomly_mask_demonstrations and args.if_comet_baseline)
    # Can't update retriever while using args.if_fast_train
    if args.if_try_one_gpu:
        assert args.if_fast_train
    if args.if_fast_train:
        assert args.if_froze_both_retriever
        # Do not support generate embedding when args.if_fast_train == True
        assert not args.if_eval_analysis
    # prevent from using wrong match
    if 'gpt2' in args.generator_model_type:
        assert 'gpt2' in args.output_dir
    elif 'bart' in args.generator_model_type:
        assert 'bart' in args.output_dir
    print("args.dataset_selection: ", args.dataset_selection)

    if args.dataset_selection == 0:
        if "t5" in args.generator_model_type:
            args.max_e2 = 46
        args.if_without_none = False
        if args.subset_selection == -1:
            args.patience = 5
        if args.subset_selection == 0:
            args.eval_per_steps = 30
        elif args.subset_selection == 1:
            args.eval_per_steps = 60
        elif args.subset_selection == 2:
            args.eval_per_steps = 120
        elif args.subset_selection == 3:
            args.eval_per_steps = 240
        elif args.subset_selection == 4:
            args.eval_per_steps = 500
        elif args.subset_selection == 5:
            args.eval_per_steps = 1000
        elif args.subset_selection == 6:
            args.eval_per_steps = 2000
        elif args.subset_selection == -1:
            args.eval_per_steps = 5000
        else:
            raise NotImplementedError
    elif args.dataset_selection == 1:
        args.max_e1 = 25
        args.max_r = 15
        args.max_e2 = 38
        # Q: changed from 200 to 250: 8/21/2021: 11:50 p.m.
        args.max_additional_cases = 250
        args.num_train_epochs = 2
        args.train_dataset = ["./Data/atomic/v4_atomic_trn.csv"]
        args.eval_dataset = ["./Data/atomic/v4_atomic_dev.csv"]
        args.test_dataset = ["./Data/atomic/v4_atomic_tst.csv"]
        args.root_data_dir = "./Data/atomic/"
        # if_without_none: if not using none data in atomic; will use different train/val/tst data
        args.if_without_none = True
        print("INFO: using atomic data without 'None' tuples")
        # 8/28/2021; if using full set, patience can be smaller
        if args.subset_selection == -1:
            args.patience = 5
        if args.subset_selection == 0:
            args.eval_per_steps = 30
        elif args.subset_selection == 1:
            args.eval_per_steps = 60
        elif args.subset_selection == 2:
            args.eval_per_steps = 120
        elif args.subset_selection == 3:
            args.eval_per_steps = 240
        elif args.subset_selection == 4:
            args.eval_per_steps = 500
        elif args.subset_selection == 5:
            args.eval_per_steps = 1000
        elif args.subset_selection == 6:
            args.eval_per_steps = 2000
        elif args.subset_selection == -1:
            args.eval_per_steps = 5000
        else:
            raise NotImplementedError
    elif args.dataset_selection == 2:
        args.max_e1 = 130
        if args.if_use_relation_for_shakes:
            args.max_r = 6
        else:
            args.max_r = 2
        args.max_e2 = 140
        args.max_additional_cases = 500
        args.num_train_epochs = 15
        args.root_data_dir = "./Data/shakes/"
        args.if_without_none = False
        print("warning: need to manually set up args.eval_per_steps.")
    elif args.dataset_selection == 3:
        args.max_e1 = 60
        args.max_r = 2
        args.max_e2 = 95
        args.max_additional_cases = 400
        args.num_train_epochs = 15
        args.root_data_dir = "./Data/e2e/"
        args.if_without_none = False
        print("warning: need to manually set up args.eval_per_steps.")
    elif args.dataset_selection == 4:
        args.max_e1 = 110
        args.max_r = 2
        args.max_e2 = 30
        args.max_additional_cases = 400
        args.num_train_epochs = 5000
        args.root_data_dir = "./Data/sentiment/splitted/"
        args.if_without_none = False
        assert args.subset_selection >= -1 and args.subset_selection <= 3
        if args.subset_selection == 3:
            args.eval_per_steps = 20
        elif args.subset_selection == 0:
            args.eval_per_steps = 50
        elif args.subset_selection == 1:
            args.eval_per_steps = 150
        elif args.subset_selection == 2:
            args.eval_per_steps = 350
        elif args.subset_selection == -1:
            args.eval_per_steps = 500
        else:
            raise NotImplementedError
    elif args.dataset_selection == 5:
        args.max_e1 = 130
        args.max_r = 2
        args.max_e2 = 5
        args.max_additional_cases = 411
        args.num_train_epochs = 5000
        args.root_data_dir = "./Data/financial_phasebank/splitted/"
        args.if_without_none = False
        if args.subset_selection == 0:
            args.eval_per_steps = 20
        elif args.subset_selection == 1:
            args.eval_per_steps = 50
        elif args.subset_selection == 2:
            args.eval_per_steps = 150
        elif args.subset_selection == 3:
            args.eval_per_steps = 350
        elif args.subset_selection == -1:
            args.eval_per_steps = 500
        else:
            raise NotImplementedError
        # financial dataset has three labels, make sure that there's examples from each of the labels
        assert args.num_cases_per_query >= 3
    elif args.dataset_selection == 6:
        args.max_e1 = 130
        args.max_r = 2
        args.max_e2 = 5
        args.max_additional_cases = 700
        args.num_train_epochs = 5000
        args.root_data_dir = "./Data/yelp/splitted/"
        args.if_without_none = False
        if args.subset_selection == 0:
            args.eval_per_steps = 20
        elif args.subset_selection == 1:
            args.eval_per_steps = 50
        elif args.subset_selection == 2:
            args.eval_per_steps = 150
        elif args.subset_selection == 3:
            args.eval_per_steps = 350
        elif args.subset_selection == -1:
            args.eval_per_steps = 500
        else:
            raise NotImplementedError
        # yelp dataset has three labels, make sure that there's examples from each of the labels
        assert args.num_cases_per_query >= 5
    else:
        raise Exception("Not supported dataset_selection")

    # Only atomic uses if_without_none
    if args.if_without_none:
        assert args.dataset_selection == 1

    # adjust args.max_additional_cases according to args.num_cases_per_query; 8/21/2021
    # although max_additional_cases has not been used strictly for restricting the length of additional_cases; 8/21/2021
    if not args.max_additional_cases >= args.num_cases_per_query * (args.max_e1 + args.max_r + args.max_e2):
        args.max_additional_cases = args.num_cases_per_query * (args.max_e1 + args.max_r + args.max_e2)
        print("Adjusted args.max_additional_cases to fit args.num_cases_per_query: ", args.max_additional_cases)
    if args.subset_selection != -1:
        # uses early stopping
        args.num_train_epochs = 500
    # change args.additional_sample_name according to args.num_sample
    if args.num_sample == 1:
        args.additional_sample_name = ""
    elif args.num_sample == 2:
        args.additional_sample_name = '_secondSample'
    elif args.num_sample == 3:
        args.additional_sample_name = '_thirdSample'
    else:
        raise Exception("not accepted num_sample")
    # change args.additional_sampling_method_name according to args.if_use_nshot_data
    if args.if_use_nshot_data:
        args.additional_sampling_method_name = '_NShot'
    else:
        args.additional_sampling_method_name = ""
    if args.if_without_none:
        args.additional_sampling_method_name += '_withoutNoneTuple'

    if args.if_comet_baseline == True:
        print('INFO: running COMET experiment..')

    if args.rand_simi_score == True:
        print('INFO: unsing random simi_score..')

    assert args.predict_part == "obj"

    set_seed(args.seed)

    n_gpu = torch.cuda.device_count()
    logger.info("device: {}, n_gpu {}".format(device1, n_gpu))
    if not n_gpu >= 2 and not args.if_try_one_gpu:
        raise Exception("n_gpu is not enough: {}".format(n_gpu))
    assert not (args.if_froze_both_retriever and args.if_only_froze_doc_retriever)

    # paths
    # use different path between train and eval to prevent interrupting with each other
    # "encoded_cases" is used in util_TST file in wait_get_remove_cases_for_bundle_while_deleting_bad_cases_file function
    path_retrieved_encoded_cases = os.path.join(args.output_dir, 'encoded_cases_train.pt')
    path_retrieved_encoded_cases_eval = os.path.join(args.output_dir, 'encoded_cases_eval.pt')
    path_retrieved_encoded_cases_test = os.path.join(args.output_dir, 'encoded_cases_test.pt')
    path_next_bundle = os.path.join(args.output_dir, 'next_bundle_train.pt')
    path_next_bundle_eval = os.path.join(args.output_dir, 'next_bundle_eval.pt')
    path_next_bundle_test = os.path.join(args.output_dir, 'next_bundle_test.pt')
    path_retriever = os.path.join(args.output_dir, 'retriever.pt')
    path_retriever_eval = os.path.join(args.output_dir, 'retriever_eval.pt')
    path_retriever_test = os.path.join(args.output_dir, 'retriever_test.pt')
    path_retriever_doc = os.path.join(args.output_dir, 'retriever_doc.pt')
    path_retriever_doc_eval = os.path.join(args.output_dir, 'retriever_doc_eval.pt')
    path_retriever_doc_test = os.path.join(args.output_dir, 'retriever_doc_test.pt')
    path_generator = os.path.join(args.output_dir, 'generator.pt')
    path_retriever_final = os.path.join(args.output_dir, 'retriever_final.pt')
    path_retriever_doc_final = os.path.join(args.output_dir, 'retriever_doc_final.pt')
    path_generator_final = os.path.join(args.output_dir, 'generator_final.pt')
    path_example_batch = os.path.join(args.output_dir, 'example_batch.pt')
    # if_finish signal to retriever
    path_prev_if_no_need_for_retrieval = os.path.join(args.output_dir, 'finished.pt')
    path_if_no_need_for_retrieval = os.path.join(args.output_dir, 'finished-no_need_for_retrieval.pt')
    path_if_finished_training = os.path.join(args.output_dir, 'training_finished.pt')
    # path_cnt_saved_bundle: number of next_bundle that has been saved in the past
    path_cnt_saved_bundle = os.path.join(args.output_dir, 'cnt_saved_bundle.pt')
    # path_cnt_saved_bundle: number of next_bundle that has been retrieved with encoded_cases in the past
    path_cnt_retrieved_bundle = os.path.join(args.output_dir, 'cnt_retrieved_bundle.pt')
    path_tensorboard = os.path.join(args.output_dir, args.output_dir.split('/')[-1])

    # # QQ: not using old data at all
    # if os.path.exists(path_if_finished_training):
    #     os.remove(path_if_finished_training)
    assert args.do_train or args.do_test
    ## File systems
    # about "$dataStore_dir$/ECBRF_shared_data_for_reuse/", which is to stored processed data for reuse
    if not os.path.exists(os.path.join(args.dataStore_dir, "ECBRF_shared_data_for_reuse")):
        os.makedirs(os.path.join(args.dataStore_dir, "ECBRF_shared_data_for_reuse"))
    # about args.output_dir
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        print("Output_dir is an empty file now")
    # output_dir already exists and training is not finished
    # elif args.do_train and not args.do_test:
    elif args.do_train and not os.path.exists(path_if_finished_training):
        # make sure the file is clear
        shutil.rmtree(args.output_dir)
        assert not os.path.exists(args.output_dir)
        os.makedirs(args.output_dir)
        print("Training not finished yet, output_dir is an empty file now")
    # (only do_test while not do_train) or (Training is finished)
    # elif not args.do_train and args.do_test:
    elif args.do_test and os.path.exists(path_if_finished_training):
        # allow the retriever to work
        if os.path.exists(path_if_no_need_for_retrieval):
            os.remove(path_if_no_need_for_retrieval)
        remaining_files = os.listdir(args.output_dir)
        for tmp_file in remaining_files:
            if tmp_file.startswith("encoded_cases_test"):
                os.remove(os.path.join(args.output_dir, tmp_file))
            if tmp_file.startswith("next_bundle_test"):
                os.remove(os.path.join(args.output_dir, tmp_file))
            if tmp_file.startswith("cnt_saved_bundle.pt"):
                os.remove(os.path.join(args.output_dir, tmp_file))
            if tmp_file.startswith("cnt_retrieved_bundle.pt"):
                os.remove(os.path.join(args.output_dir, tmp_file))
        print("Files exist in output_dir before evaluation begins:")
        print(remaining_files)
    else:
        # do_test while training is not finished
        raise Exception


    # delete files from last time's running (that is misleading for this time)
    if not args.do_train and args.do_test:
        if os.path.exists(os.path.join(args.output_dir, "cnt_retrieved_bundle.pt")):
            os.remove(os.path.join(args.output_dir, "cnt_retrieved_bundle.pt"))
        if os.path.exists(os.path.join(args.output_dir, "cnt_saved_bundle.pt")):
            os.remove(os.path.join(args.output_dir, "cnt_saved_bundle.pt"))
        if os.path.exists(os.path.join(args.output_dir, "retriever_doc_test.pt")):
            os.remove(os.path.join(args.output_dir, "retriever_doc_test.pt"))
        if os.path.exists(os.path.join(args.output_dir, "retriever_test.pt")):
            os.remove(os.path.join(args.output_dir, "retriever_test.pt"))

    # tensorboard
    writer = SummaryWriter(path_tensorboard)
    # "bert-lmhead": (BertLMHeadModel, BertTokenizer, BertConfig, "bert-base-uncased")
    # "t5-small": (T5ForConditionalGeneration, T5Tokenizer, T5Config, "t5-small")
    # "t5-base": (T5ForConditionalGeneration, T5Tokenizer, T5Config, "t5-base")
    # "bart-large": (BartForConditionalGeneration, BartTokenizer, BartConfig, "facebook/bart-large")
    MODEL_CLASSES = {
        "gpt2-lmhead":(GPT2LMHeadModel, GPT2Tokenizer, GPT2Config, "gpt2"),
        "bart-base": (BartForConditionalGeneration, BartTokenizer, BartConfig, "facebook/bart-base")
    }
    # Model, Tokenizer, Config, Model_Name = MODEL_CLASSES[args.generator_model_type]
    Generator_Model, Generator_Tokenizer, Generator_Config, Generator_Model_Name = MODEL_CLASSES[args.generator_model_type]
    # Retriever_Model, Retriever_Tokenizer, Retriever_Config, Retriever_Model_Name = MODEL_CLASSES["bert"]

    ## tokenizer
    tokenizer_generator = Generator_Tokenizer.from_pretrained(Generator_Model_Name)
    # tokenizer_retriever = Retriever_Tokenizer.from_pretrained(Retriever_Model_Name)
    tokenizer_retriever = DPRQuestionEncoderTokenizer.from_pretrained('facebook/dpr-question_encoder-single-nq-base')
    tokenizer_retriever_doc = DPRContextEncoderTokenizer.from_pretrained('facebook/dpr-ctx_encoder-single-nq-base')

    tokenizer_generator = add_special_tokens(args, tokenizer_generator)
    # tokenizer_retriever = add_special_tokens(args, tokenizer_retriever)
    # tokenizer_retriever_doc = add_special_tokens(args, tokenizer_retriever_doc)

    ## load pretrained model; fit model with tokenizer
    model_generator = Generator_Model.from_pretrained(Generator_Model_Name, device_map="auto")
    model_retriever = DPRQuestionEncoder.from_pretrained('facebook/dpr-question_encoder-single-nq-base')
    model_retriever_doc = DPRContextEncoder.from_pretrained('facebook/dpr-ctx_encoder-single-nq-base')
    print("INFO: Using DPR pre-training models")

    model_generator.resize_token_embeddings(len(tokenizer_generator))
    # word_embeddings = model_generator.transformer.wte.weight
    # position_embeddings = model_generator.transformer.wpe.weight
    # model_generator.to(device1)
    if not args.if_try_one_gpu:
        # model_retriever.resize_token_embeddings(len(tokenizer_retriever))
        model_retriever.to(device2)
        # model_retriever_doc.resize_token_embeddings(len(tokenizer_retriever_doc))
        model_retriever_doc.to(device2)

    # Load and encode the datasets
    logger.info("Loading datasets ...")

    # whether expand rel with natural sentence
    def rel_lang(filename):
        if "vg" in filename.lower():
            return False
        elif "cn" in filename.lower():
            return True
        elif "easyfb" in filename.lower():
            return False
        elif "fb" in filename.lower():
            return True
        elif "atomic" in filename.lower():
            return True
        else:
            raise Exception

    path_tensor_datasets = find_path_tensor_dataset(args)

    if not os.path.exists(path_tensor_datasets):
        print("Pre-processed tensor_datasets not found, begin to pre-process datasets...")
        start_preprocessing_time = time.time()
        if args.dataset_selection == 0:
            if args.subset_selection == -1:
                train_datasets = [load_conceptnet_pure(dataset_path=train_dataset,
                                                     rel_lang=rel_lang(train_dataset),
                                                     discard_negative=True
                                                     ) for train_dataset in args.train_dataset]
                eval_datasets = [load_conceptnet_pure(dataset_path=eval_dataset,
                                                     rel_lang=rel_lang(eval_dataset),
                                                     discard_negative=True
                                                     ) for eval_dataset in args.eval_dataset]
                test_datasets = [load_conceptnet_pure(dataset_path=test_dataset,
                                                     rel_lang=rel_lang(test_dataset),
                                                     discard_negative=True,
                                                    ) for test_dataset in args.test_dataset]
            else:
                train_dataset_dir = os.path.join(args.root_data_dir, 'train_subset_'+str(args.subset_selection) + args.additional_sampling_method_name + args.additional_sample_name)
                eval_dataset_dir = os.path.join(args.root_data_dir, 'eval_subset_'+str(600))
                test_dataset_dir = os.path.join(args.root_data_dir, 'test_subset_'+str(1200))
                with open(train_dataset_dir, 'rb') as f:
                    train_datasets = [pickle.load(f)]
                with open(eval_dataset_dir, 'rb') as f:
                    eval_datasets = [pickle.load(f)]
                with open(test_dataset_dir, 'rb') as f:
                    test_datasets = [pickle.load(f)]
        elif args.dataset_selection == 1:
            if args.subset_selection == -1:
                train_datasets = [load_atomic_pure(dataset_path=train_dataset, rel_lang=rel_lang(train_dataset), toy=args.toy, if_without_none=args.if_without_none)
                                                    for train_dataset in args.train_dataset]
                eval_datasets = [load_atomic_pure(dataset_path=eval_dataset, rel_lang=rel_lang(eval_dataset), toy=args.toy, if_without_none=args.if_without_none)
                                                    for eval_dataset in args.eval_dataset]
                test_datasets = [load_atomic_pure(dataset_path=test_dataset, rel_lang=rel_lang(test_dataset), toy=args.toy, if_without_none=args.if_without_none)
                                                    for test_dataset in args.test_dataset]
            else:
                train_dataset_dir = os.path.join(args.root_data_dir, 'train_subset_'+str(args.subset_selection) + args.additional_sampling_method_name + args.additional_sample_name)
                # 1000 means using "None" tuple data; 900 means not using
                if args.if_without_none:
                    eval_size = 900
                    test_size = 72872
                else:
                    eval_size = 1000
                    test_size = 87481
                eval_dataset_dir = os.path.join(args.root_data_dir, 'eval_subset_'+str(eval_size))
                # 87481 means using "None" tuple data; 72872 means not using
                test_dataset_dir = os.path.join(args.root_data_dir, 'test_subset_'+str(test_size))
                with open(train_dataset_dir, 'rb') as f:
                    train_datasets = [pickle.load(f)]
                with open(eval_dataset_dir, 'rb') as f:
                    eval_datasets = [pickle.load(f)]
                with open(test_dataset_dir, 'rb') as f:
                    test_datasets = [pickle.load(f)]
        elif args.dataset_selection == 2:
            if args.subset_selection == -1:
                train_datasets, eval_datasets, test_datasets = load_shakespear(args, dataset_path=args.root_data_dir)
            else:
                train_dataset_dir = os.path.join(args.root_data_dir, 'train_subset_'+str(args.subset_selection) + args.additional_sample_name)
                eval_dataset_dir = os.path.join(args.root_data_dir, 'eval_subset_'+str(1179))
                test_dataset_dir = os.path.join(args.root_data_dir, 'test_subset_'+str(1411))
                with open(train_dataset_dir, 'rb') as f:
                    train_datasets = [pickle.load(f)]
                with open(eval_dataset_dir, 'rb') as f:
                    eval_datasets = [pickle.load(f)]
                with open(test_dataset_dir, 'rb') as f:
                    test_datasets = [pickle.load(f)]
        elif args.dataset_selection == 3:
            if args.subset_selection == -1:
                train_datasets = load_e2e(args, dataset_path=args.root_data_dir, data_type='train')
                eval_datasets = load_e2e(args, dataset_path=args.root_data_dir, data_type='eval')
                test_datasets = load_e2e(args, dataset_path=args.root_data_dir, data_type='test')
            else:
                train_dataset_dir = os.path.join(args.root_data_dir, 'train_subset_'+str(args.subset_selection) + args.additional_sample_name)
                eval_dataset_dir = os.path.join(args.root_data_dir, 'eval_subset_'+str(1000))
                test_dataset_dir = os.path.join(args.root_data_dir, 'test_subset_'+str(4693))
                with open(train_dataset_dir, 'rb') as f:
                    train_datasets = [pickle.load(f)]
                with open(eval_dataset_dir, 'rb') as f:
                    eval_datasets = [pickle.load(f)]
                with open(test_dataset_dir, 'rb') as f:
                    test_datasets = [pickle.load(f)]
        elif args.dataset_selection == 4 or args.dataset_selection == 5 or args.dataset_selection == 6:
            if args.subset_selection == -1:
                train_datasets, eval_datasets, test_datasets = load_sentiment_data(splitted_data_dir=args.root_data_dir, if_add_e2Rel=True)
            else:
                _, eval_datasets, test_datasets = load_sentiment_data(splitted_data_dir=args.root_data_dir, if_add_e2Rel=True)
                with open(os.path.join(args.root_data_dir, "{}_subset_{}_data.npy".format("train", args.subset_selection)), 'rb') as f:
                    train_datasets = np.load(f)
                # print("train_datasets[0]", train_datasets[0])
                # get new line_id according to the size of subset
                def reorder_line_id(data_set):
                    data_set_reordered_line_id = []
                    for cur_id in range(len(data_set)):
                        assert len(data_set[cur_id]) == 5
                        cur_data = data_set[cur_id][0:4].tolist()
                        cur_data.append(cur_id)
                        data_set_reordered_line_id.append(cur_data)
                    return data_set_reordered_line_id
                train_datasets = reorder_line_id(train_datasets)
                # print("train_datasets[0]", train_datasets[0])
            train_datasets = [train_datasets]
            eval_datasets = [eval_datasets]
            test_datasets = [test_datasets]


        train_datasets = [data for train_dataset in train_datasets for data in train_dataset]
        eval_datasets = [data for eval_dataset in eval_datasets for data in eval_dataset]
        test_datasets = [data for test_dataset in test_datasets for data in test_dataset]

        print('train data[0: 5]', train_datasets[0:5])

        # datasets: ([#train], [#eval], [#test])
        datasets = (train_datasets, eval_datasets, test_datasets)
        logger.info("Encoding datasets ...")
        encoded_datasets_generator = tokenize_and_encode(datasets, tokenizer_generator, model_type=Generator_Model_Name)
        # model_type='bert' as DPRTokenizers are the same as BertTokenizer
        encoded_datasets_retriever = tokenize_and_encode(datasets, tokenizer_retriever, model_type=args.retriever_model_type)
        print("Encodings succeed!")

        # Prepare inputs tensors and dataloaders
        tensor_datasets = preprocess_datasets_for_generator_and_retriever_and_retriever_doc_ProperEOS(args, path_tensor_datasets, encoded_datasets_generator, encoded_datasets_retriever, tokenizer_generator = tokenizer_generator, tokenizer_retriever = tokenizer_retriever)

        print('Generate tensor_datasets successfully!')
        print("Pre-processing takes %.2f seconds"%(time.time() - start_preprocessing_time))
    else:
        while True:
            try:
                tensor_datasets = torch.load(path_tensor_datasets)
                break
            except:
                time.sleep(5)
        print("tensor_datasets loaded successfully from existing file: ", path_tensor_datasets)
    train_tensor_dataset, eval_tensor_dataset, test_tensor_dataset = tensor_datasets[0], tensor_datasets[1], tensor_datasets[2]
    print('len(train_tensor_dataset[0]): ', len(train_tensor_dataset[0]))
    print('len(eval_tensor_dataset[0]):', len(eval_tensor_dataset[0]))
    print('len(test_tensor_dataset[0]): ', len(test_tensor_dataset[0]))

    # # if args.subset_selection != -1:
    # if args.subset_selection != -1 and args.dataset_selection < 4:
    #     # Q: chaning eval_per_steps accoring to num_steps in one epoch
    #     tmp_eval_per_steps = math.ceil(len(train_tensor_dataset[0]) / args.train_batch_size)
    #     args.eval_per_steps = np.minimum(tmp_eval_per_steps, args.eval_per_steps)
    # elif args.dataset_selection == 4:
    #     print("warning: need to manually set up args.eval_per_steps.")
    # else:
    #     pass
    steps_per_epoch = math.ceil(len(train_tensor_dataset[0]) / args.train_batch_size)
    # if steps_per_epoch > args.eval_per_steps:
    if steps_per_epoch > args.eval_per_steps:
        # newly added, to prevent too large eval steps
        if steps_per_epoch > 5000:
            steps_per_epoch = 5000
        args.eval_per_steps = steps_per_epoch
        print("args.eval_per_steps has been changed with steps_per_epoch: ", steps_per_epoch)

    args.logging_steps = np.minimum(args.eval_per_steps, 150)
    print("args.eval_per_steps: ", args.eval_per_steps)
    print("args.logging_steps: ", args.logging_steps)
    print(args)
    # # Q: only for debug
    # eval_tensor_dataset = eval_tensor_dataset[:10000]

    if args.generator_model_type == "gpt2-lmhead" or "t5" in args.generator_model_type:
        generator_eos_id = tokenizer_generator.encode(tokenizer_generator.eos_token)[0]
    elif args.generator_model_type == "bart-base" or args.generator_model_type == "bart-large" or 'bert' in args.generator_model_type:
        generator_eos_id = tokenizer_generator.encode(tokenizer_generator.eos_token)[1]
    else:
        raise NotImplementError

    train_bundle_size = args.train_batch_size * args.num_btch_in_bundle
    dev_bundle_size = args.dev_batch_size * args.num_btch_in_bundle
    test_bundle_size = args.test_batch_size * args.num_btch_in_bundle
    train_data = TensorDataset(*train_tensor_dataset)
    train_sampler = RandomSampler(train_data)
    train_dataloader_in_bundle = DataLoader(train_data, sampler=train_sampler, batch_size=train_bundle_size)

    eval_data = TensorDataset(*eval_tensor_dataset)
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader_in_bundle = DataLoader(eval_data, sampler=eval_sampler, batch_size=dev_bundle_size)

    test_data = TensorDataset(*test_tensor_dataset)
    test_sampler = SequentialSampler(test_data)
    test_dataloader_in_bundle = DataLoader(test_data, sampler=test_sampler, batch_size=test_bundle_size)

    ## total steps
    # t_total: number of steps that update parameters in total
    num_btch_in_bundle = args.num_btch_in_bundle
    num_bundle_in_one_epoch = len(train_dataloader_in_bundle)
    print("num_bundle_in_one_epoch: ", num_bundle_in_one_epoch)
    t_total = (args.num_train_epochs * num_bundle_in_one_epoch * num_btch_in_bundle) // args.gradient_accumulation_steps
    print('num_train_epochs: ', args.num_train_epochs)
    print('t_total: ', t_total)

    if args.do_train and not os.path.exists(path_if_finished_training):
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(tensor_datasets[0]))
        logger.info("  Num Epochs = %d", args.num_train_epochs)
        logger.info("  Each Epoch has %d steps, and %d actual steps w/ accumulation",
                    num_bundle_in_one_epoch * num_btch_in_bundle, num_bundle_in_one_epoch * num_btch_in_bundle // args.gradient_accumulation_steps)
        logger.info("  Total train batch size (w. accumulation) = %d",
                    args.train_batch_size * args.gradient_accumulation_steps)
        logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
        logger.info("  Total optimization steps = %d", t_total)

        # param_optimizer = list(model_generator.named_parameters()) + list(model_retriever.named_parameters())
        param_generator = list(model_generator.named_parameters())
        param_retriever = list(model_retriever.named_parameters())
        param_retriever_doc = list(model_retriever_doc.named_parameters())
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        if not args.if_froze_both_retriever and not args.if_comet_baseline:
            optimizer_grouped_parameters = [
                {"params": [p for n, p in param_generator if not any(nd in n for nd in no_decay)], "weight_decay": args.weight_decay, 'lr': args.learning_rate},
                {"params": [p for n, p in param_generator if any(nd in n for nd in no_decay)], "weight_decay": 0.0, 'lr': args.learning_rate},
                {"params": [p for n, p in param_retriever if not any(nd in n for nd in no_decay)], "weight_decay": args.weight_decay, 'lr': args.learning_rate / 3},
                {"params": [p for n, p in param_retriever if any(nd in n for nd in no_decay)], "weight_decay": 0.0, 'lr': args.learning_rate / 3},
                {"params": [p for n, p in param_retriever_doc if not any(nd in n for nd in no_decay)], "weight_decay": args.weight_decay, 'lr': args.learning_rate / 3},
                {"params": [p for n, p in param_retriever_doc if any(nd in n for nd in no_decay)], "weight_decay": 0.0, 'lr': args.learning_rate / 3}
                ]
            # print('INFO: generator and retriever share the same learning rate.')
            print("INFO: retriever and retriever_doc has 1/3 of generator's lr")
        elif args.if_only_froze_doc_retriever:
            optimizer_grouped_parameters = [
                {"params": [p for n, p in param_generator if not any(nd in n for nd in no_decay)], "weight_decay": args.weight_decay, 'lr': args.learning_rate},
                {"params": [p for n, p in param_generator if any(nd in n for nd in no_decay)], "weight_decay": 0.0, 'lr': args.learning_rate},
                {"params": [p for n, p in param_retriever if not any(nd in n for nd in no_decay)], "weight_decay": args.weight_decay, 'lr': args.learning_rate / 3},
                {"params": [p for n, p in param_retriever if any(nd in n for nd in no_decay)], "weight_decay": 0.0, 'lr': args.learning_rate / 3}
                ]
            # print('INFO: generator and retriever share the same learning rate.')
            print("INFO: retriever has 1/3 of generator's lr, while retriever_doc is frozen")
        else:
            optimizer_grouped_parameters = [
                {"params": [p for n, p in param_generator if not any(nd in n for nd in no_decay)], "weight_decay": args.weight_decay, 'lr': args.learning_rate},
                {"params": [p for n, p in param_generator if any(nd in n for nd in no_decay)], "weight_decay": 0.0, 'lr': args.learning_rate}
                ]
            print('INFO: Using frozen retriever..')
        print("total steps:", t_total)
        # num_warmup_steps = args.warmup_proportion * t_total
        if args.dataset_selection == 0:
            num_warmup_steps = 200
            if args.subset_selection >= 0:
                num_warmup_steps = np.maximum(50, int(args.eval_per_steps * 2))
                num_warmup_steps = np.minimum(num_warmup_steps, 200)
        else:
            num_warmup_steps = 300
            if args.subset_selection >= 0:
                num_warmup_steps = np.maximum(50, int(args.eval_per_steps * 2))
                num_warmup_steps = np.minimum(num_warmup_steps, 300)
        print('num_warmup_steps:', num_warmup_steps)
        # optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        optimizer = AdamW(optimizer_grouped_parameters, eps=args.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=t_total)

        # ttl_batch_steps: ttl batche steps; global_steps: ttl steps that updated parameters
        # should be: ttl_batch_steps = logging_steps * global_steps
        global_steps, ttl_batch_steps = 0, -1
        tr_loss, logging_loss = 0.0, 0.0
        tr_nll_loss, logging_nll_loss = 0.0, 0.0
        if args.dataset_selection == 4 or args.dataset_selection == 5 or args.dataset_selection == 6:
            best_accuracy = 0
        elif args.dataset_selection == 0 or args.dataset_selection == 1 or args.dataset_selection == 2 or args.dataset_selection == 3:
            best_loss = 1e10
        else:
            raise NotImplementedError
        patience = args.patience
        model_generator.train()
        model_retriever.train()
        model_retriever_doc.train()
        retrieved_cases_cur_bundle = None

        for cur_epoch_num in range(int(args.num_train_epochs)):
            # early stopping
            if patience < 0:
                break
            print("Epoch:", cur_epoch_num)
            itr_train_bundleloader = iter(train_dataloader_in_bundle)
            for id_bundle in range(num_bundle_in_one_epoch):
                # early stopping
                if patience < 0:
                    break
                # Boundary conditions
                # Save next_batches and get cur_batches
                if id_bundle < num_bundle_in_one_epoch - 1 or id_bundle == 0:
                    next_bundle = next(itr_train_bundleloader)
                    ## Save Retriever
                    if not (args.if_fast_train and args.if_comet_baseline):
                        while os.path.exists(path_retriever):
                            time.sleep(5)
                        torch.save(model_retriever.state_dict(), path_retriever)
                        # save model_retriever_doc for retriever.py to use
                        while os.path.exists(path_retriever_doc):
                            time.sleep(5)
                        torch.save(model_retriever_doc.state_dict(), path_retriever_doc)
                    # about next_bundle
                    path_prev_next_bundle, path_cur_next_bundle, cnt_bundle = get_path_cur_next_bundle(args, \
                        path_cnt_saved_bundle, 'train', path_next_bundle_train=path_next_bundle)
                    if not (args.if_fast_train and args.if_comet_baseline):
                        # might need some waiting (for retriever); if generator is faster
                        while os.path.exists(path_prev_next_bundle):
                            print('Warning: previous next_bundle still exists!')
                            time.sleep(5)
                        try:
                            torch.save(next_bundle, path_cur_next_bundle)
                        except:
                            time.sleep(5)
                            print("Exception occurs when saving next_bundle")
                            torch.save(next_bundle, path_cur_next_bundle)
                    cnt_bundle[0] += 1
                    torch.save(cnt_bundle, path_cnt_saved_bundle)

                if id_bundle == 0:
                    cur_bundle = next_bundle
                # interact with retriever, get and delete retrieved_cases for cur_bundle
                #   retrieved_cases_cur_bundle: [encoded_cases_gene, encoded_cases_retr]
                #       encoded_cases_gene: [doc_gene_input_ids, doc_gene_attention_mask, doc_gene_lm_labels]
                #       encoded_cases_retr: [doc_retr_cases_input_ids, doc_retr_cases_attention_mask, doc_retr_cases_segment_ids]
                #           doc_gene_input_ids: [len_bundle, n_doc, cases_per_doc * input_len_gene]
                #           doc_retr_cases_input_ids: [len_bundle, n_doc, cases_per_doc, input_len_retr]
                # retrieved_cases_cur_bundle = wait_get_remove_cases_for_bundle(path_retrieved_encoded_cases, path_cnt_retrieved_bundle)
                ## only when if_fast_train and if_comet_baseline, we do not need to wait for retrieved_cases_cur_bundle
                if not (args.if_fast_train and args.if_comet_baseline):
                    retrieved_cases_cur_bundle = wait_get_remove_cases_for_bundle_while_deleting_bad_cases_file(args, \
                        path_cnt_retrieved_bundle, 'train',  path_retrieved_encoded_cases_train=path_retrieved_encoded_cases)
                # interact with retriever, not letting retriever wait
                if id_bundle == 0:
                    try:
                        next_bundle = next(itr_train_bundleloader)
                        if not (args.if_fast_train and args.if_comet_baseline):
                            while os.path.exists(path_retriever):
                                print('Warning: prev retriever still exists!!!!!!!!!!!!!!!!!!!!!!!!')
                                time.sleep(5)
                            torch.save(model_retriever.state_dict(), path_retriever)
                            # save model_retriever_doc for retriever.py to use
                            while os.path.exists(path_retriever_doc):
                                print('Warning: prev retriever_doc still exists!!!!!!!!!!!!!!!!!!!!!!!!')
                                time.sleep(5)
                            torch.save(model_retriever_doc.state_dict(), path_retriever_doc)
                        # about next_bundle
                        path_prev_next_bundle, path_cur_next_bundle, cnt_bundle = get_path_cur_next_bundle(args, \
                            path_cnt_saved_bundle, 'train', path_next_bundle_train=path_next_bundle)
                        if not (args.if_fast_train and args.if_comet_baseline):
                            # comment this line only during debugging
                            assert not os.path.exists(path_prev_next_bundle)
                            try:
                                torch.save(next_bundle, path_cur_next_bundle)
                            except:
                                time.sleep(5)
                                print("Exception occurs when saving next_bundle")
                                torch.save(next_bundle, path_cur_next_bundle)
                        cnt_bundle[0] += 1
                        torch.save(cnt_bundle, path_cnt_saved_bundle)
                    except StopIteration:
                        print("StopIteration happens")
                        pass

                if "bart" in args.generator_model_type or "gpt2" in args.generator_model_type or "bert" in args.generator_model_type or "t5" in args.generator_model_type:
                    if args.if_fast_train:
                        case_aug_cur_bundle = concat_cur_bundle_and_encoded_cases_EOSfixed_Bart_randomly_mask_demonstrations(args, cur_bundle, retrieved_cases_cur_bundle, tokenizer_generator, 'train')
                    else:
                        case_aug_cur_bundle = concat_cur_bundle_and_encoded_cases_EOSfixed_Bart(args, cur_bundle, retrieved_cases_cur_bundle, tokenizer_generator)
                else:
                    raise Exception("Not supported generator_model_type: ", args.generator_model_type)

                # get dataloader_in_batch for current bundle
                data_in_batch = TensorDataset(*case_aug_cur_bundle)
                sampler_in_batch = SequentialSampler(data_in_batch)
                dataloader_in_batch = DataLoader(data_in_batch, sampler=sampler_in_batch, batch_size=args.train_batch_size)
                # begin train in batches
                for step, batch in enumerate(dataloader_in_batch):
                    ttl_batch_steps += 1
                    # seq_logprobs: [batch_size, n_doc, tgt_length, #vocab]
                    if args.if_fast_train:
                        loss, nll_loss, seq_logprobs, batch_accuracy = batch_step_fast_train_1GPU(args, model_generator, model_retriever, model_retriever_doc, batch, tokenizer_generator, args.smooth_score, args.if_try_one_gpu)
                        # print("batch_accuracy: ", batch_accuracy)
                    else:
                        # seq_logprobs: [batch_size, n_doc, tgt_length, #vocab]
                        loss, nll_loss, seq_logprobs, doc_logprobs, batch_accuracy = batch_step(args, model_generator, model_retriever, model_retriever_doc, batch, tokenizer_generator, args.smooth_score)
                    if step == 0:
                        print("seq_logprobs.size(): ", seq_logprobs.size())
                    if step % 150 == 0:
                        torch.save(batch, path_example_batch)
                        # torch.save(seq_logprobs.to('cpu'), os.path.join(args.output_dir, 'batch_logits.pt'))
                        # torch.save(doc_logprobs.to('cpu'), os.path.join(args.output_dir, 'doc_logprobs.pt'))
                        # raise Exception
                        #   case_aug_cur_bundle: [case_aug_gene_input_id, case_aug_gene_attention_mask, case_aug_gene_lm_labels, \
                        # doc_retr_cases_input_ids, doc_retr_cases_attention_mask, doc_retr_cases_segment_ids, \
                        # input_retr_input_ids, input_retr_attention_mask, input_retr_segment_ids]
                        # case_aug_gene_input_id: [len_bundle, n_doc, cases_per_doc * input_len_gene + 1 + input_len_gene]
                        _case_aug_gene_input_id, _case_aug_gene_attention_mask, _case_aug_gene_lm_labels = batch[0:3]
                        value, indices = seq_logprobs.max(dim=-1)
                        sample_index = random.randint(0, args.train_batch_size - 1)
                        print("input_id:", tokenizer_generator.decode(_case_aug_gene_input_id[sample_index][0].tolist()))
                        # print("input_mask:", _case_aug_gene_attention_mask[sample_index][0].tolist())
                        # tmp_lm_labels = _case_aug_gene_lm_labels[sample_index][0].tolist()
                        # tmp_lm_labels = [1 if tmp_lm_labels[i] == -100 else tmp_lm_labels[i] for i in range(len(tmp_lm_labels))]
                        # print("input_lm_labels:", tokenizer_generator.decode(tmp_lm_labels))
                        # IMPORTANT: add max_additional_cases
                        if step == 0:
                            print("indices.size(): ", indices.size())
                        if "gpt2" in args.generator_model_type:
                            output = indices[sample_index][0].tolist()[-args.max_e2:]
                            output = tokenizer_generator.decode(output)
                        elif "bart" in args.generator_model_type or "bert" in args.generator_model_type or "t5" in args.generator_model_type:
                            output = indices[sample_index][0].tolist()
                            try:
                                eos_pos = output.index(generator_eos_id)
                                output = tokenizer_generator.decode(output[:eos_pos])
                            except:
                                output = tokenizer_generator.decode(output)
                        else:
                            raise NotImplementError
                        # print("output ids:", output)
                        print("output:", output)

                    if args.gradient_accumulation_steps > 1:
                        loss = loss / args.gradient_accumulation_steps
                        nll_loss = nll_loss / args.gradient_accumulation_steps
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model_generator.parameters(), args.max_grad_norm)
                    torch.nn.utils.clip_grad_norm_(model_retriever.parameters(), args.max_grad_norm)
                    torch.nn.utils.clip_grad_norm_(model_retriever_doc.parameters(), args.max_grad_norm)
                    tr_loss += loss.item()
                    tr_nll_loss += nll_loss.item()
                    # if (step + 1) % args.gradient_accumulation_steps == 0:
                    if (ttl_batch_steps + 1) % args.gradient_accumulation_steps == 0:
                        optimizer.step()
                        scheduler.step()
                        optimizer.zero_grad()
                        global_steps += 1
                        if global_steps % args.logging_steps == 0:
                            loss = (tr_loss - logging_loss)/args.logging_steps
                            nll_loss = (tr_nll_loss - logging_nll_loss)/args.logging_steps
                            PPL = np.exp(nll_loss) if nll_loss < 300 else np.inf
                            print("Step", global_steps, "Training Loss:", loss, "Nll Loss:", nll_loss, "Smooth loss:", loss-nll_loss, "ppl:", PPL, "batch accuracy: ", batch_accuracy)
                            writer.add_scalar('Train Loss', loss, global_step=global_steps)
                            writer.add_scalar('Train PPL', PPL, global_step=global_steps)
                            if args.dataset_selection == 4 or args.dataset_selection == 5 or args.dataset_selection == 6:
                                writer.add_scalar('Train Accuracy', batch_accuracy, global_step=global_steps)
                            logging_loss = tr_loss
                            logging_nll_loss = tr_nll_loss

                        if global_steps % args.eval_per_steps == 0:
                            model_generator.eval()
                            model_retriever.eval()
                            model_retriever_doc.eval()
                            # evaluate
                            eval_loss, eval_accuracy = evaluate(args, model_generator, model_retriever, \
                                model_retriever_doc, tokenizer_generator, eval_dataloader_in_bundle, \
                                path_next_bundle_eval, path_retriever_eval, path_retriever_doc_eval, \
                                path_retrieved_encoded_cases_eval, 'eval', path_cnt_saved_bundle, path_cnt_retrieved_bundle)
                            print("\n\nevaluating\neval loss:", eval_loss, "ppl", np.exp(eval_loss) if eval_loss < 300 else np.inf)
                            writer.add_scalar('Val Loss', eval_loss, global_step=global_steps)
                            writer.add_scalar('Val PPL', np.exp(eval_loss), global_step=global_steps)
                            if args.dataset_selection == 4 or args.dataset_selection == 5 or args.dataset_selection == 6:
                                writer.add_scalar('Val Accuracy', eval_accuracy, global_step=global_steps)
                            # decide to save; the criteria depends on whether the task is a classification task or generation task
                            if args.dataset_selection == 4 or args.dataset_selection == 5 or args.dataset_selection == 6:
                                whether_best_loss = True if eval_accuracy > best_accuracy else False
                            elif args.dataset_selection == 0 or args.dataset_selection == 1 or args.dataset_selection == 2 or args.dataset_selection == 3:
                                whether_best_loss = True if eval_loss < best_loss else False
                            else:
                                raise NotImplementedError
                            if whether_best_loss:
                                # save models
                                torch.save(model_retriever.state_dict(), path_retriever_final)
                                torch.save(model_retriever_doc.state_dict(), path_retriever_doc_final)
                                torch.save(model_generator.state_dict(), path_generator_final)
                                # save_model(model_generator, tokenizer, args.output_dir)
                                print("model saved at step", global_steps)
                                print(str(datetime.datetime.now()))
                                patience = args.patience
                                if args.dataset_selection == 4 or args.dataset_selection == 5 or args.dataset_selection == 6:
                                    print("prev accuracy:", best_accuracy, "cur loss:", eval_accuracy)
                                    best_accuracy = eval_accuracy
                                elif args.dataset_selection == 0 or args.dataset_selection == 1 or args.dataset_selection == 2 or args.dataset_selection == 3:
                                    print("prev loss:", best_loss, "cur loss:", eval_loss)
                                    best_loss = eval_loss
                                else:
                                    raise NotImplementedError
                            else:
                                # early stopping
                                patience -= 1
                                print("patience: ", patience)
                                if patience < 0:
                                    break
                            # test
                            # test_loss, test_accuracy = evaluate(model, args.generator_model_type, args.predict_part, test_dataloader, tokenizer, max_e1, max_r, max_e2, max_additional_cases, args.add_prefix)
                            # print("\n\ntesting\ntest loss:", test_loss, "ppl:", np.exp(test_loss) if test_loss < 300 else np.inf)
                            model_generator.train()
                            model_retriever.train()
                            model_retriever_doc.train()
                cur_bundle = next_bundle
        torch.save(torch.ones(1,1), path_if_finished_training)
        # send if_finish signal to retriever
        if not args.do_test:
            torch.save(torch.ones(1,1), path_if_no_need_for_retrieval)
        if patience < 0:
            print("Early breaking happens!")


    if args.do_test:
        # Begin testing
        model_generator.load_state_dict(torch.load(path_generator_final, map_location='cuda:0'))
        model_generator.eval()
        if args.if_try_one_gpu:
            model_retriever.load_state_dict(torch.load(path_retriever_final, map_location='cpu'))
            model_retriever_doc.load_state_dict(torch.load(path_retriever_doc_final, map_location='cpu'))
        else:
            model_retriever.load_state_dict(torch.load(path_retriever_final, map_location='cuda:1'))
            model_retriever_doc.load_state_dict(torch.load(path_retriever_doc_final, map_location='cuda:1'))
        model_retriever.eval()
        model_retriever_doc.eval()
        # evaluate
        test_loss, test_accuracy = evaluate(args, model_generator, model_retriever, model_retriever_doc, \
            tokenizer_generator, test_dataloader_in_bundle, path_next_bundle_test, \
            path_retriever_test, path_retriever_doc_test, path_retrieved_encoded_cases_test, \
            'test', path_cnt_saved_bundle, path_cnt_retrieved_bundle)
        print("\n\ntesting\ntest loss:", test_loss, "ppl:", np.exp(test_loss) if test_loss < 300 else np.inf)
        # send if_finish signal to retriever
        torch.save(torch.ones(1,1), path_if_no_need_for_retrieval)

if __name__ == "__main__":
    main()
