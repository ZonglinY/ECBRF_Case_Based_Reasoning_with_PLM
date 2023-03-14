import argparse, logging, os, sys, random, datetime, math
os.environ["CUDA_VISIBLE_DEVICES"]="1"
import numpy as np
import torch
import math
import time
import random
import copy
import pickle

sys.path.insert(0, "..")
# from transformers import (CONFIG_NAME, WEIGHTS_NAME, AdamW,
#                                   get_linear_schedule_with_warmup, cached_path)
# from transformers import (BertModel, BertTokenizer, BertConfig)
from transformers import (DPRQuestionEncoder, DPRContextEncoder,
                        DPRQuestionEncoderTokenizer, DPRContextEncoderTokenizer)
from torch.utils.data import (DataLoader, SequentialSampler,
                              TensorDataset)
# from tqdm import tqdm
# import torch.nn.functional as F
from utils_TST import (save_model, tokenize_and_encode,
                            set_seed, load_data_atomic, find_path_tensor_dataset,
                            find_path_sample_ckb_dict, get_k_for_topk_subRel,
                            get_selected_idxs_and_prob_from_double_topk_result,
                            get_id_for_id_with_different_source_during_larger_range,
                            add_special_tokens)

logging.basicConfig(format = "%(asctime)s - %(levelname)s - %(name)s - %(message)s",
                    datefmt = "%m/%d/%Y %H:%M:%S",
                    level = logging.INFO)
logger = logging.getLogger(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# retr_input_ids, retr_attention_mask, retr_segment_ids = batch
def batch_get_embed(model, model_type, batch):
    batch = tuple(t.to(device) for t in batch)
    retr_input_ids, retr_attention_mask, retr_segment_ids = batch
    if model_type =='dpr':
        outputs = model(retr_input_ids, attention_mask=retr_attention_mask, token_type_ids=retr_segment_ids)
        # # final_hid_embed: [batch_size, (max)512, 768]
        # final_hid_embed = outputs[0]
        # # pooled_embedding: [batch_size, 768]
        # pooled_embedding = final_hid_embed[:, 0, :]

        # # pooled_embedding: [batch_size, 768]
        # pooled_embedding = outputs[1]
        # pooled_embedding: [32, 768], verified
        pooled_embedding = outputs[0]
        # print('pooled_embedding.size(): ', pooled_embedding.size())

    else:
        raise Exception
    return pooled_embedding


# input:
#   sample_size_each_rel: minimum size of sampled case for each rel
#   sample_size_total: size of sampled case knowledge base
#   num_cases: number of cases for each obj_emb to retrieve
#   simi_batch_size: batch size used for calculate similarity during finding cases
#   num_steps: number of steps during training to build cases in advance
class embedding_builder:
    def __init__(self, args, model_type, num_cases=15, n_doc=3, simi_batch_size=64, sample_size_each_rel=100, sample_size_total=20000, num_steps=500, data_type='train'):
        self.args = args
        self.num_cases = num_cases
        self.n_doc = n_doc
        self.simi_batch_size = simi_batch_size
        self.sample_size_each_rel = sample_size_each_rel
        self.sample_size_total = sample_size_total
        self.num_steps = num_steps
        self.data_type = data_type
        # self.path_sample_ckb_dict = path_sample_ckb_dict
        self.initialize_paths_variables(args.output_dir)
        self.check_input_variables()
        # train_lines should be sorted with rel
        if args.dataset_selection == 0:
            self.get_ori_text_conceptnet()
        elif args.dataset_selection == 1:
            self.get_ori_text_atomic()
        elif args.dataset_selection == 2:
            self.get_ori_text_shakespeare()
        elif args.dataset_selection == 3:
            self.get_ori_text_e2e()
        elif args.dataset_selection == 4:
            self.get_ori_text_sentiment()

        self.tensor_datasets = self.wait_and_get_tensor_datasets()
        # self.get_data_loader()
        # self.train_tensor_dataset, self.eval_tensor_dataset, self.test_tensor_dataset = self.tensor_datasets[0], self.tensor_datasets[1], self.tensor_datasets[2]
        self.train_tensor_dataset = self.tensor_datasets[0]

        if self.args.if_double_retrieval:
            self.model, self.model_doc, self.tokenizer_retriever_doc = self.initialize_model(model_type)
            print("self.tokenizer_retriever_doc has been initialized")
        else:
            self.model, self.model_doc = self.initialize_model(model_type)

    def check_input_variables(self):
        assert self.num_cases % self.n_doc == 0
        # Eval need to use this path to load sample_ckb_dict
        assert self.path_sample_ckb_dict != None

    def initialize_paths_variables(self, output_dir):
        # will get it during the inference
        self.n_shot = None
        # data dir
        self.train_data_dir = self.args.train_dataset[0]
        self.val1_data_dir = self.args.eval_dataset[0]
        self.test_data_dir = self.args.test_dataset[0]
        # full embedding; used for evaluation
        self.full_train_emb_dir = os.path.join(output_dir, 'full_train_embedding.pt')
        self.full_val_emb_dir = os.path.join(output_dir, 'full_val_embedding.pt')
        self.full_test_emb_dir = os.path.join(output_dir, 'full_test_embedding.pt')
        # full cases; used for evaluation
        self.full_train_cases_dir = os.path.join(output_dir, 'full_train_cases.txt')
        self.full_val_cases_dir = os.path.join(output_dir, 'full_val_cases.txt')
        self.full_test_cases_dir = os.path.join(output_dir, 'full_test_cases.txt')
        # sample embedding
        self.sample_CKB_train_embed_dir = os.path.join(output_dir, 'sample_CKB_train_embedding.pt')
        self.batch_train_embed_dir = os.path.join(output_dir, 'batch_train_embed.pt')
        # Q: not used yet
        self.batch_eval_embed_dir = os.path.join(output_dir, 'batch_eval_embed.pt')
        # Q: not used yet
        self.batch_test_embed_dir = os.path.join(output_dir, 'batch_test_embed.pt')
        self.bundle_train_cases_dir = os.path.join(output_dir, 'bundle_train_cases.txt')
        self.bundle_eval_cases_dir = os.path.join(output_dir, 'bundle_eval_cases.txt')
        self.bundle_test_cases_dir = os.path.join(output_dir, 'bundle_test_cases.txt')
        # for debug
        self.batch_rel_dir = os.path.join(output_dir, 'batch_rel.pt')
        # to interact with generator
        self.path_tensor_datasets = find_path_tensor_dataset(self.args)
        print("INFO: path_tensor_datasets: ", self.path_tensor_datasets)
        # 'next_bundle_train' is used in self.wait_get_remove_bundle
        self.path_next_bundle = os.path.join(self.args.output_dir, 'next_bundle_train.pt')
        # 'next_bundle_eval' and 'next_bundle_test' are used in self.wait_get_remove_bundle_eval
        self.path_next_bundle_eval = os.path.join(self.args.output_dir, 'next_bundle_eval.pt')
        self.path_next_bundle_test = os.path.join(self.args.output_dir, 'next_bundle_test.pt')
        self.path_model_retriever = os.path.join(self.args.output_dir, 'retriever.pt')
        self.path_model_retriever_eval = os.path.join(self.args.output_dir, 'retriever_eval.pt')
        self.path_model_retriever_test = os.path.join(self.args.output_dir, 'retriever_test.pt')
        self.path_retriever_doc = os.path.join(self.args.output_dir, 'retriever_doc.pt')
        self.path_retriever_doc_eval = os.path.join(self.args.output_dir, 'retriever_doc_eval.pt')
        self.path_retriever_doc_test = os.path.join(self.args.output_dir, 'retriever_doc_test.pt')
        self.path_retrieved_encoded_cases = os.path.join(self.args.output_dir, 'encoded_cases_train.pt')
        self.path_retrieved_encoded_cases_eval = os.path.join(self.args.output_dir, 'encoded_cases_eval.pt')
        self.path_retrieved_encoded_cases_test = os.path.join(self.args.output_dir, 'encoded_cases_test.pt')
        # signal from generator that indicates whether it's eval or test mode
        self.path_signal_file_if_eval = os.path.join(self.args.output_dir, 'under_evaluation.pt')
        self.path_signal_file_if_test = os.path.join(self.args.output_dir, 'under_evaluation_test.pt')

        # path to save ttl_similar_cases during eval or test time
        self.path_ttl_similar_cases_eval = os.path.join(self.args.output_dir, 'ttl_similar_cases_eval.txt')
        # change the name for a different sample (to differentiate different ttl_similar_cases for different sample)
        if self.args.num_sample != 1:
            self.path_ttl_similar_cases_eval = self.path_ttl_similar_cases_eval.split('.')
            assert len(self.path_ttl_similar_cases_eval) == 2
            self.path_ttl_similar_cases_eval[0] += '_sample' + str(self.args.num_sample)
            self.path_ttl_similar_cases_eval = '.'.join(self.path_ttl_similar_cases_eval)
        # Should be different
        if self.args.if_use_full_memory_store_while_subset:
            self.path_ttl_similar_cases_test = os.path.join(self.args.output_dir, 'ttl_similar_cases_test_fullMS.txt')
        else:
            self.path_ttl_similar_cases_test = os.path.join(self.args.output_dir, 'ttl_similar_cases_test.txt')
        # change the name for a different sample (to differentiate different ttl_similar_cases for different sample)
        if self.args.num_sample != 1:
            self.path_ttl_similar_cases_test = self.path_ttl_similar_cases_test.split('.')
            assert len(self.path_ttl_similar_cases_test) == 2
            self.path_ttl_similar_cases_test[0] += '_sample' + str(self.args.num_sample)
            self.path_ttl_similar_cases_test = '.'.join(self.path_ttl_similar_cases_test)
        self.path_if_no_need_for_retrieval = os.path.join(self.args.output_dir, 'finished-no_need_for_retrieval.pt')
        # to store ttl_similar_cases
        self.ttl_similar_cases_eval, self.ttl_similar_cases_test = [], []
        # when using frozen retriever or comet baseline, sample_ckb_dict is fixed
        self.sample_ckb_dict = None
        # num_cases_tmp = np.minimum(num_cases*self.times_more_num_cases_to_retrieve+1, similarity.size()[1])
        # rlt_topk = torch.topk(similarity, num_cases_tmp)
        if self.args.rerank_selection == 0:
            self.times_more_num_cases_to_retrieve = 1
        elif self.args.rerank_selection == 1 or self.args.rerank_selection == 2:
            if self.n_doc >= 3:
                self.times_more_num_cases_to_retrieve = 3
            else:
                self.times_more_num_cases_to_retrieve = 10
            # Conceptnet needs more times_more_num_cases_to_retrieve, since the number of cases
            # that with the same sub and rel can be large
            if self.args.dataset_selection == 0:
                self.times_more_num_cases_to_retrieve *= 24
            elif self.args.dataset_selection == 3:
                self.times_more_num_cases_to_retrieve *= 2
            if self.args.use_only_sub_rel_for_retrieval:
                self.times_more_num_cases_to_retrieve *= 3
            # newly added 2023/03/14; since with less num_cases, times_more_num_cases_to_retrieve might not be enough
            if self.args.num_cases < 3:
                self.times_more_num_cases_to_retrieve = int(self.times_more_num_cases_to_retrieve * 3 / self.args.num_cases)
        else:
            raise Exception('Invalid rerank_selection')
        # cnt_next_bundle: the number of current next_bundle to retrieve
        # (this number is used by both train/eval/test mode)
        self.cnt_next_bundle = None

        # path_sample_ckb_dict is not influenced by args.use_obj_for_retrieval
        self.path_sample_ckb_dict = find_path_sample_ckb_dict(self.args)
        if self.args.if_double_retrieval:
            self.path_sample_ckb_dict_subRel = find_path_sample_ckb_dict(self.args, use_only_sub_rel_for_retrieval=True)


    # Add tokenizer_gene
    def initialize_model(self, model_type='dpr'):
        # currently only support bert
        assert model_type == 'dpr'
        # print("INFO: tokenizer_gene uses GPT2Tokenizer to initialize")
        # tokenizer_gene = GPT2Tokenizer.from_pretrained('gpt2')

        if self.args.if_double_retrieval:
            # tokenizer_retriever = DPRQuestionEncoderTokenizer.from_pretrained('facebook/dpr-question_encoder-single-nq-base')
            tokenizer_retriever_doc = DPRContextEncoderTokenizer.from_pretrained('facebook/dpr-ctx_encoder-single-nq-base')
            # tokenizer_retriever = add_special_tokens(self.args, tokenizer_retriever)
            tokenizer_retriever_doc = add_special_tokens(self.args, tokenizer_retriever_doc)

        model = DPRQuestionEncoder.from_pretrained('facebook/dpr-question_encoder-single-nq-base')
        model_doc = DPRContextEncoder.from_pretrained('facebook/dpr-ctx_encoder-single-nq-base')

        # model.resize_token_embeddings(len(tokenizer_retriever))
        model.to(device)
        # model_doc.resize_token_embeddings(len(tokenizer_retriever_doc))
        model_doc.to(device)
        if self.args.if_double_retrieval:
            return model, model_doc, tokenizer_retriever_doc
        else:
            return model, model_doc

    def wait_and_get_tensor_datasets(self):
        # if_longer_wait_time: if self.path_tensor_datasets is just created; give it more time to create a complete file
        if_longer_wait_time = False
        while not os.path.exists(self.path_tensor_datasets):
            time.sleep(5)
            if_longer_wait_time = True
        # do not need to remove since it won't be changed during experiment
        if if_longer_wait_time:
            time.sleep(10)
        while True:
            try:
                tensor_datasets = torch.load(self.path_tensor_datasets)
                break
            except:
                time.sleep(5)
        print('Loaded tensor_datasets Successfully.')
        return tensor_datasets

    # get self.next_bundle
    def wait_get_remove_bundle(self):
        print('Waiting for next_bundle...')
        start_waiting_time = time.time()
        # while(not os.path.exists(RFself.path_next_bundle)):
        while True:
            if os.path.exists(self.path_signal_file_if_eval):
                print("Begin evaluating in validation set...")
                self.evaluation_mode('eval')
                assert not os.path.exists(self.path_signal_file_if_eval)
            elif os.path.exists(self.path_signal_file_if_test):
                print("Begin evaluating in test set...")
                self.evaluation_mode('test')
                assert not os.path.exists(self.path_signal_file_if_test)
            elif os.path.exists(self.path_if_no_need_for_retrieval):
                # if continue
                return False
            else:
                possible_other_next_bundle_files = \
                    [i for i in os.listdir(self.args.output_dir) if i.startswith('next_bundle_train')]
                if len(possible_other_next_bundle_files) > 0:
                    assert len(possible_other_next_bundle_files) == 1
                    # to get self.cnt_next_bundle
                    self.cnt_next_bundle = int(possible_other_next_bundle_files[0].split('.')[0].replace('next_bundle_train_', ''))
                    # to get the exact path_next_bundle
                    path_next_bundle = os.path.join(self.args.output_dir, possible_other_next_bundle_files[0])
                    break
                else:
                    time.sleep(5)

        print('--- Waiting for next_bundle: %s ---' % (time.time() - start_waiting_time))
        time.sleep(5)
        while True:
            try:
                self.next_bundle = torch.load(path_next_bundle)
                break
            except:
                time.sleep(5)
        os.remove(path_next_bundle)
        assert not os.path.exists(path_next_bundle)
        # print('Loaded and deleted next_bundle successfully!')
        # if continue
        return True

    # both model_retriever and model_retriever_doc
    def load_remove_retriever(self, data_type='train'):
        if data_type == 'train':
            path_retriever = self.path_model_retriever
            path_retriever_doc = self.path_retriever_doc
        elif data_type == 'eval':
            path_retriever = self.path_model_retriever_eval
            path_retriever_doc = self.path_retriever_doc_eval
        elif data_type == 'test':
            path_retriever = self.path_model_retriever_test
            path_retriever_doc = self.path_retriever_doc_test
        else:
            raise Exception("Wrong data_type: ", data_type)
        # should be saved within 15 sec
        starting_wait_time = time.time()
        while not os.path.exists(path_retriever) or not os.path.exists(path_retriever_doc):
            print('Warning: waiting for model_retriever or path_retriever_doc')
            time.sleep(5)
        waiting_time_for_retriever = time.time() - starting_wait_time
        if waiting_time_for_retriever > 20:
            print('Warning: waiting_time_for_retriever > 20s')
        # Q: should we check whether loaded retriever has added special tokens?
        # model_retriever
        time.sleep(5)
        self.model.load_state_dict(torch.load(path_retriever, map_location='cuda:0'))
        # model_retriever_doc
        self.model_doc.load_state_dict(torch.load(path_retriever_doc, map_location='cuda:0'))
        self.model.eval()
        self.model_doc.eval()
        os.remove(path_retriever)
        os.remove(path_retriever_doc)
        assert not os.path.exists(path_retriever)
        assert not os.path.exists(path_retriever_doc)
        # print('Loaded and deleted path_retriever successfully!')

    # get_sampled_ckb_data_loader accoring to sample_ids
    # sample_ids: a tensor contains sampled ids
    def get_sampled_ckb_data_loader(self, sample_ids):
        # Q: need to extend for eval and test
        # SQ: seems do not need extension
        assert self.data_type == 'train'
        sampled_train_tensor_dataset = [d[sample_ids] for d in self.train_tensor_dataset]
        train_data = TensorDataset(*sampled_train_tensor_dataset)
        assert len(train_data) == len(sample_ids)
        train_sampler = SequentialSampler(train_data)
        sample_train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=self.args.train_batch_size)
        return sample_train_dataloader

    def get_ori_text_atomic(self):
        # eval_size: 1000 means using "None" tuple data; 900 means not using
        # test_size: 87481 means using "None" tuple data; 72872 means not using
        if self.args.if_without_none:
            eval_size = 900
            test_size = 72872
        else:
            eval_size = 1000
            test_size = 87481
        if self.args.subset_selection == -1:
            # if self.args.if_without_none == True, the returned lines should not contain "None"
            # [('PersonX plays a ___ in the war', '<oReact>', 'none'), (), ...]
            self.train_lines, self.cat_len_noter_train = load_data_atomic(self.train_data_dir, self.args.if_without_none)
            self.val_lines, self.cat_len_noter_val = load_data_atomic(self.val1_data_dir, self.args.if_without_none)
            self.test_lines, self.cat_len_noter_test = load_data_atomic(self.test_data_dir, self.args.if_without_none)
        else:
            # Q:
            data_dir = "./Data/atomic/"
            with open(os.path.join(data_dir, 'train'+'_subset_'+str(self.args.subset_selection)+'_lines' + self.args.additional_sampling_method_name + self.args.additional_sample_name), 'rb') as f:
                self.train_lines = pickle.load(f)
            with open(os.path.join(data_dir, 'eval'+'_subset_'+str(eval_size)+'_lines'), 'rb') as f:
                self.val_lines = pickle.load(f)
            with open(os.path.join(data_dir, 'test'+'_subset_'+str(test_size)+'_lines'), 'rb') as f:
                self.test_lines = pickle.load(f)

            with open(os.path.join(data_dir, 'train'+'_subset_'+str(self.args.subset_selection)+'_cat_len_noter' + self.args.additional_sampling_method_name + self.args.additional_sample_name), 'rb') as f:
                self.cat_len_noter_train = pickle.load(f)
            with open(os.path.join(data_dir, 'eval'+'_subset_'+str(eval_size)+'_cat_len_noter'), 'rb') as f:
                self.cat_len_noter_val = pickle.load(f)
            with open(os.path.join(data_dir, 'test'+'_subset_'+str(test_size)+'_cat_len_noter'), 'rb') as f:
                self.cat_len_noter_test = pickle.load(f)

        if self.args.if_use_full_memory_store_while_subset:
            data_dir = "./Data/atomic/"
            with open(os.path.join(data_dir, 'train'+'_subset_'+str(self.args.subset_selection_while_use_full_memory_store)+'_lines' + self.args.additional_sampling_method_name + self.args.additional_sample_name), 'rb') as f:
                self.train_subset_lines = pickle.load(f)
            with open(os.path.join(data_dir, 'eval'+'_subset_'+str(eval_size)+'_lines'), 'rb') as f:
                self.val_subset_lines = pickle.load(f)
            with open(os.path.join(data_dir, 'test'+'_subset_'+str(test_size)+'_lines'), 'rb') as f:
                self.test_subset_lines = pickle.load(f)

            with open(os.path.join(data_dir, 'train'+'_subset_'+str(self.args.subset_selection_while_use_full_memory_store)+'_cat_len_noter' + self.args.additional_sampling_method_name + self.args.additional_sample_name), 'rb') as f:
                self.cat_len_noter_subset_train = pickle.load(f)
            with open(os.path.join(data_dir, 'eval'+'_subset_'+str(eval_size)+'_cat_len_noter'), 'rb') as f:
                self.cat_len_noter_subset_val = pickle.load(f)
            with open(os.path.join(data_dir, 'test'+'_subset_'+str(test_size)+'_cat_len_noter'), 'rb') as f:
                self.cat_len_noter_subset_test = pickle.load(f)


    def get_cat_len_noter_conceptnet(self, lines):
        cat_len_noter = []
        prev_rel, cur_rel = None, None
        for id, line in enumerate(lines):
            cur_rel = line.split('\t')[0]
            if cur_rel != prev_rel and prev_rel != None:
                cat_len_noter.append(id-1)
            prev_rel = cur_rel
        cat_len_noter.append(id)
        return cat_len_noter

    # train_lines/val_lines/test_lines must be sorted
    def get_ori_text_conceptnet(self):
        if self.args.subset_selection == -1:
            with open(self.train_data_dir, 'r') as f:
                self.train_lines = f.readlines()
                self.cat_len_noter_train = self.get_cat_len_noter_conceptnet(self.train_lines)
                print('cat_len_noter_train: ', self.cat_len_noter_train)
                print('len(cat_len_noter_train): ', len(self.cat_len_noter_train))

            with open(self.val1_data_dir, 'r') as f:
                self.val_lines = f.readlines()
                self.cat_len_noter_val = self.get_cat_len_noter_conceptnet(self.val_lines)
                print('cat_len_noter_val: ', self.cat_len_noter_val)
                print('len(cat_len_noter_val): ', len(self.cat_len_noter_val))

            with open(self.test_data_dir, 'r') as f:
                self.test_lines = f.readlines()
                self.cat_len_noter_test = self.get_cat_len_noter_conceptnet(self.test_lines)
                print('cat_len_noter_test: ', self.cat_len_noter_test)
                print('len(cat_len_noter_test): ', len(self.cat_len_noter_test))
        else:
            data_dir = "./Data/conceptnet/"
            with open(os.path.join(data_dir, 'train'+'_subset_'+str(self.args.subset_selection)+'_lines' + self.args.additional_sampling_method_name + self.args.additional_sample_name), 'rb') as f:
                self.train_lines = pickle.load(f)
            with open(os.path.join(data_dir, 'eval'+'_subset_'+str(600)+'_lines'), 'rb') as f:
                self.val_lines = pickle.load(f)
            with open(os.path.join(data_dir, 'test'+'_subset_'+str(1200)+'_lines'), 'rb') as f:
                self.test_lines = pickle.load(f)

            with open(os.path.join(data_dir, 'train'+'_subset_'+str(self.args.subset_selection)+'_cat_len_noter' + self.args.additional_sampling_method_name + self.args.additional_sample_name), 'rb') as f:
                self.cat_len_noter_train = pickle.load(f)
            with open(os.path.join(data_dir, 'eval'+'_subset_'+str(600)+'_cat_len_noter'), 'rb') as f:
                self.cat_len_noter_val = pickle.load(f)
            with open(os.path.join(data_dir, 'test'+'_subset_'+str(1200)+'_cat_len_noter'), 'rb') as f:
                self.cat_len_noter_test = pickle.load(f)

        if self.args.if_use_full_memory_store_while_subset:
            data_dir = "./Data/conceptnet/"
            with open(os.path.join(data_dir, 'train'+'_subset_'+str(self.args.subset_selection_while_use_full_memory_store)+'_lines' + self.args.additional_sampling_method_name + self.args.additional_sample_name), 'rb') as f:
                self.train_subset_lines = pickle.load(f)
            with open(os.path.join(data_dir, 'eval'+'_subset_'+str(600)+'_lines'), 'rb') as f:
                self.val_subset_lines = pickle.load(f)
            with open(os.path.join(data_dir, 'test'+'_subset_'+str(1200)+'_lines'), 'rb') as f:
                self.test_subset_lines = pickle.load(f)

            with open(os.path.join(data_dir, 'train'+'_subset_'+str(self.args.subset_selection_while_use_full_memory_store)+'_cat_len_noter' + self.args.additional_sampling_method_name + self.args.additional_sample_name), 'rb') as f:
                self.cat_len_noter_subset_train = pickle.load(f)
            with open(os.path.join(data_dir, 'eval'+'_subset_'+str(600)+'_cat_len_noter'), 'rb') as f:
                self.cat_len_noter_subset_val = pickle.load(f)
            with open(os.path.join(data_dir, 'test'+'_subset_'+str(1200)+'_cat_len_noter'), 'rb') as f:
                self.cat_len_noter_subset_test = pickle.load(f)

    def get_ori_text_shakespeare(self):
        if self.args.subset_selection == -1:
            while not os.path.exists(self.train_data_dir):
                time.sleep(5)
            with open(self.train_data_dir, 'r') as f:
                self.train_lines = f.readlines()
                self.cat_len_noter_train = self.get_cat_len_noter_conceptnet(self.train_lines)
                print('cat_len_noter_train: ', self.cat_len_noter_train)
                print('len(cat_len_noter_train): ', len(self.cat_len_noter_train))

            with open(self.val1_data_dir, 'r') as f:
                self.val_lines = f.readlines()
                self.cat_len_noter_val = self.get_cat_len_noter_conceptnet(self.val_lines)
                print('cat_len_noter_val: ', self.cat_len_noter_val)
                print('len(cat_len_noter_val): ', len(self.cat_len_noter_val))

            with open(self.test_data_dir, 'r') as f:
                self.test_lines = f.readlines()
                self.cat_len_noter_test = self.get_cat_len_noter_conceptnet(self.test_lines)
                print('cat_len_noter_test: ', self.cat_len_noter_test)
                print('len(cat_len_noter_test): ', len(self.cat_len_noter_test))
        else:
            data_dir = "./Data/shakes/"
            with open(os.path.join(data_dir, 'train'+'_subset_'+str(self.args.subset_selection)+'_lines' + self.args.additional_sample_name), 'rb') as f:
                self.train_lines = pickle.load(f)
            with open(os.path.join(data_dir, 'eval'+'_subset_'+str(1179)+'_lines'), 'rb') as f:
                self.val_lines = pickle.load(f)
            with open(os.path.join(data_dir, 'test'+'_subset_'+str(1411)+'_lines'), 'rb') as f:
                self.test_lines = pickle.load(f)

            with open(os.path.join(data_dir, 'train'+'_subset_'+str(self.args.subset_selection)+'_cat_len_noter' + self.args.additional_sample_name), 'rb') as f:
                self.cat_len_noter_train = pickle.load(f)
            with open(os.path.join(data_dir, 'eval'+'_subset_'+str(1179)+'_cat_len_noter'), 'rb') as f:
                self.cat_len_noter_val = pickle.load(f)
            with open(os.path.join(data_dir, 'test'+'_subset_'+str(1411)+'_cat_len_noter'), 'rb') as f:
                self.cat_len_noter_test = pickle.load(f)

        if self.args.if_use_full_memory_store_while_subset:
            data_dir = "./Data/shakes/"
            with open(os.path.join(data_dir, 'train'+'_subset_'+str(self.args.subset_selection_while_use_full_memory_store)+'_lines' + self.args.additional_sample_name), 'rb') as f:
                self.train_subset_lines = pickle.load(f)
            with open(os.path.join(data_dir, 'eval'+'_subset_'+str(1179)+'_lines'), 'rb') as f:
                self.val_subset_lines = pickle.load(f)
            with open(os.path.join(data_dir, 'test'+'_subset_'+str(1411)+'_lines'), 'rb') as f:
                self.test_subset_lines = pickle.load(f)

            with open(os.path.join(data_dir, 'train'+'_subset_'+str(self.args.subset_selection_while_use_full_memory_store)+'_cat_len_noter' + self.args.additional_sample_name), 'rb') as f:
                self.cat_len_noter_subset_train = pickle.load(f)
            with open(os.path.join(data_dir, 'eval'+'_subset_'+str(1179)+'_cat_len_noter'), 'rb') as f:
                self.cat_len_noter_subset_val = pickle.load(f)
            with open(os.path.join(data_dir, 'test'+'_subset_'+str(1411)+'_cat_len_noter'), 'rb') as f:
                self.cat_len_noter_subset_test = pickle.load(f)

    def get_ori_text_e2e(self):
        if self.args.subset_selection == -1:
            while not os.path.exists(self.train_data_dir):
                time.sleep(5)
            with open(self.train_data_dir, 'r') as f:
                self.train_lines = f.readlines()
                self.cat_len_noter_train = self.get_cat_len_noter_conceptnet(self.train_lines)
                print('cat_len_noter_train: ', self.cat_len_noter_train)
                print('len(cat_len_noter_train): ', len(self.cat_len_noter_train))

            with open(self.val1_data_dir, 'r') as f:
                self.val_lines = f.readlines()
                self.cat_len_noter_val = self.get_cat_len_noter_conceptnet(self.val_lines)
                print('cat_len_noter_val: ', self.cat_len_noter_val)
                print('len(cat_len_noter_val): ', len(self.cat_len_noter_val))

            with open(self.test_data_dir, 'r') as f:
                self.test_lines = f.readlines()
                self.cat_len_noter_test = self.get_cat_len_noter_conceptnet(self.test_lines)
                print('cat_len_noter_test: ', self.cat_len_noter_test)
                print('len(cat_len_noter_test): ', len(self.cat_len_noter_test))
        else:
            data_dir = "./Data/e2e/"
            with open(os.path.join(data_dir, 'train'+'_subset_'+str(self.args.subset_selection)+'_lines' + self.args.additional_sample_name), 'rb') as f:
                self.train_lines = pickle.load(f)
            with open(os.path.join(data_dir, 'eval'+'_subset_'+str(1000)+'_lines'), 'rb') as f:
                self.val_lines = pickle.load(f)
            with open(os.path.join(data_dir, 'test'+'_subset_'+str(4693)+'_lines'), 'rb') as f:
                self.test_lines = pickle.load(f)

            with open(os.path.join(data_dir, 'train'+'_subset_'+str(self.args.subset_selection)+'_cat_len_noter' + self.args.additional_sample_name), 'rb') as f:
                self.cat_len_noter_train = pickle.load(f)
            with open(os.path.join(data_dir, 'eval'+'_subset_'+str(1000)+'_cat_len_noter'), 'rb') as f:
                self.cat_len_noter_val = pickle.load(f)
            with open(os.path.join(data_dir, 'test'+'_subset_'+str(4693)+'_cat_len_noter'), 'rb') as f:
                self.cat_len_noter_test = pickle.load(f)

        if self.args.if_use_full_memory_store_while_subset:
            data_dir = "./Data/e2e/"
            with open(os.path.join(data_dir, 'train'+'_subset_'+str(self.args.subset_selection_while_use_full_memory_store)+'_lines' + self.args.additional_sample_name), 'rb') as f:
                self.train_subset_lines = pickle.load(f)
            with open(os.path.join(data_dir, 'eval'+'_subset_'+str(1000)+'_lines'), 'rb') as f:
                self.val_subset_lines = pickle.load(f)
            with open(os.path.join(data_dir, 'test'+'_subset_'+str(4693)+'_lines'), 'rb') as f:
                self.test_subset_lines = pickle.load(f)

            with open(os.path.join(data_dir, 'train'+'_subset_'+str(self.args.subset_selection_while_use_full_memory_store)+'_cat_len_noter' + self.args.additional_sample_name), 'rb') as f:
                self.cat_len_noter_subset_train = pickle.load(f)
            with open(os.path.join(data_dir, 'eval'+'_subset_'+str(1000)+'_cat_len_noter'), 'rb') as f:
                self.cat_len_noter_subset_val = pickle.load(f)
            with open(os.path.join(data_dir, 'test'+'_subset_'+str(4693)+'_cat_len_noter'), 'rb') as f:
                self.cat_len_noter_subset_test = pickle.load(f)


    def get_ori_text_sentiment(self):
        while not os.path.exists(self.train_data_dir):
            time.sleep(5)
        data_dir = "./Data/sentiment/splitted/"
        with open(self.train_data_dir, 'r') as f:
            if self.args.subset_selection == -1:
                self.train_lines = f.readlines()
            else:
                all_lines = f.readlines()
                with open(os.path.join(data_dir, "{}_subset_{}_index.npy".format("train", self.args.subset_selection)), "rb") as f:
                    subset_indexes = np.load(f)
                subset_indexes = sorted(subset_indexes)
                self.train_lines = [all_lines[index] for index in subset_indexes]
            self.cat_len_noter_train = self.get_cat_len_noter_conceptnet(self.train_lines)
            print('cat_len_noter_train: ', self.cat_len_noter_train)
            print('len(cat_len_noter_train): ', len(self.cat_len_noter_train))

        with open(self.val1_data_dir, 'r') as f:
            self.val_lines = f.readlines()
            self.cat_len_noter_val = self.get_cat_len_noter_conceptnet(self.val_lines)
            print('cat_len_noter_val: ', self.cat_len_noter_val)
            print('len(cat_len_noter_val): ', len(self.cat_len_noter_val))

        with open(self.test_data_dir, 'r') as f:
            self.test_lines = f.readlines()
            self.cat_len_noter_test = self.get_cat_len_noter_conceptnet(self.test_lines)
            print('cat_len_noter_test: ', self.cat_len_noter_test)
            print('len(cat_len_noter_test): ', len(self.cat_len_noter_test))

        if self.args.if_use_full_memory_store_while_subset:
            raise NotImplementedError


    # get embed for train batches
    # input:
    # self.next_bundle: #train(gene_doc_input_ids, gene_doc_attention_mask, gene_doc_lm_labels, \
                            #  gene_cur_input_ids, gene_cur_attention_mask, gene_cur_lm_labels, \
    #                          data_idx_ids, rel_collection, \
    #                          retr_doc_input_ids, retr_doc_attention_mask, retr_doc_segment_ids, \
    #                          retr_input_ids, retr_attention_mask, retr_segment_ids)
    # output: embedding, rel_list(should be sorted), line_id, ori_id_in_next_bundle(to recover order)
    def get_embed_for_train_batches(self, model_type=None, save_dir=None, next_bundle=None):
        self.model.eval()
        len_next_bundle = next_bundle[0].size()[0]
        id_next_bundle = torch.tensor(list(range(len_next_bundle)))

        reorder_to_sort_rel = torch.argsort(next_bundle[-7])
        # now next_bundle has ordered rel_list
        next_bundle = [d[reorder_to_sort_rel] for d in next_bundle]
        # to recover the ori batch id
        ori_id_in_next_bundle = id_next_bundle[reorder_to_sort_rel]

        builded_embedding = None
        num_steps = math.ceil(len_next_bundle / self.args.train_batch_size)
        for step in range(num_steps):
            batch = [d[step*self.args.train_batch_size: np.minimum((step+1)*self.args.train_batch_size, len_next_bundle)] for d in next_bundle]
            # batch: (retr_input_ids, retr_attention_mask, retr_segment_ids)
            batch = tuple(t.to(device) for t in batch[-3:])
            batch_size = batch[0].size()[0]
            # DEBUG
            # if step == 0 and self.data_type == 'train':
            #     print('batch_size: ', batch_size)
            #     print(torch.cuda.memory_summary(device))
            with torch.no_grad():
                # embedding: [batch_size, 768]
                embedding = batch_get_embed(self.model, model_type, batch)
                if step == 0:
                    builded_embedding = embedding
                else:
                    builded_embedding = torch.cat((builded_embedding, embedding), 0)
        # rel_for_embedding: torch.Size([709996])
        rel_for_embedding = next_bundle[-7]
        line_id_for_embedding = next_bundle[-8]
        assert rel_for_embedding.size()[0] == builded_embedding.size()[0]
        # save
        if save_dir:
            torch.save([builded_embedding.cpu(), rel_for_embedding, ori_id_in_next_bundle], save_dir)
        return builded_embedding, rel_for_embedding, line_id_for_embedding, ori_id_in_next_bundle

    ## Get a sample id and use it to get embed for sample CKB, then build a dict for the embed
    # INPUT:
    # use_only_sub_rel_for_retrieval: for if_double_retrieval, different from args.use_only_sub_rel_for_retrieval
    # OUTPUT:
    # sample_ckb_dict: dict for builded_embedding of sample CKB
    def get_sample_caseKB_embed_dict(self, data_type='train', use_only_sub_rel_for_retrieval=False):
        if data_type == 'train':
            ## get sample_ids; each rel has sample_size_each_rel samples
            if self.args.if_only_embed_ckb_once:
                sample_ids = torch.tensor(list(range(len(self.train_lines))))
                print('INFO: Using full train instances as memory store during training...{}'.format(len(self.train_lines)))
            else:
                sample_ids = self.get_sample_ids(data_type=data_type)
                print('len(sample_ids): ', len(sample_ids))
        elif data_type == 'eval' or data_type == 'test':
            # Q: just for debug
            # sample_ids = self.get_sample_ids()
            # print('Using sampled train instances as memory store during evaling or testing, len(sample_ids):', len(sample_ids))
            sample_ids = torch.tensor(list(range(len(self.train_lines))))
            print('INFO: Using full train instances as memory store during evaling or testing...')
        else:
            raise Exception("Wrong data_type: ", data_type)

        # print('sample_ids.size()', sample_ids.size())
        ## get sampled train embeddings
        # sample_ckb_rel_for_embed: sorted if data in train_dir is sorted
        # sample_ckb_rel_for_embed: id for rel
        # sample_ckb_line_id_for_embed: line id
        sample_ckb_train_embed, sample_ckb_rel_for_embed, sample_ckb_line_id_for_embed = self.get_embed_for_sample_CKB(sample_ids=sample_ids, use_only_sub_rel_for_retrieval=use_only_sub_rel_for_retrieval)
        # print('sample_ckb_train_embed:', sample_ckb_train_embed.size())
        print('sample_ckb_rel_for_embed:', sample_ckb_rel_for_embed)
        # print('sample_ckb_line_id_for_embed:', sample_ckb_line_id_for_embed.size())
        ## get sample_ckb_dict
        sample_ckb_dict = self.get_sample_ckb_dict(sample_ckb_train_embed=sample_ckb_train_embed, sample_ckb_rel_for_embed=sample_ckb_rel_for_embed, sample_ckb_line_id_for_embed=sample_ckb_line_id_for_embed)
        return sample_ckb_dict

    # OUTPUT:
    # sample_ckb_dict: dict of embed for sample ckb
    # bundle_embed_rlts: embed& for bundle
    # rel_ttl_batch: used for check correctness of result
    def get_embed_for_sample_ckb_and_bundle(self):
        ## get dict of embedding of sample_ckb
        sample_ckb_dict = self.get_sample_caseKB_embed_dict()
        # print('CaseKB embedded!')
        ## get embedding of bundle
        # rel_ttl_batch: used for check correctness of result
        bundle_embed_rlts, rel_ttl_batch = self.get_bundle_embed_list(model_type=self.args.retriever_model_type)
        # print('Bundle embedded!')
        return sample_ckb_dict, bundle_embed_rlts, rel_ttl_batch

    ## INPUT:
    # idx_for_ttl_similar_cases_for_bundle: [len_bundle, num_cases*self.times_more_num_cases_to_retrieve]
    # prob_for_ttl_similar_cases_for_bundle: [len_bundle, num_cases*self.times_more_num_cases_to_retrieve]
    # ttl_similar_cases_with_bundle_order: [len_bundle] (num_cases*self.times_more_num_cases_to_retrieve)
    # ttl_cur_tuple_with_bundle_order: [len_bundle] (num_cases*self.times_more_num_cases_to_retrieve)
    ## OUTPUT:
    # idx_for_ttl_similar_cases_for_bundle: [len_bundle, num_cases]
    # prob_for_ttl_similar_cases_for_bundle: [len_bundle, num_cases]
    # ttl_similar_cases_with_bundle_order: [len_bundle] (num_cases)
    # ttl_cur_tuple_with_bundle_order: [len_bundle]
    # obj_line_id_bundleOrder: [len_bundle]
    def rerank(self, idx_for_ttl_similar_cases_for_bundle, prob_for_ttl_similar_cases_for_bundle, ttl_similar_cases_with_bundle_order, ttl_cur_tuple_with_bundle_order, obj_line_id_bundleOrder, data_type):
        # make sure the four input has the same len_bundle
        debug_normal_retrieval, debug_abnormal_retrieval = 0, 0
        assert idx_for_ttl_similar_cases_for_bundle.size()[0] == prob_for_ttl_similar_cases_for_bundle.size()[0]
        assert idx_for_ttl_similar_cases_for_bundle.size()[0] == len(ttl_similar_cases_with_bundle_order)
        assert idx_for_ttl_similar_cases_for_bundle.size()[0] == len(ttl_cur_tuple_with_bundle_order)
        assert idx_for_ttl_similar_cases_for_bundle.size()[0] == obj_line_id_bundleOrder.size()[0]
        if self.args.rerank_selection == 0:
            # reverse order of demonstrations
            if self.args.if_reverse_order_demonstrations:
                raise NotImplementedError
            assert self.times_more_num_cases_to_retrieve == 1
            # save cases to further test BLEU
            if data_type == 'eval':
                self.ttl_similar_cases_eval += ttl_similar_cases_with_bundle_order
            elif data_type == 'test':
                self.ttl_similar_cases_test += ttl_similar_cases_with_bundle_order
            return idx_for_ttl_similar_cases_for_bundle, prob_for_ttl_similar_cases_for_bundle, \
                    ttl_similar_cases_with_bundle_order, ttl_cur_tuple_with_bundle_order
        elif self.args.rerank_selection == 1 or self.args.rerank_selection == 2:
            ## get len_bundle and num_cases
            len_bundle = idx_for_ttl_similar_cases_for_bundle.size()[0]
            assert idx_for_ttl_similar_cases_for_bundle.size()[1] % self.times_more_num_cases_to_retrieve == 0
            num_cases = int(idx_for_ttl_similar_cases_for_bundle.size()[1] / self.times_more_num_cases_to_retrieve)
            # number of items in id_with_different_source to collect
            if self.args.larger_range_to_select_retrieval_randomly_from != -1 and data_type == 'train':
                print("num_cases: ", num_cases)
                num_maybe_expanded_cases = np.maximum(self.args.larger_range_to_select_retrieval_randomly_from, num_cases)
                # num_maybe_expanded_cases shouldn't exceed self.n_shot-2;
                # idx_for_ttl_similar_cases_for_bundle.size()[1] >= self.n_shot-2 and can be filled with irrelavant ids
                # num_maybe_expanded_cases = np.minimum(idx_for_ttl_similar_cases_for_bundle.size()[1], num_maybe_expanded_cases)
                # *0.25 means we only want to use top retrievals for in-context demonstration combinations
                num_maybe_expanded_cases = np.minimum(int((self.n_shot-2)*0.25), num_maybe_expanded_cases)
                num_maybe_expanded_cases = np.maximum(num_maybe_expanded_cases, num_cases)

                # count the distribution of number of items in id_with_different_source
                cnt_id_with_different_source = {}
            else:
                num_maybe_expanded_cases = num_cases
            ## begin selection (retrieved cases' source can't be the same with current query)
            for id_bundle in range(len_bundle):
                # If rerank == 2 and data_type == 'eval' or 'test', then the model can at most get one case that is with same source
                if self.args.rerank_selection == 2 and data_type != 'train':
                    num_allowed_case_with_same_source = 1
                else:
                    num_allowed_case_with_same_source = 0
                id_with_different_source = []
                dict_retrieved_cases_source = {}
                # id_with_different_relation is used when train set do not contain any case with same relation
                id_with_different_relation = []
                cur_query = ttl_cur_tuple_with_bundle_order[id_bundle].strip('\n').split('\t')[1]
                cur_rel = ttl_cur_tuple_with_bundle_order[id_bundle].strip('\n').split('\t')[0]
                tmp_ttl_cases = ttl_similar_cases_with_bundle_order[id_bundle].strip('\n').split('\t\t')
                # when using subset, it's very possible that we do not have tmp_ttl_cases as full size as idx_for_ttl_similar_cases_for_bundle.size()[1]
                # many in idx_for_ttl_similar_cases_for_bundle.size()[1] would be -1
                if self.args.subset_selection == -1 and self.args.dataset_selection != 0:
                    try:
                        assert len(tmp_ttl_cases) == idx_for_ttl_similar_cases_for_bundle.size()[1]
                    except:
                        print("len(tmp_ttl_cases): ", len(tmp_ttl_cases))
                        print("idx_for_ttl_similar_cases_for_bundle.size()[1]: ", idx_for_ttl_similar_cases_for_bundle.size()[1])
                        raise Exception("Not enough tmp_ttl_cases is retrieved")
                # get rid of retrieved cases that share the same source
                try:
                    assert len(tmp_ttl_cases) >= num_cases
                except:
                    print("len(tmp_ttl_cases): ", len(tmp_ttl_cases))
                    print("num_cases: ", num_cases)
                    assert len(tmp_ttl_cases) >= num_cases
                for id_case in range(len(tmp_ttl_cases)):
                    tmp_ttl_cases[id_case] = tmp_ttl_cases[id_case].strip('\n')
                    tmp_source = tmp_ttl_cases[id_case].split('\t')[1]
                    tmp_rel = tmp_ttl_cases[id_case].split('\t')[0]
                    # Only when dataset is conceptnet, tmp_rel != cur_rel is allowed;
                    # But we do not use case whose rel != cur_rel in rerank1 and rerank2
                    if tmp_rel != cur_rel:
                        if self.args.dataset_selection == 0:
                            # print("Warning: tmp_rel != cur_rel")
                            id_with_different_relation.append(id_case)
                            continue
                        else:
                            print('tmp_rel: ', tmp_ttl_cases[id_case].split('\t'))
                            print('cur_rel: ', ttl_cur_tuple_with_bundle_order[id_bundle].strip('\n').split('\t'))
                            raise Exception("tmp_rel != cur_rel")
                    if tmp_source.strip() != cur_query.strip():
                        if not self.args.use_only_sub_rel_for_retrieval:
                            # old method 8/17/2021: 11.54 p.m.
                            id_with_different_source.append(id_case)
                        elif tmp_source not in dict_retrieved_cases_source:
                            dict_retrieved_cases_source[tmp_source] = [id_case]
                            id_with_different_source.append(id_case)
                        else:
                            print("{} founded again".format(tmp_source))
                            continue
                    elif num_allowed_case_with_same_source > 0:
                        id_with_different_source.append(id_case)
                        num_allowed_case_with_same_source -= 1
                    # if len(id_with_different_source) >= num_cases:
                    if len(id_with_different_source) >= num_maybe_expanded_cases:
                        break
                # print("1. len(id_with_different_source): ", len(id_with_different_source))
                if self.args.dataset_selection != 0:
                    try:
                        assert len(id_with_different_source) >= num_cases
                    except:
                        print("len(id_with_different_source): ", len(id_with_different_source))
                        print("num_cases: ", num_cases)
                        raise Exception
                else:
                    # if len(id_with_different_source) == self.args.num_cases and len(id_with_different_relation) == 0:
                    if len(id_with_different_source) >= self.args.num_cases and len(id_with_different_relation) == 0:
                        debug_normal_retrieval += 1
                    else:
                        # print("len(id_with_different_source): {}, len(id_with_different_relation): {}".format(len(id_with_different_source), len(id_with_different_relation)))
                        debug_abnormal_retrieval += 1
                    # when the dataset is conceptnet, for some relation the available retrieval cases is less than num_cases
                    # In this circumstance, we choose to replicate the existing available retrieval cases
                    while len(id_with_different_source) < num_cases:
                        # if len(id_with_different_source) == 0, this while loop will stuck
                        if len(id_with_different_source) > 0:
                            id_with_different_source = id_with_different_source + id_with_different_source
                        else:
                            try:
                                assert len(id_with_different_relation) > 0
                            except:
                                print("len(id_with_different_relation): ", len(id_with_different_relation))
                                print("tmp_ttl_cases: ", tmp_ttl_cases)
                                print('cur_rel: ', ttl_cur_tuple_with_bundle_order[id_bundle].strip('\n').split('\t'))
                                print("times_more_num_cases_to_retrieve: ", self.times_more_num_cases_to_retrieve)
                                raise Exception("Can't load id_with_different_relation OR times_more_num_cases_to_retrieve is not enough.")
                            id_with_different_source = id_with_different_relation
                        # restrict the length of id_with_different_source when self.args.larger_range_to_select_retrieval_randomly_from != -1
                        if self.args.larger_range_to_select_retrieval_randomly_from != -1 and len(id_with_different_source) >= self.args.larger_range_to_select_retrieval_randomly_from:
                            id_with_different_source = id_with_different_source[:self.args.larger_range_to_select_retrieval_randomly_from]

                # Modify this "if" code largely in 8/22/2021: 11:53 p.m.;
                # Check github if you are interested in the previous version of this "if" code
                # According to "Recency Bias", I think we should keep the
                #       demonstrations (to put in input_ids) in order of similarity
                if self.args.larger_range_to_select_retrieval_randomly_from != -1 and data_type == 'train':
                    # print("2. len(id_with_different_source): ", len(id_with_different_source))
                    # count the number of items in id_with_different_source
                    if len(id_with_different_source) not in cnt_id_with_different_source:
                        cnt_id_with_different_source[len(id_with_different_source)] = 1
                    else:
                        cnt_id_with_different_source[len(id_with_different_source)] += 1
                    # id in tmp_id_for_id_with_different_source should be sorted \
                    #       (to keep the order of id_with_different_source)
                    tmp_id_for_id_with_different_source = get_id_for_id_with_different_source_during_larger_range(id_with_different_source, self.args.larger_range_to_select_retrieval_randomly_from, num_cases)
                    assert len(tmp_id_for_id_with_different_source) == num_cases
                    id_with_different_source = np.array(id_with_different_source)
                    id_with_different_source = id_with_different_source[tmp_id_for_id_with_different_source]
                    id_with_different_source = id_with_different_source.tolist()
                    # only select num_cases cases
                    id_with_different_source = id_with_different_source[:num_cases]
                else:
                    id_with_different_source = id_with_different_source[:num_cases]
                # reverse order of demonstrations
                if self.args.if_reverse_order_demonstrations:
                    id_with_different_source.reverse()
                    if id_bundle == 0:
                        print("INFO: Demonstrations reversed")
                # get selected_idx_for_ttl_similar_cases_for_bundle / selected_prob_for_ttl_similar_cases_for_bundle
                if id_bundle == 0:
                    selected_idx_for_ttl_similar_cases_for_bundle = idx_for_ttl_similar_cases_for_bundle[id_bundle, id_with_different_source].unsqueeze(0)
                    selected_prob_for_ttl_similar_cases_for_bundle = prob_for_ttl_similar_cases_for_bundle[id_bundle, id_with_different_source].unsqueeze(0)
                    selected_ttl_similar_cases_with_bundle_order = ['\t\t'.join([tmp_ttl_cases[i] for i in id_with_different_source]) + '\n']
                else:
                    # cur_selected_idx
                    cur_selected_idx = idx_for_ttl_similar_cases_for_bundle[id_bundle, id_with_different_source].unsqueeze(0)
                    selected_idx_for_ttl_similar_cases_for_bundle = torch.cat((selected_idx_for_ttl_similar_cases_for_bundle, cur_selected_idx), dim=0)
                    # cur_selected_idx_prob
                    cur_selected_idx_prob = prob_for_ttl_similar_cases_for_bundle[id_bundle, id_with_different_source].unsqueeze(0)
                    selected_prob_for_ttl_similar_cases_for_bundle = torch.cat((selected_prob_for_ttl_similar_cases_for_bundle, cur_selected_idx_prob), dim=0)
                    # tmp_ttl_cases
                    cur_tmp_ttl_cases = '\t\t'.join([tmp_ttl_cases[i] for i in id_with_different_source]) + '\n'
                    selected_ttl_similar_cases_with_bundle_order.append(cur_tmp_ttl_cases)
            # save cases to further test BLEU
            if data_type == 'eval':
                self.ttl_similar_cases_eval += selected_ttl_similar_cases_with_bundle_order
            elif data_type == 'test':
                self.ttl_similar_cases_test += selected_ttl_similar_cases_with_bundle_order
            # Add current (sub, rel, obj) tuple to retrieved case
            if self.args.rerank_selection == 2 and data_type == 'train':
                for id_bundle in range(len_bundle):
                    tmp_rand = np.random.rand(1)[0]
                    if tmp_rand > self.args.possibility_add_cur_tuple_to_its_retrieved_cases:
                    # if tmp_rand >  0.0:
                        tmp_id = int(tmp_rand * 101) % num_cases
                        # if if_use_full_memory_store_while_subset == True, obj_line_id_bundleOrder is line_id in few shot set, while \
                        # selected_idx_for_ttl_similar_cases_for_bundle is line_id in full train set; they are not aligned
                        assert not self.args.if_use_full_memory_store_while_subset
                        # tmp_id = 0
                        selected_idx_for_ttl_similar_cases_for_bundle[id_bundle, tmp_id] = obj_line_id_bundleOrder[id_bundle]
                        selected_prob_for_ttl_similar_cases_for_bundle[id_bundle, tmp_id] = 1.0
                        tmp_ttl_similar_cases = selected_ttl_similar_cases_with_bundle_order[id_bundle].strip().split('\t\t')
                        tmp_ttl_similar_cases[tmp_id] = ttl_cur_tuple_with_bundle_order[id_bundle].strip()
                        selected_ttl_similar_cases_with_bundle_order[id_bundle] = '\t\t'.join(tmp_ttl_similar_cases) + '\n'

            print("debug_normal_retrieval: {}, debug_abnormal_retrieval: {}".format(debug_normal_retrieval, debug_abnormal_retrieval))
            if self.args.larger_range_to_select_retrieval_randomly_from != -1 and data_type == 'train':
                print("cnt_id_with_different_source: ", cnt_id_with_different_source)
            return selected_idx_for_ttl_similar_cases_for_bundle, selected_prob_for_ttl_similar_cases_for_bundle, selected_ttl_similar_cases_with_bundle_order, ttl_cur_tuple_with_bundle_order





    # get cases for train data
    # INPUT:
    # bundle_embed_rlts: [obj_emb, obj_rel_list, obj_line_id, obj_ori_id_in_bundle]
    # counter_while_loop: when is 0, print time.time()
    # sample_ckb_dict_subRel: is not None only when args.if_double_retrieval == True; same usage as sample_ckb_dict
    # output:
    #   sample_ckb_train_embed: [sample_size, 968]
    #   ttl_similar_cases_with_bundle_order: [num_steps*train_batch_size]  (after '\t\t'.join())
    #   ttl_cur_tuple_with_bundle_order: [num_steps*train_batch_size] (['rel\tsub\tobj\n', ...])
    #   idx_for_ttl_similar_cases_for_bundle: [num_steps*train_batch_size, num_cases]
    #   prob_for_ttl_similar_cases_for_bundle: [num_steps*train_batch_size, num_cases]
    #   encoded_cases: [encoded_cases_gene, encoded_cases_retr]
    #       encoded_cases_gene: [doc_gene_input_ids, doc_gene_attention_mask, doc_gene_lm_labels]
    #       encoded_cases_retr: [doc_retr_cases_input_ids, doc_retr_cases_attention_mask, doc_retr_cases_segment_ids]
    #           doc_gene_input_ids: [len_bundle, n_doc, cases_per_doc * input_len_gene]
    #           doc_retr_cases_input_ids: [len_bundle, n_doc, cases_per_doc, input_len_retr]
    def get_and_save_encoded_cases_and_check_result(self, sample_ckb_dict, bundle_embed_rlts, \
            rel_ttl_batch, counter_while_loop, num_cases=None, n_doc=None, data_type='train', sample_ckb_dict_subRel=None):
        assert data_type == "train" or data_type == "eval" or data_type == "test"
        ## get train cases
        start_time = time.time()
        if self.args.if_double_retrieval:
            # double retrieval for better retrieval quality
            idx_for_ttl_similar_cases_for_bundle, prob_for_ttl_similar_cases_for_bundle, ttl_similar_cases_with_bundle_order, ttl_cur_tuple_with_bundle_order, obj_line_id_bundleOrder = self.get_selected_idx_and_probs_double_retrieval(
            sample_ckb_dict=sample_ckb_dict, sample_ckb_dict_subRel=sample_ckb_dict_subRel, bundle_embed_rlts=bundle_embed_rlts, num_cases=num_cases, data_type=data_type)
        else:
            # classical once retrieval
            idx_for_ttl_similar_cases_for_bundle, prob_for_ttl_similar_cases_for_bundle, ttl_similar_cases_with_bundle_order, ttl_cur_tuple_with_bundle_order, obj_line_id_bundleOrder = self.get_selected_idx_and_probs(
            sample_ckb_dict=sample_ckb_dict, bundle_embed_rlts=bundle_embed_rlts, num_cases=num_cases, data_type=data_type)
        if counter_while_loop == 0:
            start_time_step1 = time.time()
            print("--- Finish MIPS step 1: %s seconds ---" % (start_time_step1 - start_time))
        # only to get the retrieval_index_id for sentiment sentence dataset (so to get the case difference feature)
        # if self.args.dataset_selection == 4 and data_type != "train":
        if self.args.dataset_selection == 4:
            torch.save(idx_for_ttl_similar_cases_for_bundle, os.path.join(self.args.output_dir, "retrieved_ids_sentiment_sentence_classification_{}_{}.pt".format(data_type, counter_while_loop)))
        idx_for_ttl_similar_cases_for_bundle, prob_for_ttl_similar_cases_for_bundle, ttl_similar_cases_with_bundle_order, ttl_cur_tuple_with_bundle_order = self.rerank(idx_for_ttl_similar_cases_for_bundle, prob_for_ttl_similar_cases_for_bundle, ttl_similar_cases_with_bundle_order, ttl_cur_tuple_with_bundle_order, obj_line_id_bundleOrder, data_type)
        if counter_while_loop == 0:
            # print('MIPS finished! Successfully got selected_idx')
            print("--- Finish MIPS step 2: %s seconds ---" % (time.time() - start_time_step1))
        # for DEBUG:
        torch.save(prob_for_ttl_similar_cases_for_bundle, os.path.join(self.args.output_dir, 'prob_for_ttl_similar_cases_for_bundle.pt'))

        start_time = time.time()
        encoded_cases = self.get_and_save_encoded_cases(idx_for_ttl_similar_cases_for_bundle, num_cases=num_cases, n_doc=n_doc, data_type=data_type)
        if counter_while_loop == 0:
            # print('God encoded_cases successfully!')
            print("--- Get encoded cases: %s seconds ---" % (time.time() - start_time))
            # print(len(ttl_similar_cases_for_batch), prob_for_ttl_similar_cases_for_batch.size())
            # print('ttl_similar_cases_for_batch[0:20]:', ttl_similar_cases_for_batch[0:20])

        start_time = time.time()
        self.check_result(rel_ttl_batch, ttl_similar_cases_with_bundle_order, prob_for_ttl_similar_cases_for_bundle, num_cases=num_cases)
        torch.save(rel_ttl_batch, self.batch_rel_dir)
        if counter_while_loop == 0:
            print("--- Result checked: %s seconds ---" % (time.time() - start_time))
        # return ttl_similar_cases_for_batch, prob_for_ttl_similar_cases_for_batch

    # check results
    # check ttl_similar_cases_for_batch match with prob_for_ttl_similar_cases_for_batch
    # check rel in ttl_similar_cases_for_batch match with batch
    def check_result(self, rel_ttl_batch, ttl_similar_cases_for_batch, prob_for_ttl_similar_cases_for_batch, num_cases):
        if num_cases == None:
            num_cases = self.num_cases
        # check rel
        assert rel_ttl_batch.size()[0] == len(ttl_similar_cases_for_batch)
        assert prob_for_ttl_similar_cases_for_batch.size()[0] == len(ttl_similar_cases_for_batch)

        # if self.args.dataset_selection == 0, then we have to keep some ttl_similar_cases_for_batch even the relation doesn't match
        # when rerank_selection == 1 or 2, which will violate the assertion in check_result function, but that's ok
        if self.args.dataset_selection == 0 and (self.args.rerank_selection == 1 or self.args.rerank_selection == 2):
            return 0

        rel_id_map = {}
        for id in range(rel_ttl_batch.size()[0]):
            # check ttl_similar_cases_for_batch match with prob_for_ttl_similar_cases_for_batch
            prob = prob_for_ttl_similar_cases_for_batch[id]

            if ttl_similar_cases_for_batch[id] == '\n':
                # if not torch.sum((prob == 0).float()) == num_cases:
                #     print(torch.sum((prob == 0).float()))
                    # print ValueError
                if not torch.sum((prob == 0).float()) == num_cases:
                    print('torch.sum((prob == 0).float()): ', torch.sum((prob == 0).float()))
                    print('num_cases: ', num_cases)
            elif ttl_similar_cases_for_batch[id] == '':
                raise ValueError
            else:
                # if not (num_cases - torch.sum((prob == 0).float()).item()) == len(ttl_similar_cases_for_batch[id].split('\t\t')):
                #     print(torch.sum((prob == 0).float()), len(ttl_similar_cases_for_batch[id].split('\t\t')), ttl_similar_cases_for_batch[id].split('\t\t'))
                if not (num_cases - torch.sum((prob == 0).float()).item()) == len(ttl_similar_cases_for_batch[id].split('\t\t')):
                    print("num_cases: ", num_cases)
                    print("torch.sum((prob == 0).float()).item(): ", torch.sum((prob == 0).float()).item())
                    print("prob: ", prob)
                    print("len(ttl_similar_cases_for_batch[id].split('\t\t')): ", len(ttl_similar_cases_for_batch[id].split('\t\t')))
                    print("ttl_similar_cases_for_batch[id].split('\t\t'): ", ttl_similar_cases_for_batch[id].split('\t\t'))
                    raise Exception("assertion error")
            # check rel in ttl_similar_cases_for_batch match with batch
            if ttl_similar_cases_for_batch[id] != '\n':
                rel = ttl_similar_cases_for_batch[id].split('\t\t')[0].split('\t')[0]
                if rel not in rel_id_map:
                    rel_id_map[rel] = rel_ttl_batch[id]
                else:
                    # if not rel_id_map[rel] == rel_ttl_batch[id]:
                    #     print(rel_id_map[rel], rel_ttl_batch[id])
                    assert rel_id_map[rel] == rel_ttl_batch[id]

    # get sample_size_each_rel of sample for each relation;
    # get a sample with size sample_size_total
    # if exists relation with size less than sample_size_each_rel, then random get other not selected cases to fill sample_size_total
    def get_sample_ids(self, data_type='train'):
        if data_type == 'train':
            cat_len_noter = self.cat_len_noter_train
            # train_lines is required to be sorted by rel
            lines = self.train_lines
        else:
            raise Exception("Full set is expected when using eval set or test set")
        len_lines = len(lines)
        assert self.sample_size_total <= len_lines

        # available_ids
        available_ids = list(range(len_lines))
        # sample_ids
        sample_ids = []
        ## sample 50 data items for each rel
        for id, end_id in enumerate(cat_len_noter):
            if id == 0:
                start_id = 0
            else:
                start_id = cat_len_noter[id-1] + 1
            sample_size_tmp = np.minimum(self.sample_size_each_rel, end_id-start_id)
            # if sample_size_tmp < sample_size_each_rel:
            #     print('sample_size_tmp:', sample_size_tmp)
            sample_ids_tmp = random.sample(available_ids[start_id:end_id], sample_size_tmp)
            # available_ids.pop(sample_ids_tmp)
            for sample_id in sample_ids_tmp:
                available_ids.remove(sample_id)
            sample_ids += sample_ids_tmp
            # for DEBUG
            # print('id: ', id, 'sample_size_tmp: ', sample_size_tmp)

        ## other samples
        sample_ids += random.sample(available_ids, self.sample_size_total-len(sample_ids))
        sample_ids = sorted(set(sample_ids))
        assert len(sample_ids) == self.sample_size_total
        return torch.tensor(sample_ids)

    def get_sample_ckb_dict(self, sample_ckb_train_embed=None, sample_ckb_rel_for_embed=None, sample_ckb_line_id_for_embed=None):
        ## get CKB train_embedding according to different rel
        sample_ckb_dict = {}
        begin_rel, last_rel, cur_rel = None, None, None
        begin_rel_id, last_rel_id = 0, 0
        len_sample_ckb_train_embed = sample_ckb_train_embed.size()[0]
        print('len_sample_ckb_train_embed: ', len_sample_ckb_train_embed)
        for id_emb in range(len_sample_ckb_train_embed):
            cur_rel = sample_ckb_rel_for_embed[id_emb]
            # FQ: not adding 'or id_emb == len_sample_ckb_train_embed-1'
            if (cur_rel != last_rel and id_emb != 0):
                # if not cur_rel == last_rel + 1:
                #     print('last_rel: ', last_rel, ' cur_rel: ', cur_rel)
                #     raise ValueError('One inferential dimension get no CKB embedding')
                # print(begin_rel, last_rel)
                # print(begin_rel_id, last_rel_id)
                sample_ckb_dict[last_rel.item()] = [sample_ckb_train_embed[begin_rel_id:id_emb, :], begin_rel_id, sample_ckb_line_id_for_embed[begin_rel_id:id_emb]]
                # print("sample_ckb_dict, last_rel: ", last_rel.item(), "len: ", last_rel_id - begin_rel_id)
                # print('sample_ckb_dict[last_rel.item()]:', sample_ckb_dict[last_rel.item()][0].size())
                begin_rel = cur_rel
                begin_rel_id = id_emb
                # not consider two "if" happen at the same time yet
                assert id_emb != len_sample_ckb_train_embed-1
            elif id_emb == len_sample_ckb_train_embed-1:
                id_emb += 1
                sample_ckb_dict[last_rel.item()] = [sample_ckb_train_embed[begin_rel_id:id_emb, :], begin_rel_id, sample_ckb_line_id_for_embed[begin_rel_id:id_emb]]
                begin_rel = cur_rel
                begin_rel_id = id_emb
            last_rel = cur_rel
            last_rel_id = id_emb
        print("sample_ckb_dict.keys(): ", sample_ckb_dict.keys())
        return sample_ckb_dict

    # self.next_bundle: #train(gene_doc_input_ids, gene_doc_attention_mask, gene_doc_lm_labels, \
                            #  gene_cur_input_ids, gene_cur_attention_mask, gene_cur_lm_labels, \
    #                          data_idx_ids, rel_collection, \
    #                          retr_doc_input_ids, retr_doc_attention_mask, retr_doc_segment_ids,
    #                          retr_input_ids, retr_attention_mask, retr_segment_ids)
    def get_bundle_embed_list(self, model_type):
        ## load obj_emb
        # 'train', 'eval' and 'test' all share self.next_bundle and self.batch_train_embed_dir
        save_dir = self.batch_train_embed_dir
        rel_ttl_batch = self.next_bundle[-7]
        # obj_rel_list should be sorted
        obj_emb, obj_rel_list, obj_line_id, obj_ori_id_in_bundle = self.get_embed_for_train_batches(model_type=model_type, save_dir=save_dir, next_bundle=copy.deepcopy(self.next_bundle))
        # print('obj_emb:', obj_emb.size())
        # print('obj_rel_list: (should be sorted)', obj_rel_list)
        return [obj_emb, obj_rel_list, obj_line_id, obj_ori_id_in_bundle], rel_ttl_batch

    # get similar_cases for all train data
    # only used for training
    # sample_ckb_rel_for_embed: should be sorted
    # INPUT:
    # bundle_embed_rlts: [obj_emb, obj_rel_list, obj_line_id, obj_ori_id_in_bundle]
    # OUTPUT:
    # idx_for_ttl_similar_cases: [len(next_bundle), num_cases]
    # prob_for_ttl_similar_cases: [len(next_bundle), num_cases]
    # ttl_similar_cases_with_bundle_order: [len(next_bundle)]
    # ttl_cur_tuple_with_bundle_order: [len(next_bundle)]
    def get_selected_idx_and_probs(self, sample_ckb_dict, bundle_embed_rlts, num_cases=None, data_type=None):
        if num_cases == None:
            num_cases = self.num_cases
        if data_type == None:
            data_type = self.data_type

        # some variable to initialize
        if data_type == 'train':
            if_need_prevent_same_case = 1
            save_dir = self.bundle_train_cases_dir
            if self.args.dataset_selection == 1:
                if self.args.subset_selection == 0 or self.args.subset_selection == 1 or \
                    self.args.subset_selection_while_use_full_memory_store == 0 or \
                    self.args.subset_selection_while_use_full_memory_store == 1:
                    simi_batch_size = 8
                elif self.args.subset_selection == 2 or self.args.subset_selection == 3 or \
                    self.args.subset_selection_while_use_full_memory_store == 2 or \
                    self.args.subset_selection_while_use_full_memory_store == 3:
                    simi_batch_size = 16
                else:
                    simi_batch_size = 32
            else:
                simi_batch_size = self.simi_batch_size
        elif data_type == 'eval':
            if_need_prevent_same_case = 0
            save_dir = self.bundle_eval_cases_dir
            if self.args.dataset_selection == 1:
                simi_batch_size = 32
            else:
                simi_batch_size = self.simi_batch_size
        elif data_type == 'test':
            if_need_prevent_same_case = 0
            save_dir = self.bundle_test_cases_dir
            if self.args.dataset_selection == 1:
                simi_batch_size = 16
            else:
                simi_batch_size = self.simi_batch_size
        else:
            print("data_type: ", data_type)
            raise Exception("Not supported data_type: {}".format(data_type))
        print("simi_batch_size: ", simi_batch_size)

        # get detailed batch data
        obj_emb, obj_rel_list, obj_line_id, obj_ori_id_in_batches = bundle_embed_rlts

        ## get ttl_similar_cases
        # idx_for_ttl_similar_cases: idx means idx in original train data
        idx_for_ttl_similar_cases, prob_for_ttl_similar_cases, ttl_similar_cases = [], [], []
        ttl_cur_tuple = []
        # only for analysis
        num_retrieved_cases = {}
        # obj_emb_idx_cnter: beginning of next iteration
        obj_emb_idx_cnter = 0
        prev_obj_emb_idx_cnter = 0
        # begin iteration over obj_emb
        assert obj_emb.size()[0] > 0
        while obj_emb_idx_cnter < obj_emb.size()[0]:
            time1 = time.time()
            ## Get a batch of bundle's embedding with same rel (for batch computation)
            # check if stucks
            if obj_emb_idx_cnter == prev_obj_emb_idx_cnter and prev_obj_emb_idx_cnter != 0:
                print('obj_emb_idx_cnter stucked in : ', obj_emb_idx_cnter)
                raise ValueError
            # note obj_emb_idx_cnter
            prev_obj_emb_idx_cnter = obj_emb_idx_cnter
            # to get corresponding train_emb chunk
            obj_rel = obj_rel_list[obj_emb_idx_cnter]
            ## get batch_emb and new obj_emb_idx_cnter
            if obj_emb_idx_cnter + simi_batch_size >= obj_emb.size()[0]:
                if not obj_rel_list[obj_emb_idx_cnter] == obj_rel_list[-1]:
                    raise NotImplementedError
                batch_emb = obj_emb[obj_emb_idx_cnter:]
                batch_line_id = obj_line_id[obj_emb_idx_cnter:]
                obj_emb_idx_cnter = obj_emb.size()[0]
            elif obj_rel_list[obj_emb_idx_cnter] == obj_rel_list[obj_emb_idx_cnter + simi_batch_size]:
                batch_emb = obj_emb[obj_emb_idx_cnter: obj_emb_idx_cnter + simi_batch_size]
                batch_line_id = obj_line_id[obj_emb_idx_cnter: obj_emb_idx_cnter + simi_batch_size]
                obj_emb_idx_cnter += simi_batch_size
            else:
                for incre_idx in range(simi_batch_size):
                    if obj_rel_list[obj_emb_idx_cnter+incre_idx] != obj_rel_list[obj_emb_idx_cnter]:
                        break
                batch_emb = obj_emb[obj_emb_idx_cnter: obj_emb_idx_cnter + incre_idx]
                batch_line_id = obj_line_id[obj_emb_idx_cnter: obj_emb_idx_cnter + incre_idx]
                obj_emb_idx_cnter += incre_idx

            # 100k(conceptnet) / ~700k(atomic)
            batch_emb = batch_emb.to(device)
            ## get corresponding train_emb chunk accoring to rel
            # if can't find cases in sample_ckb_dict
            if not obj_rel.item() in sample_ckb_dict:
                for id_emb in range(len(batch_line_id)):
                    ttl_similar_cases.append('\n')
                print("INFO: obj_rel not found in sample_ckb_dict, obj_rel: ", obj_rel.item(), 'sample_ckb_dict.keys(): ', sample_ckb_dict.keys())
                if self.args.dataset_selection == 1 or self.args.dataset_selection == 2 or self.args.dataset_selection == 3 or self.args.dataset_selection == 4:
                    # suppose we can always find cases with same relation in memory store
                    raise Exception("obj_rel not found in sample_ckb_dict")
                elif self.args.dataset_selection == 0:
                    # In ConceptNet, it happens when we can find cases with same relation in memory store
                    # Q: this two lines might not be able to pass the check_result() function
                    idx_for_ttl_similar_cases.append(-1 * torch.ones(len(batch_line_id), num_cases))
                    prob_for_ttl_similar_cases.append(torch.zeros(len(batch_line_id), num_cases))
                print('rel not found in sample_ckb_dict, obj_rel: ', obj_rel.item(), 'num_cases: ', num_cases)
                continue
            # get corresponding train_emb chunk
            train_emb_sameRel = torch.transpose(sample_ckb_dict[obj_rel.item()][0], 0, 1).to(device)
            ## similarity: [1~simi_batch_size, 1~(num_cases+1)]
            similarity = torch.matmul(batch_emb, train_emb_sameRel)
            similarity = similarity.cpu()
            self.n_shot = similarity.size()[1]
            # selected_idxs: [simi_batch_size, min(25, similarity.size()[1])]
            # use 26 instead of 25
            # num_cases_tmp
            num_cases_tmp = np.minimum(num_cases*self.times_more_num_cases_to_retrieve+1, similarity.size()[1])
            # The number of cases we need to use in this function is inflated_num_cases
            # (in rerank function, we will only select num_cases cases to use)
            inflated_num_cases = num_cases*self.times_more_num_cases_to_retrieve
            if num_cases_tmp < num_cases+1 and self.args.dataset_selection != 0:
                print('obj_rel: ', obj_rel.item(), 'num_cases_tmp: ', num_cases_tmp)
                raise Exception("num_cases_tmp should be larger than num_cases+1")
            if self.args.random_retrieval:
                # rlt_topk = torch.topk(similarity, num_cases_tmp)
                # Q: modified in 8/17/2021: 1:16 a.m.
                rlt_topk = torch.topk(similarity, similarity.size()[1])
                tmp_selected_idxs = rlt_topk[1]
                tmp_selected_idx_prob = rlt_topk[0]
                tmp_data_seq = np.arange(similarity.size()[1])
                random.shuffle(tmp_data_seq)
                selected_idxs = tmp_selected_idxs[:, tmp_data_seq[:num_cases_tmp]]
                selected_idx_prob = tmp_selected_idx_prob[:, tmp_data_seq[:num_cases_tmp]]
            # Normal circumstance
            else:
                rlt_topk = torch.topk(similarity, num_cases_tmp)
                selected_idxs = rlt_topk[1]
                selected_idx_prob = rlt_topk[0]
            ## get the idxs of selected cases in full train_emb
            # print('selected_idxs.size()', selected_idxs.size())
            selected_idxs = sample_ckb_dict[obj_rel.item()][2][selected_idxs]

            # time2 = time.time()
            # print("time2 - time1: ", time2-time1)
            # batch_line_id: [simi_batch_size]
            ## Prevent same cases
            # clean_selected_idxs don't contain self id
            clean_selected_idxs, clean_selected_idxs_prob = None, None
            if if_need_prevent_same_case:
                # id_emb: idx in obj_emb
                for id_emb in range(len(batch_line_id)):
                    # id_selected_idxs: start from 0
                    cur_selected_idx = selected_idxs[id_emb]
                    cur_selected_idx_prob = selected_idx_prob[id_emb]
                    cur_obj_idx = batch_line_id[id_emb]
                    # if exist self's id
                    self_idx = (cur_selected_idx == cur_obj_idx).nonzero()
                    # only keep 25 out of 26 for cur_selected_idx
                    # if contains self_id
                    if self_idx.size()[0] > 0:
                        self_idx = self_idx[0, 0]
                        # get index
                        if self_idx < cur_selected_idx.size()[0] - 1:
                            cur_selected_idx = torch.cat((cur_selected_idx[0:self_idx], cur_selected_idx[self_idx+1:]), dim=0)
                            cur_selected_idx_prob = torch.cat((cur_selected_idx_prob[0:self_idx], cur_selected_idx_prob[self_idx+1:]), dim=0)
                        else:
                            cur_selected_idx = cur_selected_idx[0:self_idx]
                            cur_selected_idx_prob = cur_selected_idx_prob[0:self_idx]
                    else:
                        cur_selected_idx = cur_selected_idx[:-1]
                        cur_selected_idx_prob = cur_selected_idx_prob[:-1]
                    # clean_selected_idxs
                    if id_emb == 0:
                        clean_selected_idxs = cur_selected_idx.unsqueeze(0)
                        clean_selected_idxs_prob = cur_selected_idx_prob.unsqueeze(0)
                    else:
                        clean_selected_idxs = torch.cat((clean_selected_idxs, cur_selected_idx.unsqueeze(0)), dim=0)
                        clean_selected_idxs_prob = torch.cat((clean_selected_idxs_prob, cur_selected_idx_prob.unsqueeze(0)), dim=0)
            else:
                clean_selected_idxs = selected_idxs[:, :-1]
                clean_selected_idxs_prob = selected_idx_prob[:, :-1]

            # time3 = time.time()
            # print("time3 - time2: ", time3-time2)
            ## In case clean_selected_idxs don't have enough cases for some rel
            assert clean_selected_idxs.size() == clean_selected_idxs_prob.size()
            # in case clean_selected_idxs.size() == torch.Size([0])
            if clean_selected_idxs.size()[1] == 0:
                clean_selected_idxs = (-1) * torch.ones(len(batch_line_id), inflated_num_cases, dtype=torch.long)
                clean_selected_idxs_prob = torch.zeros((len(batch_line_id), inflated_num_cases))
                if self.args.dataset_selection != 0:
                    print('Warning: clean_selected_idxs.size()[1] == 0')
                # raise Exception('How can clean_selected_idxs.size()[1] == 0!')
            elif clean_selected_idxs_prob.size()[1] != 0 and clean_selected_idxs_prob.size()[1] < inflated_num_cases:
                if self.args.dataset_selection != 0 and self.args.subset_selection == -1:
                    print('Warning: clean_selected_idxs_prob.size()[1] < inflated_num_cases: ', clean_selected_idxs_prob.size()[1], inflated_num_cases)
                clean_selected_idxs = torch.cat((clean_selected_idxs, (-1) * torch.ones(len(batch_line_id), inflated_num_cases-clean_selected_idxs.size()[1], dtype=torch.long)), dim=1)
                clean_selected_idxs_prob = torch.cat((clean_selected_idxs_prob, torch.zeros(len(batch_line_id), inflated_num_cases-clean_selected_idxs_prob.size()[1])), dim=1)
            assert clean_selected_idxs.size()[1] == inflated_num_cases
            assert clean_selected_idxs_prob.size()[1] == inflated_num_cases
            ## Get cases using index
            # similar_cases: ['<oReact>\tPersonX plays a ___ in the war\tsad', ...]
            # torch.save(batch_line_id, os.path.join(self.args.output_dir, 'batch_line_id.pt'))
            # torch.save(clean_selected_idxs, os.path.join(self.args.output_dir, 'clean_selected_idxs.pt'))
            # time4 = time.time()
            # print("time4 - time3: ", time4-time3)
            for id_emb in range(len(batch_line_id)):
                if self.args.dataset_selection == 0:
                    # cur_triple: 'rel\tsub\tobj\n'
                    similar_cases, cur_triple = self.get_current_similar_case_conceptnet(batch_line_id[id_emb], clean_selected_idxs[id_emb], data_type=data_type)
                elif self.args.dataset_selection == 1:
                    # cur_triple: 'rel\tsub\tobj\n'
                    similar_cases, cur_triple = self.get_current_similar_case_atomic(batch_line_id[id_emb], clean_selected_idxs[id_emb], data_type=data_type)
                elif self.args.dataset_selection == 2:
                    # cur_triple: 'rel\tsub\tobj\n'
                    similar_cases, cur_triple = self.get_current_similar_case_shakespear_or_e2e(batch_line_id[id_emb], clean_selected_idxs[id_emb], data_type=data_type)
                elif self.args.dataset_selection == 3:
                    similar_cases, cur_triple = self.get_current_similar_case_shakespear_or_e2e(batch_line_id[id_emb], clean_selected_idxs[id_emb], data_type=data_type)
                elif self.args.dataset_selection == 4:
                    similar_cases, cur_triple = self.get_current_similar_case_shakespear_or_e2e(batch_line_id[id_emb], clean_selected_idxs[id_emb], data_type=data_type)

                # count for analysis
                len_similar_cases = len(similar_cases)
                if len_similar_cases not in num_retrieved_cases:
                    num_retrieved_cases[len_similar_cases] = 1
                else:
                    num_retrieved_cases[len_similar_cases] += 1
                # note down similar_cases
                appended_item = '\t\t'.join(similar_cases) + '\n'
                if appended_item == '':
                    raise ValueError
                ttl_similar_cases.append(appended_item)
                ttl_cur_tuple.append(cur_triple)

            idx_for_ttl_similar_cases.append(clean_selected_idxs)
            prob_for_ttl_similar_cases.append(clean_selected_idxs_prob)
            # time5 = time.time()
            # print("time5 - time4: ", time5-time4)


        # Q: .to(torch.long)
        idx_for_ttl_similar_cases = torch.cat(([d.to(torch.long) for d in idx_for_ttl_similar_cases]), dim=0)
        prob_for_ttl_similar_cases = torch.cat(([d for d in prob_for_ttl_similar_cases]), dim=0)
        print(num_retrieved_cases)
        # Get reordered idx_for_ttl_similar_cases, prob_for_ttl_similar_cases and ttl_similar_cases
        # Same order of original bundle
        re_reorder_id = torch.argsort(obj_ori_id_in_batches)
        # obj_line_id_bundleOrder: the line_id (in train set) of the bundle in original bundle orger
        obj_line_id_bundleOrder = obj_line_id[re_reorder_id]
        # idx_for_ttl_similar_cases
        idx_for_ttl_similar_cases = idx_for_ttl_similar_cases[re_reorder_id]
        # prob_for_ttl_similar_cases
        prob_for_ttl_similar_cases = prob_for_ttl_similar_cases[re_reorder_id]
        # ttl_similar_cases_with_bundle_order
        ttl_similar_cases_with_bundle_order = []
        ttl_cur_tuple_with_bundle_order = []
        for ori_id in re_reorder_id:
            if len(ttl_similar_cases[ori_id]) < 10:
                print('ori_id: ', ori_id)
                print('ttl_similar_cases[ori_id]: ', ttl_similar_cases[ori_id])
                print("Warning: ttl_similar_cases[ori_id] might not contain enough information")
                # raise Exception("ttl_similar_cases[ori_id] might not contain enough information")
            ttl_similar_cases_with_bundle_order.append(ttl_similar_cases[ori_id])
            ttl_cur_tuple_with_bundle_order.append(ttl_cur_tuple[ori_id])

        with open(save_dir, 'w') as f:
            f.writelines(ttl_similar_cases_with_bundle_order)
        # time6 = time.time()
        # print("time6 - time5: ", time6-time5)

        # idx_for_ttl_similar_cases: [len_bundle, num_cases]
        # prob_for_ttl_similar_cases: [len_bundle, num_cases]
        # ttl_similar_cases_with_bundle_order: [len_bundle]
        # ttl_cur_tuple_with_bundle_order: [len_bundle]
        return idx_for_ttl_similar_cases, prob_for_ttl_similar_cases, ttl_similar_cases_with_bundle_order, ttl_cur_tuple_with_bundle_order, obj_line_id_bundleOrder


    # get similar_cases for all train data
    # only used for training
    # sample_ckb_rel_for_embed: should be sorted
    # INPUT:
    # bundle_embed_rlts: [obj_emb, obj_rel_list, obj_line_id, obj_ori_id_in_bundle]
    # sample_ckb_dict_subRel: same usage as sample_ckb_dict
    # OUTPUT:
    # idx_for_ttl_similar_cases: [len(next_bundle), num_cases]
    # prob_for_ttl_similar_cases: [len(next_bundle), num_cases]
    # ttl_similar_cases_with_bundle_order: [len(next_bundle)]
    # ttl_cur_tuple_with_bundle_order: [len(next_bundle)]
    def get_selected_idx_and_probs_double_retrieval(self, sample_ckb_dict, sample_ckb_dict_subRel, bundle_embed_rlts, num_cases=None, data_type=None):
        if num_cases == None:
            num_cases = self.num_cases
        if data_type == None:
            data_type = self.data_type

        # some variable to initialize
        if data_type == 'train':
            if_need_prevent_same_case = 1
            save_dir = self.bundle_train_cases_dir
            if self.args.dataset_selection == 1:
                if self.args.subset_selection == 0 or self.args.subset_selection == 1 or \
                    self.args.subset_selection_while_use_full_memory_store == 0 or \
                    self.args.subset_selection_while_use_full_memory_store == 1:
                    simi_batch_size = 8
                elif self.args.subset_selection == 2 or self.args.subset_selection == 3 or \
                    self.args.subset_selection_while_use_full_memory_store == 2 or \
                    self.args.subset_selection_while_use_full_memory_store == 3:
                    simi_batch_size = 16
                else:
                    simi_batch_size = 32
            else:
                simi_batch_size = self.simi_batch_size
        elif data_type == 'eval':
            if_need_prevent_same_case = 0
            save_dir = self.bundle_eval_cases_dir
            if self.args.dataset_selection == 1:
                simi_batch_size = 32
            else:
                simi_batch_size = self.simi_batch_size
        elif data_type == 'test':
            if_need_prevent_same_case = 0
            save_dir = self.bundle_test_cases_dir
            if self.args.dataset_selection == 1:
                simi_batch_size = 16
            else:
                simi_batch_size = self.simi_batch_size
        else:
            print("data_type: ", data_type)
            raise Exception("Not supported data_type: {}".format(data_type))
        print("simi_batch_size: ", simi_batch_size)

        # get detailed batch data
        obj_emb, obj_rel_list, obj_line_id, obj_ori_id_in_batches = bundle_embed_rlts

        ## get ttl_similar_cases
        # idx_for_ttl_similar_cases: idx means idx in original train data
        idx_for_ttl_similar_cases, prob_for_ttl_similar_cases, ttl_similar_cases = [], [], []
        ttl_cur_tuple = []
        # only for analysis
        num_retrieved_cases = {}
        # obj_emb_idx_cnter: beginning of next iteration
        obj_emb_idx_cnter = 0
        prev_obj_emb_idx_cnter = 0
        # begin iteration over obj_emb
        assert obj_emb.size()[0] > 0
        while obj_emb_idx_cnter < obj_emb.size()[0]:
            time1 = time.time()
            ## Get a batch of bundle's embedding with same rel (for batch computation)
            # check if stucks
            if obj_emb_idx_cnter == prev_obj_emb_idx_cnter and prev_obj_emb_idx_cnter != 0:
                print('obj_emb_idx_cnter stucked in : ', obj_emb_idx_cnter)
                raise ValueError
            # note obj_emb_idx_cnter
            prev_obj_emb_idx_cnter = obj_emb_idx_cnter
            # to get corresponding train_emb chunk
            obj_rel = obj_rel_list[obj_emb_idx_cnter]
            ## get batch_emb and new obj_emb_idx_cnter
            if obj_emb_idx_cnter + simi_batch_size >= obj_emb.size()[0]:
                if not obj_rel_list[obj_emb_idx_cnter] == obj_rel_list[-1]:
                    raise NotImplementedError
                batch_emb = obj_emb[obj_emb_idx_cnter:]
                batch_line_id = obj_line_id[obj_emb_idx_cnter:]
                obj_emb_idx_cnter = obj_emb.size()[0]
            elif obj_rel_list[obj_emb_idx_cnter] == obj_rel_list[obj_emb_idx_cnter + simi_batch_size]:
                batch_emb = obj_emb[obj_emb_idx_cnter: obj_emb_idx_cnter + simi_batch_size]
                batch_line_id = obj_line_id[obj_emb_idx_cnter: obj_emb_idx_cnter + simi_batch_size]
                obj_emb_idx_cnter += simi_batch_size
            else:
                for incre_idx in range(simi_batch_size):
                    if obj_rel_list[obj_emb_idx_cnter+incre_idx] != obj_rel_list[obj_emb_idx_cnter]:
                        break
                batch_emb = obj_emb[obj_emb_idx_cnter: obj_emb_idx_cnter + incre_idx]
                batch_line_id = obj_line_id[obj_emb_idx_cnter: obj_emb_idx_cnter + incre_idx]
                obj_emb_idx_cnter += incre_idx

            # 100k(conceptnet) / ~700k(atomic)
            batch_emb = batch_emb.to(device)
            ## get corresponding train_emb chunk accoring to rel
            # if can't find cases in sample_ckb_dict or sample_ckb_dict_subRel
            # print("obj_rel.item(): ", obj_rel.item())
            # if not obj_rel.item() in sample_ckb_dict or not obj_rel.item() in sample_ckb_dict_subRel:
            try:
                tmp = sample_ckb_dict_subRel[obj_rel.item()]
            except:
                print("obj_rel: ", obj_rel)
                print("obj_rel.item(): ", obj_rel.item())
                print("sample_ckb_dict_subRel.keys(): ", sample_ckb_dict_subRel.keys())
                raise Exception
            if not obj_rel.item() in sample_ckb_dict:
                for id_emb in range(len(batch_line_id)):
                    ttl_similar_cases.append('\n')
                print("INFO: obj_rel not found in sample_ckb_dict, obj_rel: ", obj_rel.item(), 'sample_ckb_dict.keys(): ', sample_ckb_dict.keys())
                if self.args.dataset_selection == 1 or self.args.dataset_selection == 2 or self.args.dataset_selection == 3 or self.args.dataset_selection == 4:
                    # suppose we can always find cases with same relation in memory store
                    raise Exception("obj_rel not found in sample_ckb_dict")
                elif self.args.dataset_selection == 0:
                    # In ConceptNet, it happens when we can find cases with same relation in memory store
                    # Q: this two lines might not be able to pass the check_result() function
                    idx_for_ttl_similar_cases.append(-1 * torch.ones(len(batch_line_id), num_cases))
                    prob_for_ttl_similar_cases.append(torch.zeros(len(batch_line_id), num_cases))
                print('rel not found in sample_ckb_dict, obj_rel: ', obj_rel.item(), 'num_cases: ', num_cases)
                continue
            # get corresponding train_emb chunk
            train_emb_sameRel = torch.transpose(sample_ckb_dict[obj_rel.item()][0], 0, 1).to(device)
            train_emb_subRel_sameRel = torch.transpose(sample_ckb_dict_subRel[obj_rel.item()][0], 0, 1).to(device)
            ## similarity: [1~simi_batch_size, 1~(num_cases+1)]
            similarity = torch.matmul(batch_emb, train_emb_sameRel)
            similarity = similarity.cpu()
            # note down nshot
            self.n_shot = similarity.size()[1]
            ## similarity_subRel
            similarity_subRel = torch.matmul(batch_emb, train_emb_subRel_sameRel)
            similarity_subRel = similarity_subRel.cpu()
            assert similarity.size() == similarity_subRel.size()
            # selected_idxs: [simi_batch_size, min(25, similarity.size()[1])]
            # use 26 instead of 25
            # num_cases_tmp
            num_cases_tmp = np.minimum(num_cases*self.times_more_num_cases_to_retrieve+1, similarity.size()[1])
            print("a: {}, b: {}, c:{}, d:{}, num_cases_tmp: {}".format(num_cases*self.times_more_num_cases_to_retrieve+1, similarity.size()[1], num_cases, self.times_more_num_cases_to_retrieve, num_cases_tmp))
            # The number of cases we need to use in this function is inflated_num_cases
            # (in rerank function, we will only select num_cases cases to use)
            inflated_num_cases = num_cases*self.times_more_num_cases_to_retrieve
            if num_cases_tmp < num_cases+1 and self.args.dataset_selection != 0:
                print('obj_rel: ', obj_rel.item(), 'num_cases_tmp: ', num_cases_tmp)
                raise Exception("num_cases_tmp should be larger than num_cases+1")

            num_cases_tmp_subRel = get_k_for_topk_subRel(similarity_subRel.size()[1], num_cases, filter_ratio=self.args.filter_ratio)
            rlt_subRel_topk = torch.topk(similarity_subRel, num_cases_tmp_subRel)
            rlt_topk = torch.topk(similarity, num_cases_tmp)
            selected_idxs, selected_idx_prob = get_selected_idxs_and_prob_from_double_topk_result(rlt_topk, rlt_subRel_topk)
            ## get the idxs of selected cases in full train_emb
            # print('selected_idxs.size()', selected_idxs.size())
            selected_idxs = sample_ckb_dict[obj_rel.item()][2][selected_idxs]

            # time2 = time.time()
            # print("time2 - time1: ", time2-time1)
            # batch_line_id: [simi_batch_size]
            ## Prevent same cases
            # clean_selected_idxs don't contain self id
            clean_selected_idxs, clean_selected_idxs_prob = None, None
            if if_need_prevent_same_case:
                # id_emb: idx in obj_emb
                for id_emb in range(len(batch_line_id)):
                    # id_selected_idxs: start from 0
                    cur_selected_idx = selected_idxs[id_emb]
                    cur_selected_idx_prob = selected_idx_prob[id_emb]
                    cur_obj_idx = batch_line_id[id_emb]
                    # if exist self's id
                    self_idx = (cur_selected_idx == cur_obj_idx).nonzero()
                    # only keep 25 out of 26 for cur_selected_idx
                    # if contains self_id
                    if self_idx.size()[0] > 0:
                        self_idx = self_idx[0, 0]
                        # get index
                        if self_idx < cur_selected_idx.size()[0] - 1:
                            cur_selected_idx = torch.cat((cur_selected_idx[0:self_idx], cur_selected_idx[self_idx+1:]), dim=0)
                            cur_selected_idx_prob = torch.cat((cur_selected_idx_prob[0:self_idx], cur_selected_idx_prob[self_idx+1:]), dim=0)
                        else:
                            cur_selected_idx = cur_selected_idx[0:self_idx]
                            cur_selected_idx_prob = cur_selected_idx_prob[0:self_idx]
                    else:
                        cur_selected_idx = cur_selected_idx[:-1]
                        cur_selected_idx_prob = cur_selected_idx_prob[:-1]
                    # clean_selected_idxs
                    if id_emb == 0:
                        clean_selected_idxs = cur_selected_idx.unsqueeze(0)
                        clean_selected_idxs_prob = cur_selected_idx_prob.unsqueeze(0)
                    else:
                        clean_selected_idxs = torch.cat((clean_selected_idxs, cur_selected_idx.unsqueeze(0)), dim=0)
                        clean_selected_idxs_prob = torch.cat((clean_selected_idxs_prob, cur_selected_idx_prob.unsqueeze(0)), dim=0)
            else:
                clean_selected_idxs = selected_idxs[:, :-1]
                clean_selected_idxs_prob = selected_idx_prob[:, :-1]

            # time3 = time.time()
            # print("time3 - time2: ", time3-time2)
            ## In case clean_selected_idxs don't have enough cases for some rel
            assert clean_selected_idxs.size() == clean_selected_idxs_prob.size()
            # in case clean_selected_idxs.size() == torch.Size([0])
            if clean_selected_idxs.size()[1] == 0:
                clean_selected_idxs = (-1) * torch.ones(len(batch_line_id), inflated_num_cases, dtype=torch.long)
                clean_selected_idxs_prob = torch.zeros((len(batch_line_id), inflated_num_cases))
                if self.args.dataset_selection != 0:
                    print('Warning: clean_selected_idxs.size()[1] == 0')
                # raise Exception('How can clean_selected_idxs.size()[1] == 0!')
            elif clean_selected_idxs_prob.size()[1] != 0 and clean_selected_idxs_prob.size()[1] < inflated_num_cases:
                if self.args.dataset_selection != 0 and self.args.subset_selection == -1:
                    print('Warning: clean_selected_idxs_prob.size()[1] < inflated_num_cases: ', clean_selected_idxs_prob.size()[1], inflated_num_cases)
                clean_selected_idxs = torch.cat((clean_selected_idxs, (-1) * torch.ones(len(batch_line_id), inflated_num_cases-clean_selected_idxs.size()[1], dtype=torch.long)), dim=1)
                clean_selected_idxs_prob = torch.cat((clean_selected_idxs_prob, torch.zeros(len(batch_line_id), inflated_num_cases-clean_selected_idxs_prob.size()[1])), dim=1)
            assert clean_selected_idxs.size()[1] == inflated_num_cases
            assert clean_selected_idxs_prob.size()[1] == inflated_num_cases
            ## Get cases using index
            # similar_cases: ['<oReact>\tPersonX plays a ___ in the war\tsad', ...]
            # torch.save(batch_line_id, os.path.join(self.args.output_dir, 'batch_line_id.pt'))
            # torch.save(clean_selected_idxs, os.path.join(self.args.output_dir, 'clean_selected_idxs.pt'))
            # time4 = time.time()
            # print("time4 - time3: ", time4-time3)
            for id_emb in range(len(batch_line_id)):
                if self.args.dataset_selection == 0:
                    # cur_triple: 'rel\tsub\tobj\n'
                    similar_cases, cur_triple = self.get_current_similar_case_conceptnet(batch_line_id[id_emb], clean_selected_idxs[id_emb], data_type=data_type)
                elif self.args.dataset_selection == 1:
                    # cur_triple: 'rel\tsub\tobj\n'
                    similar_cases, cur_triple = self.get_current_similar_case_atomic(batch_line_id[id_emb], clean_selected_idxs[id_emb], data_type=data_type)
                elif self.args.dataset_selection == 2:
                    # cur_triple: 'rel\tsub\tobj\n'
                    similar_cases, cur_triple = self.get_current_similar_case_shakespear_or_e2e(batch_line_id[id_emb], clean_selected_idxs[id_emb], data_type=data_type)
                elif self.args.dataset_selection == 3:
                    similar_cases, cur_triple = self.get_current_similar_case_shakespear_or_e2e(batch_line_id[id_emb], clean_selected_idxs[id_emb], data_type=data_type)
                elif self.args.dataset_selection == 4:
                    similar_cases, cur_triple = self.get_current_similar_case_shakespear_or_e2e(batch_line_id[id_emb], clean_selected_idxs[id_emb], data_type=data_type)

                # count for analysis
                len_similar_cases = len(similar_cases)
                if len_similar_cases not in num_retrieved_cases:
                    num_retrieved_cases[len_similar_cases] = 1
                else:
                    num_retrieved_cases[len_similar_cases] += 1
                # note down similar_cases
                appended_item = '\t\t'.join(similar_cases) + '\n'
                if appended_item == '':
                    raise ValueError
                ttl_similar_cases.append(appended_item)
                ttl_cur_tuple.append(cur_triple)

            idx_for_ttl_similar_cases.append(clean_selected_idxs)
            prob_for_ttl_similar_cases.append(clean_selected_idxs_prob)
            # time5 = time.time()
            # print("time5 - time4: ", time5-time4)


        # Q: .to(torch.long)
        idx_for_ttl_similar_cases = torch.cat(([d.to(torch.long) for d in idx_for_ttl_similar_cases]), dim=0)
        prob_for_ttl_similar_cases = torch.cat(([d for d in prob_for_ttl_similar_cases]), dim=0)
        print(num_retrieved_cases)
        # Get reordered idx_for_ttl_similar_cases, prob_for_ttl_similar_cases and ttl_similar_cases
        # Same order of original bundle
        re_reorder_id = torch.argsort(obj_ori_id_in_batches)
        # obj_line_id_bundleOrder: the line_id (in train set) of the bundle in original bundle orger
        obj_line_id_bundleOrder = obj_line_id[re_reorder_id]
        # idx_for_ttl_similar_cases
        idx_for_ttl_similar_cases = idx_for_ttl_similar_cases[re_reorder_id]
        # prob_for_ttl_similar_cases
        prob_for_ttl_similar_cases = prob_for_ttl_similar_cases[re_reorder_id]
        # ttl_similar_cases_with_bundle_order
        ttl_similar_cases_with_bundle_order = []
        ttl_cur_tuple_with_bundle_order = []
        for ori_id in re_reorder_id:
            if len(ttl_similar_cases[ori_id]) < 10:
                print('ori_id: ', ori_id)
                print('ttl_similar_cases[ori_id]: ', ttl_similar_cases[ori_id])
                print("Warning: ttl_similar_cases[ori_id] might not contain enough information")
                # raise Exception("ttl_similar_cases[ori_id] might not contain enough information")
            ttl_similar_cases_with_bundle_order.append(ttl_similar_cases[ori_id])
            ttl_cur_tuple_with_bundle_order.append(ttl_cur_tuple[ori_id])

        with open(save_dir, 'w') as f:
            f.writelines(ttl_similar_cases_with_bundle_order)
        # time6 = time.time()
        # print("time6 - time5: ", time6-time5)

        # idx_for_ttl_similar_cases: [len_bundle, num_cases]
        # prob_for_ttl_similar_cases: [len_bundle, num_cases]
        # ttl_similar_cases_with_bundle_order: [len_bundle]
        # ttl_cur_tuple_with_bundle_order: [len_bundle]
        return idx_for_ttl_similar_cases, prob_for_ttl_similar_cases, ttl_similar_cases_with_bundle_order, ttl_cur_tuple_with_bundle_order, obj_line_id_bundleOrder

    # idx_for_ttl_similar_cases: [len_bundle, num_cases]
    # OUTPUT: encoded_cases: [encoded_cases_gene, encoded_cases_retr]
    # encoded_cases_gene: [doc_gene_input_ids, doc_gene_attention_mask, doc_gene_lm_labels]
    # encoded_cases_retr: [doc_retr_cases_input_ids, doc_retr_cases_attention_mask, doc_retr_cases_segment_ids]
    #     doc_gene_input_ids: [len_bundle, n_doc, cases_per_doc * input_len_gene]
    #     doc_retr_cases_input_ids: [len_bundle, n_doc, cases_per_doc, input_len_retr]
    # F: encoded_cases_retr contains obj
    def get_and_save_encoded_cases(self, idx_for_ttl_similar_cases, num_cases, n_doc, data_type):
        # tensor_datasets: [#train(gene_doc_input_ids, gene_doc_attention_mask, gene_doc_lm_labels, \
        #                          gene_cur_input_ids, gene_cur_attention_mask, gene_cur_lm_labels, \
        #                          data_idx_ids, rel_collection, \
        #                          retr_doc_input_ids, retr_doc_attention_mask, retr_doc_segment_ids, \
        #                          retr_input_ids, retr_attention_mask, retr_segment_ids), #eval(), #test()]
        tensor_datasets_train = self.train_tensor_dataset
        # print('INFO: tensor_datasets_train[0][-1]: ', tensor_datasets_train[0][-1])
        if num_cases == None:
            num_cases = self.num_cases
        if n_doc == None:
            n_doc = self.n_doc
        if data_type == 'train':
            save_dir = self.path_retrieved_encoded_cases
        elif data_type == 'eval':
            save_dir = self.path_retrieved_encoded_cases_eval
        elif data_type == 'test':
            save_dir = self.path_retrieved_encoded_cases_test
        else:
            raise Exception("Wrong data_type: ", data_type)
        # get real save_dir that is with sequence number
        save_dir = save_dir.split('.')
        save_dir[0] += '_' + str(self.cnt_next_bundle)
        save_dir = '.'.join(save_dir)
        self.cnt_next_bundle = None

        assert num_cases % n_doc == 0
        assert num_cases == idx_for_ttl_similar_cases.size()[1]
        cases_per_doc = num_cases // n_doc
        len_bundle = idx_for_ttl_similar_cases.size()[0]

        # doc_idx_for_ttl_similar_cases: [len_bundle, n_doc, cases_per_doc]
        doc_idx_for_ttl_similar_cases = idx_for_ttl_similar_cases.view(len_bundle, n_doc, cases_per_doc)

        ### encoded_cases_gene
        input_len_gene_doc = tensor_datasets_train[0].size()[1]
        ## doc_gene_input_ids: [len_bundle, n_doc, cases_per_doc, input_len_gene_doc]
        doc_gene_input_ids = tensor_datasets_train[0][doc_idx_for_ttl_similar_cases]
        # doc_gene_input_ids: [len_bundle, n_doc, cases_per_doc * input_len_gene_doc]
        doc_gene_input_ids = doc_gene_input_ids.view(len_bundle, n_doc, cases_per_doc * input_len_gene_doc)
        ## doc_gene_attention_mask: [len_bundle, n_doc, cases_per_doc, input_len_gene_doc]
        doc_gene_attention_mask = tensor_datasets_train[1][doc_idx_for_ttl_similar_cases]
        # doc_gene_attention_mask: [len_bundle, n_doc, cases_per_doc * input_len_gene_doc]
        doc_gene_attention_mask = doc_gene_attention_mask.view(len_bundle, n_doc, cases_per_doc * input_len_gene_doc)
        ## doc_gene_lm_labels: [len_bundle, n_doc, cases_per_doc * input_len_gene_doc]
        doc_gene_lm_labels = tensor_datasets_train[2][doc_idx_for_ttl_similar_cases]
        doc_gene_lm_labels = doc_gene_lm_labels.view(len_bundle, n_doc, cases_per_doc * input_len_gene_doc)
        encoded_cases_gene = [doc_gene_input_ids, doc_gene_attention_mask, doc_gene_lm_labels]

        ### encoded_cases_retr
        # input_len_retr = tensor_datasets_train[-3].size()[1]
        ## doc_retr_cases_input_ids: [len_bundle, n_doc, cases_per_doc, input_len_retr]
        doc_retr_cases_input_ids = tensor_datasets_train[-6][doc_idx_for_ttl_similar_cases]
        ## doc_retr_cases_attention_mask: [len_bundle, n_doc, cases_per_doc, input_len_retr]
        doc_retr_cases_attention_mask = tensor_datasets_train[-5][doc_idx_for_ttl_similar_cases]
        ## doc_retr_cases_segment_ids: [len_bundle, n_doc, cases_per_doc, input_len_retr]
        doc_retr_cases_segment_ids = tensor_datasets_train[-4][doc_idx_for_ttl_similar_cases]
        encoded_cases_retr = [doc_retr_cases_input_ids, doc_retr_cases_attention_mask, doc_retr_cases_segment_ids]

        ### encoded_cases
        encoded_cases = [encoded_cases_gene, encoded_cases_retr]
        while os.path.exists(save_dir):
            print('Warning: last encoded_cases still exists')
            time.sleep(5)
        try:
            torch.save(encoded_cases, save_dir)
        except:
            time.sleep(5)
            print("Exception occurs when saving encoded_cases")
            torch.save(encoded_cases, save_dir)
        return encoded_cases

    # num_cases: num of cases to get
    # return:
    # same rel, positive label
    # output:
    # similar_cases: ['<oReact>\tPersonX plays a ___ in the war\tsad', ...]
    # cur_triple: 'rel\tsub\tobj\n'
    def get_current_similar_case_atomic(self, cur_id, selected_idxs, data_type):
        if not data_type:
            data_type = self.data_type
        else:
            data_type = data_type
        # cur_lines, to get cur_rel
        if data_type == 'train':
            if self.args.if_use_full_memory_store_while_subset:
                cur_lines = self.train_subset_lines
            else:
                cur_lines = self.train_lines
        elif data_type == 'eval':
            if self.args.if_use_full_memory_store_while_subset:
                cur_lines = self.val_subset_lines
            else:
                cur_lines = self.val_lines
        elif data_type == 'test':
            if self.args.if_use_full_memory_store_while_subset:
                cur_lines = self.test_subset_lines
            else:
                cur_lines = self.test_lines
        # print("len(cur_lines): {}, len(self.train_lines): {}".format(len(cur_lines), len(self.train_lines)))
        # cur_data, cur_rel
        try:
            cur_d = cur_lines[cur_id]
        except:
            print("len(cur_lines): ", len(cur_lines))
            print("cur_id: ", cur_id)
            raise Exception
        cur_triple = '\t'.join([cur_d[1], cur_d[0], cur_d[2]]) + '\n'
        cur_rel = cur_d[1]

        similar_cases = []
        # print('len(self.train_lines)', len(self.train_lines))
        # print('selected_idxs', selected_idxs)
        assert len(self.train_lines) > 2
        # print('selected_idxs.size()', selected_idxs.size())
        cnt_mismatch_rel = 0
        for id in selected_idxs:
            selected_d = self.train_lines[id]
            selected_rel = selected_d[1]
            if selected_rel != cur_rel:
                if self.args.subset_selection < 0:
                    print('INFO: selected_rel != cur_rel')
                    # print("selected_idxs:", selected_idxs)
                    print("selected_rel: {}, cur_rel: {}".format(selected_rel, cur_rel))
                    # torch.save(selected_idxs, os.path.join(self.args.output_dir, 'selected_idxs.pt'))
                    # torch.save(cur_d, os.path.join(self.args.output_dir, 'cur_d.pt'))
                    cnt_mismatch_rel += 1
                continue
            # ['<oReact>\tPersonX plays a ___ in the war\tsad', ...]
            similar_cases.append('\t'.join([selected_d[1], selected_d[0], selected_d[2]]))
            if len(similar_cases) == self.num_cases * self.times_more_num_cases_to_retrieve:
                break
        # print("len(similar_cases): {}, self.num_cases * self.times_more_num_cases_to_retrieve: {}".format(len(similar_cases), self.num_cases * self.times_more_num_cases_to_retrieve))
        if cnt_mismatch_rel > 0:
            print("cnt_mismatch_rel: ", cnt_mismatch_rel)
        return similar_cases, cur_triple

    # num_cases: num of cases to get
    # return:
    # same rel, positive label
    # ['IsA\ttool\tobject', ...]
    # cur_triple: 'rel\tsub\tobj\n'
    def get_current_similar_case_conceptnet(self, cur_id, selected_idxs, data_type):
        # cur_lines, to get cur_rel
        # if data_type == 'train':
        #     cur_lines = self.train_lines
        # elif data_type == 'eval':
        #     cur_lines = self.val_lines
        # elif data_type == 'test':
        #     cur_lines = self.test_lines
        if data_type == 'train':
            if self.args.if_use_full_memory_store_while_subset:
                cur_lines = self.train_subset_lines
            else:
                cur_lines = self.train_lines
        elif data_type == 'eval':
            if self.args.if_use_full_memory_store_while_subset:
                cur_lines = self.val_subset_lines
            else:
                cur_lines = self.val_lines
        elif data_type == 'test':
            if self.args.if_use_full_memory_store_while_subset:
                cur_lines = self.test_subset_lines
            else:
                cur_lines = self.test_lines
        # cur_data, cur_rel
        cur_d = cur_lines[cur_id].strip().split('\t')
        cur_triple = '\t'.join(cur_d[0:3]) + '\n'
        cur_rel = cur_d[0]

        similar_cases = []
        for id in selected_idxs:
            selected_d = self.train_lines[id].strip()
            selected_d_split = selected_d.split('\t')
            selected_rel = selected_d_split[0]
            # TOCHECK: do not use data whose label is not positibe
            if float(selected_d_split[-1]) == 0:
                continue
            if selected_rel != cur_rel:
                # print("selected_rel != cur_rel")
                # Q: we do not use continue if rerank == 1 or 2, when we need at least one case in similar_cases
                if self.args.rerank_selection == 1 or self.args.rerank_selection == 2:
                    pass
                else:
                    continue
            # ['IsA\ttool\tobject', ...]
            similar_cases.append('\t'.join(selected_d_split[0:3]))
            if len(similar_cases) == self.num_cases * self.times_more_num_cases_to_retrieve:
                break
        assert len(similar_cases) == self.num_cases * self.times_more_num_cases_to_retrieve
        return similar_cases, cur_triple

    # OUTPUT:
    # similar_cases: ['rel\tsub\tobj', ...]
    # cur_triple: 'rel\tsub\tobj\n'
    def get_current_similar_case_shakespear_or_e2e(self, cur_id, selected_idxs, data_type):
        # cur_lines, to get cur_rel
        # if data_type == 'train':
        #     cur_lines = self.train_lines
        # elif data_type == 'eval':
        #     cur_lines = self.val_lines
        # elif data_type == 'test':
        #     cur_lines = self.test_lines
        if data_type == 'train':
            if self.args.if_use_full_memory_store_while_subset:
                cur_lines = self.train_subset_lines
            else:
                cur_lines = self.train_lines
        elif data_type == 'eval':
            if self.args.if_use_full_memory_store_while_subset:
                cur_lines = self.val_subset_lines
            else:
                cur_lines = self.val_lines
        elif data_type == 'test':
            if self.args.if_use_full_memory_store_while_subset:
                cur_lines = self.test_subset_lines
            else:
                cur_lines = self.test_lines
        # cur_data, cur_rel
        # Do not use strip here
        try:
            cur_d = cur_lines[cur_id].split('\t')
        except:
            print("WARNING: Can't find cur_lines[cur_id]")
            print('cur_id: ', cur_id)
            print('len(cur_lines): ', len(cur_lines))
            cur_d = cur_lines[cur_id].split('\t')
        cur_rel = cur_d[0]
        cur_triple = '\t'.join(cur_d[0:3]) + '\n'

        similar_cases = []
        for id in selected_idxs:
            # Do not use strip here
            try:
                selected_d = self.train_lines[id]
            except:
                print("len(self.train_lines): ", len(self.train_lines))
                print("id: ", id)
                raise Exception
            selected_d_split = selected_d.split('\t')
            # newly added
            selected_d_split = [i.strip('\n') for i in selected_d_split]
            selected_rel = selected_d_split[0]
            # # TOCHECK: do not use data whose label is not positibe
            # if float(selected_d_split[-1]) == 0:
            #     continue
            if selected_rel != cur_rel:
                print("selected_rel != cur_rel")
                # continue
                raise Exception('All relation should be the same')
            # ['IsA\ttool\tobject', ...]
            similar_cases.append('\t'.join(selected_d_split[0:3]))
            if len(similar_cases) == self.num_cases * self.times_more_num_cases_to_retrieve:
                break
        return similar_cases, cur_triple

    # INPUT:
    # model_type: 'bert'
    # data_type: 'train' or 'eval' or 'test'
    # sample_ids: sorted
    # use_only_sub_rel_for_retrieval: for args.if_double_retrieval, different from args.use_only_sub_rel_for_retrieval
    # OUTPUT:
    # builded_embedding
    # rel_for_embedding: id for rel
    # line_id_for_embedding: line id
    def get_embed_for_sample_CKB(self, sample_ids, use_only_sub_rel_for_retrieval=False):
        if not self.data_type == 'train':
            raise Exception("Train set is expected for building sample_CKB")

        self.model_doc.eval()
        dataloader = self.get_sampled_ckb_data_loader(sample_ids)

        builded_embedding = None
        rel_for_embedding = None
        line_id_for_embedding = None
        # If use_only_sub_rel_for_retrieval == True, it means arg.if_double_retrieval == True and args.use_only_sub_rel_for_retrieval == False
        #       And here batch data are including sub, rel and obj, which means we need to truncate obj
        if use_only_sub_rel_for_retrieval == True:
            assert self.args.if_double_retrieval == True and self.args.use_only_sub_rel_for_retrieval == False
            sep_token_id = self.tokenizer_retriever_doc.encode(self.tokenizer_retriever_doc.sep_token)[1]

        # batch:    #train(gene_doc_input_ids, gene_doc_attention_mask, gene_doc_lm_labels, \
                        #  gene_cur_input_ids, gene_cur_attention_mask, gene_cur_lm_labels, \
#                          data_idx_ids, rel_collection, \
#                          retr_doc_input_ids, retr_doc_attention_mask, retr_doc_segment_ids, \
#                          retr_input_ids, retr_attention_mask, retr_segment_ids)
        for batch_idx, batch in enumerate(dataloader):
            # rel_each_batch: torch.Size([32])
            rel_each_batch = batch[-7]
            # data_idx_id_batch: torch.Size([32])
            data_idx_id_batch = batch[-8]
            # retr_input_ids, retr_attention_mask, retr_segment_ids = batch
            # batch = tuple(t.to(device) for t in batch[-6:-3])
            batch = batch[-6:-3]
            # truncated batch if self.args.if_double_retrieval == True and here use_only_sub_rel_for_retrieval == True
            # Q: the last [SEP] token might be missed in this calculation
            if use_only_sub_rel_for_retrieval:
                # 1. get rid of e2 part of the data
                # [simi_batch_size, max_e1 + max_r + max_e2]
                if batch_idx == 0:
                    print("batch[0].size(): ", batch[0].size())
                assert batch[0].size()[-1] == self.args.max_e1 + self.args.max_r + self.args.max_e2
                assert batch[1].size()[-1] == self.args.max_e1 + self.args.max_r + self.args.max_e2
                assert batch[2].size()[-1] == self.args.max_e1 + self.args.max_r + self.args.max_e2
                batch = [t[:, :(self.args.max_e1 + self.args.max_r)] for t in batch]
                # 2. Add [SEP] token in the end
                sep_for_input_id = sep_token_id * torch.ones((batch[0].size()[0], 1), dtype=torch.long)
                sep_for_attention_mask = torch.ones((batch[0].size()[0], 1))
                sep_for_segment_id = torch.zeros((batch[0].size()[0], 1), dtype=torch.long)
                # concat batch
                batch[0] = torch.cat((batch[0], sep_for_input_id), dim=1)
                batch[1] = torch.cat((batch[1], sep_for_attention_mask), dim=1)
                batch[2] = torch.cat((batch[2], sep_for_segment_id), dim=1)
            batch = tuple(t.to(device) for t in batch)
                # print("INFO: batch successfully truncated!")
            batch_size = batch[0].size()[0]
            # if batch_idx == 0 and data_type == 'train':
            #     print('batch_size: ', batch_size)
            #     print(torch.cuda.memory_summary(device))
            # input_ids, _, input_mask, token_type_ids, data_idx_ids = ori_batch
            with torch.no_grad():
                # embedding: [batch_size, 768]
                # D:
                embedding = batch_get_embed(self.model_doc, self.args.retriever_model_type, batch)
                if batch_idx == 0:
                    builded_embedding = embedding
                    rel_for_embedding = rel_each_batch
                    line_id_for_embedding = data_idx_id_batch
                else:
                    builded_embedding = torch.cat((builded_embedding, embedding), 0)
                    rel_for_embedding = torch.cat((rel_for_embedding, rel_each_batch), 0)
                    line_id_for_embedding = torch.cat((line_id_for_embedding, data_idx_id_batch), 0)
        # rel_for_embedding: torch.Size([709996])
        assert rel_for_embedding.size()[0] == builded_embedding.size()[0]
        assert rel_for_embedding.size()[0] == line_id_for_embedding.size()[0]
        # save
        torch.save([builded_embedding.cpu(), rel_for_embedding, line_id_for_embedding], self.sample_CKB_train_embed_dir)
        return builded_embedding, rel_for_embedding, line_id_for_embedding

    # return value: whether we should continue evaluation mode
    def wait_get_remove_bundle_eval(self, data_type):
        print('Waiting for next_bundle_eval...')
        if data_type == 'eval':
            path_signal_file_if_eval_or_test = self.path_signal_file_if_eval
            # path_next_bundle_eval_or_test = self.path_next_bundle_eval
            next_bundle_file_prompt = 'next_bundle_eval'
        elif data_type == 'test':
            path_signal_file_if_eval_or_test = self.path_signal_file_if_test
            # path_next_bundle_eval_or_test = self.path_next_bundle_test
            next_bundle_file_prompt = 'next_bundle_test'
        else:
            raise Exception('Wrong data type')
        start_waiting_time = time.time()
        # while(not os.path.exists(path_next_bundle_eval_or_test)):
        while True:
            # in case evaluation mode shold end now
            if not os.path.exists(path_signal_file_if_eval_or_test):
                # False means should end evaluation model now
                return False
            possible_other_next_bundle_files = \
                [i for i in os.listdir(self.args.output_dir) if i.startswith(next_bundle_file_prompt)]
            if len(possible_other_next_bundle_files) > 0:
                assert len(possible_other_next_bundle_files) == 1
                # to get self.cnt_next_bundle
                self.cnt_next_bundle = int(possible_other_next_bundle_files[0].split('.')[0].replace(next_bundle_file_prompt+'_', ''))
                # to get the exact path_next_bundle
                path_next_bundle_eval_or_test = os.path.join(self.args.output_dir, possible_other_next_bundle_files[0])
                break
            else:
                time.sleep(5)
        print('--- Waiting for next_bundle_eval: %s ---' % (time.time() - start_waiting_time))
        # try to avoid Exception occruing here.
        time.sleep(10)
        while True:
            try:
                # 'train', 'eval' and 'test' share using self.next_bundle
                self.next_bundle = torch.load(path_next_bundle_eval_or_test)
                break
            except:
                print("Exception occurs during loading next_bundle_eval_or_test, wait to load it again...")
                time.sleep(5)
                # self.next_bundle = torch.load(self.path_next_bundle_eval_or_test)

        os.remove(path_next_bundle_eval_or_test)
        assert not os.path.exists(path_next_bundle_eval_or_test)
        # True means this function works normally
        print('Loaded next_bundle_eval_or_test.')
        return True

    def evaluation_mode(self, data_type):
        print('Evaluation mode begin...')
        # paths
        if data_type == 'eval':
            path_signal_file_if_eval_or_test = self.path_signal_file_if_eval
            path_next_bundle_eval_or_test = self.path_next_bundle_eval
            path_model_retriever_eval_or_test = self.path_model_retriever_eval
            path_retriever_doc_eval_or_test = self.path_retriever_doc_eval
        elif data_type == 'test':
            path_signal_file_if_eval_or_test = self.path_signal_file_if_test
            path_next_bundle_eval_or_test = self.path_next_bundle_test
            path_model_retriever_eval_or_test = self.path_model_retriever_test
            path_retriever_doc_eval_or_test = self.path_retriever_doc_test
        else:
            raise Exception('Wrong data type')

        ## change n_doc for evaluation mode
        n_doc = 1
        num_cases = int(self.num_cases / self.n_doc)
        print('n_doc:', n_doc, 'num_cases:', num_cases)

        counter_while_loop = 0
        while os.path.exists(path_signal_file_if_eval_or_test):
            print('Retrieval times: %d'%counter_while_loop)
            start_time = time.time()
            if_continue_eva_mode = self.wait_get_remove_bundle_eval(data_type=data_type)
            if not if_continue_eva_mode:
                break
            if counter_while_loop == 0:
                # only load retriever once since it's unchanged
                self.load_remove_retriever(data_type=data_type)
                print("--- load bundle and retriever: %s seconds ---" % (time.time() - start_time))
                # only need to get embedding for train cases once since it's unchanged
                if self.args.if_only_embed_ckb_once:
                    assert os.path.exists(self.path_sample_ckb_dict)
                    sample_ckb_dict = torch.load(self.path_sample_ckb_dict)
                    print('sample_ckb_dict loaded successfully')
                    if self.args.if_double_retrieval:
                        assert os.path.exists(self.path_sample_ckb_dict_subRel)
                        sample_ckb_dict_subRel = torch.load(self.path_sample_ckb_dict_subRel)
                        print('sample_ckb_dict_subRel loaded successfully')
                else:
                    start_time = time.time()
                    sample_ckb_dict = self.get_sample_caseKB_embed_dict(data_type=data_type)
                    if args.if_double_retrieval:
                        sample_ckb_dict_subRel = self.get_sample_caseKB_embed_dict(data_type=data_type, use_only_sub_rel_for_retrieval=True)
                    print("--- Embed CKB: %s seconds ---" % (time.time() - start_time))

            # use self.model to build embed for self.next_bundle
            start_time = time.time()
            bundle_embed_rlts, rel_ttl_batch = self.get_bundle_embed_list(model_type=self.args.retriever_model_type)
            if counter_while_loop == 0:
                print("--- Embed bundle: %s seconds ---" % (time.time() - start_time))

            start_time = time.time()
            if not self.args.if_double_retrieval:
                self.get_and_save_encoded_cases_and_check_result(sample_ckb_dict, bundle_embed_rlts, \
                    rel_ttl_batch, counter_while_loop, num_cases=num_cases, n_doc=n_doc, data_type=data_type)
            else:
                # if args.if_double_retrieval == True, sample_ckb_dict_subRel should also be passed to the function
                self.get_and_save_encoded_cases_and_check_result(sample_ckb_dict, bundle_embed_rlts, \
                    rel_ttl_batch, counter_while_loop, num_cases=num_cases, n_doc=n_doc, data_type=data_type, \
                    sample_ckb_dict_subRel=sample_ckb_dict_subRel)
            if counter_while_loop == 0:
                print("--- Finish MIPS and get encoded_cases: %s seconds ---" % (time.time() - start_time))
            counter_while_loop += 1
        # delete retriever since
        # if os.path.exists(path_model_retriever_eval_or_test) or os.path.exists(path_retriever_doc_eval_or_test):
        #     os.remove(path_model_retriever_eval_or_test)
        #     os.remove(path_retriever_doc_eval_or_test)
        #     print("Error: path_model_retriever_eval_or_test or path_retriever_doc_eval_or_test still exists")
        assert not os.path.exists(path_next_bundle_eval_or_test)
        assert not os.path.exists(path_model_retriever_eval_or_test)
        assert not os.path.exists(path_retriever_doc_eval_or_test)
        # save eval or test cases
        if data_type == 'eval':
            with open(self.path_ttl_similar_cases_eval, 'w') as f:
                f.writelines(self.ttl_similar_cases_eval)
            self.ttl_similar_cases_eval = []
            print('Successfully saved and reset self.ttl_similar_cases_eval')
        elif data_type == 'test':
            with open(self.path_ttl_similar_cases_test, 'w') as f:
                f.writelines(self.ttl_similar_cases_test)
            self.ttl_similar_cases_test = []
            print('Successfully saved and reset self.ttl_similar_cases_test')
        else:
            raise Exception
        # assert not os.path.exists(self.path_next_bundle)
        print('Evaluation model done!')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default=None, type=str, required=False, help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--dataStore_dir", default=None, type=str, required=False, help="The home directory of zonglin.")
    parser.add_argument("--train_dataset", type=str, nargs="+", default=["./Data/atomic/v4_atomic_trn.csv"])
    parser.add_argument("--eval_dataset", type=str, nargs="+", default=["./Data/atomic/v4_atomic_dev.csv"])
    parser.add_argument("--test_dataset", type=str, nargs="+", default=["./Data/atomic/v4_atomic_tst.csv"])
    # add for Shakespeare TST task
    parser.add_argument("--path_dataset", type=str, default="")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--train_batch_size", type=int, default=32)
    parser.add_argument("--eval_batch_size", type=int, default=32)
    # dataset_selection: 0: conceptnet 1: atomic 2: Shakespeare text style transfer
    # 3: e2e (table2text) 4: sentiment sentence classification dataset
    parser.add_argument("--dataset_selection", type=int, default=0)
    # parser.add_argument("--model_type", type=str, default="dpr")
    parser.add_argument("--generator_model_type", type=str, default="gpt2-lmhead",
                        help="model type: bart-base/t5-base/gpt2-lmhead/...")
    parser.add_argument("--retriever_model_type", type=str, default="dpr",
                        help="model type: dpr/bert/...")
    parser.add_argument("--max_additional_cases", type=int, default=150)
    parser.add_argument("--max_e1", type=int, default=24)
    parser.add_argument("--max_r", type=int, default=10)
    parser.add_argument("--max_e2", type=int, default=36)
    parser.add_argument("--num_cases", type=int, default=25)
    parser.add_argument("--n_doc", type=int, default=3)
    parser.add_argument("--simi_batch_size", type=int, default=64)
    parser.add_argument("--sample_size", type=int, default=20000)
    parser.add_argument("--num_steps", type=int, default=500)
    parser.add_argument("--use_obj_for_retrieval", action="store_true", help="if using obj for retrieval (get embedding and similarity score)")
    parser.add_argument("--use_only_sub_rel_for_retrieval", action="store_true", help="if only using sub for retrieval (get embedding and similarity score)")
    parser.add_argument("--if_only_embed_ckb_once", action="store_true", help="if use frozen retriever")
    parser.add_argument("--if_use_relation_for_shakes", action="store_true", help="Whether use relation for shakes dataset (Shakespeare's style is )")
    parser.add_argument("--use_special_tokens_to_split_retrieved_cases", action="store_true", help="<split_cases> and <split_source/target>")
    parser.add_argument("--rerank_selection", type=int, default=1, help="rerank method selection; 0 is for no reranking; 1 is to delete the retrieved cases that share the same source with current query; 2 is the same as 1 except that some small proportion of train instances can use itself in the retrieved cases")
    parser.add_argument("--possibility_add_cur_tuple_to_its_retrieved_cases", type=float, default=0.9, help="possibility that a tuple add itself to its own retrieval")
    # subset_selection: 0~6, -1 means not using subset
    parser.add_argument("--subset_selection", type=int, default=-1)
    parser.add_argument("--if_not_adding_special_relation_tokens", action="store_true", help="not adding <oReact> for instance")
    parser.add_argument("--if_use_full_memory_store_while_subset", action="store_true", help="if use full memory store during retrieval (no matter if using subset)")
    parser.add_argument("--subset_selection_while_use_full_memory_store", type=int, default=-1, help="-1 when not using subset&full MS; 0~6 when using subset&full MS")
    parser.add_argument("--random_retrieval", action="store_true", help="use totally random retrieval")
    parser.add_argument("--larger_range_to_select_retrieval_randomly_from", type=int, default=-1, help="select cases randomly from top(larger_range_to_select_retrieval_randomly_from); -1 when don't use it")
    parser.add_argument("--if_without_none", action="store_true", help="You do NOT need to include it in command line, as it will adjust itself in the following code; if not using none data in atomic; will use different train/val/tst data; ")
    parser.add_argument("--num_sample", type=int, default=1, help="the nth time of sampling data to use")
    parser.add_argument("--if_use_nshot_data", action="store_true", help="The old version of data doesn't try to maintain nshot but only keep the same number of total few-shot data; this new version of data try to maintain nshot")
    parser.add_argument("--if_double_retrieval", action="store_true", help="if use sub/rel and sub/rel for retrieval once, filter ones with least similarity, then use sub/rel and sub/rel/obj for retrieval the second time, and adopt the top retrievals as in-context demonstrations")
    parser.add_argument("--filter_ratio", type=float, default=0.34, help="the initial filter ratio for if_double_retrieval")
    parser.add_argument("--if_reverse_order_demonstrations", action="store_true", help="if reverse the order of demonstrations in input_ids (we try to reverse it becase recency bias)")
    args = parser.parse_args()


    assert not (args.random_retrieval and (args.larger_range_to_select_retrieval_randomly_from != -1))
    assert not (args.if_use_full_memory_store_while_subset and args.subset_selection != -1)
    # if use args.if_double_retrieval, the retrieval method is fixed; Can't use use_only_sub_rel_for_retrieval, which might mess it up
    assert not (args.if_double_retrieval and args.use_only_sub_rel_for_retrieval)
    assert not (args.if_double_retrieval and args.random_retrieval)
    if not args.if_use_full_memory_store_while_subset:
        assert args.subset_selection_while_use_full_memory_store == -1
    if args.if_use_full_memory_store_while_subset:
        assert args.subset_selection_while_use_full_memory_store != -1
    # change args according to args.dataset
    if args.dataset_selection == 0:
        args.train_dataset = ["./Data/conceptnet/train100k_CN_sorted.txt"]
        args.eval_dataset = ["./Data/conceptnet/dev1_CN_sorted.txt"]
        args.test_dataset = ["./Data/conceptnet/test_CN_sorted.txt"]
        args.if_without_none = False
    elif args.dataset_selection == 1:
        args.max_e1 = 25
        args.max_r = 15
        args.max_e2 = 38
        # Q: changed from 200 to 250: 8/21/2021: 11:50 p.m.
        args.max_additional_cases = 250
        # if_without_none: if not using none data in atomic; will use different train/val/tst data
        # if if_without_none == True, both data and lines should be updated
        args.if_without_none = True
        print("INFO: using atomic data without 'None' tuples")
    elif args.dataset_selection == 2:
        # args.path_dataset = "/home/zy223/CBR/Data/Shakes_data/"
        args.train_dataset = ["./Data/shakes/Shakes_processed_lines/train_lines.txt"]
        args.eval_dataset = ["./Data/shakes/Shakes_processed_lines/eval_lines.txt"]
        args.test_dataset = ["./Data/shakes/Shakes_processed_lines/test_lines.txt"]
        args.max_e1 = 130
        if args.if_use_relation_for_shakes:
            args.max_r = 6
        else:
            args.max_r = 2
        args.max_e2 = 140
        args.max_additional_cases = 500
    elif args.dataset_selection == 3:
        args.train_dataset = ["./Data/e2e/train_lines.txt"]
        args.eval_dataset = ["./Data/e2e/eval_lines.txt"]
        args.test_dataset = ["./Data/e2e/test_lines.txt"]
        args.max_e1 = 60
        args.max_r = 2
        args.max_e2 = 95
        args.max_additional_cases = 400
    elif args.dataset_selection == 4:
        args.train_dataset = ["./Data/sentiment/splitted/train_lines.txt"]
        args.eval_dataset = ["./Data/sentiment/splitted/eval_lines.txt"]
        args.test_dataset = ["./Data/sentiment/splitted/test_lines.txt"]
        args.max_e1 = 110
        args.max_r = 2
        args.max_e2 = 30
        args.max_additional_cases = 400
        assert args.subset_selection >= -1 and args.subset_selection <= 3
    elif not args.dataset_selection == 1:
        raise Exception("Not supported dataset_selection")

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
    print(args)

    # set_seed(args.seed)
    n_gpu = torch.cuda.device_count()
    logger.info("device: {}, n_gpu {}".format(device, n_gpu))
    if not n_gpu >= 1:
        raise Exception("n_gpu is not enough: {}".format(n_gpu))
    while not os.path.exists(args.output_dir):
        time.sleep(5)

    # get embedding and cases
    # loaded tensor_datasets
    builder = embedding_builder(args, args.retriever_model_type, num_cases=args.num_cases, n_doc=args.n_doc, simi_batch_size=args.simi_batch_size, sample_size_total=args.sample_size, num_steps=args.num_steps)

    counter_while_loop = 0
    while True:
        print('Retrieval times: %d'%counter_while_loop)
        start_time = time.time()
        if_continue = builder.wait_get_remove_bundle()
        if not if_continue:
            break
        builder.load_remove_retriever()
        if counter_while_loop == 0:
            print("--- load bundle and retriever: %s seconds ---" % (time.time() - start_time))

        ## Embed CKB
        start_time = time.time()
        # only build sample_ckb_dict in the first time
        if args.if_only_embed_ckb_once:
            if counter_while_loop == 0:
                # sample_ckb_dict
                if os.path.exists(builder.path_sample_ckb_dict):
                    sample_ckb_dict = torch.load(builder.path_sample_ckb_dict)
                    print('INFO: sample_ckb_dict loaded successfully before training: ', builder.path_sample_ckb_dict)
                else:
                    # if args.if_use_full_memory_store_while_subset:
                    #     raise Exception("Should generate sample_CKB using full train data")
                    print("Building sample_ckb_dict...")
                    sample_ckb_dict = builder.get_sample_caseKB_embed_dict()
                    torch.save(sample_ckb_dict, builder.path_sample_ckb_dict)
                    print("sample_ckb_dict successfully generated: ", builder.path_sample_ckb_dict)
                # sample_ckb_dict_subRel, if args.if_double_retrieval == True
                if args.if_double_retrieval:
                    if os.path.exists(builder.path_sample_ckb_dict_subRel):
                        sample_ckb_dict_subRel = torch.load(builder.path_sample_ckb_dict_subRel)
                        print('INFO: sample_ckb_dict_subRel loaded successfully before training: ', builder.path_sample_ckb_dict_subRel)
                    else:
                        print("Building sample_ckb_dict_subRel...")
                        sample_ckb_dict_subRel = builder.get_sample_caseKB_embed_dict(use_only_sub_rel_for_retrieval=True)
                        torch.save(sample_ckb_dict_subRel, builder.path_sample_ckb_dict_subRel)
                        print("sample_ckb_dict_subRel successfully generated: ", builder.path_sample_ckb_dict_subRel)
        else:
            # if args.if_use_full_memory_store_while_subset:
            #     raise Error("Should generate sample_CKB using full train data")
            # sample_ckb_dict
            sample_ckb_dict = builder.get_sample_caseKB_embed_dict()
            # sample_ckb_dict_subRel, if args.if_double_retrieval == True
            if args.if_double_retrieval:
                sample_ckb_dict_subRel = builder.get_sample_caseKB_embed_dict(use_only_sub_rel_for_retrieval=True)
        bundle_embed_rlts, rel_ttl_batch = builder.get_bundle_embed_list(model_type=builder.args.retriever_model_type)
        if counter_while_loop == 0:
            print("--- Embed CKB and bundle: %s seconds ---" % (time.time() - start_time))

        start_time = time.time()
        if not args.if_double_retrieval:
            builder.get_and_save_encoded_cases_and_check_result(sample_ckb_dict, bundle_embed_rlts, rel_ttl_batch, counter_while_loop)
        else:
            # if args.if_double_retrieval == True, sample_ckb_dict_subRel should also be passed to the function
            builder.get_and_save_encoded_cases_and_check_result(sample_ckb_dict, bundle_embed_rlts, rel_ttl_batch, counter_while_loop, sample_ckb_dict_subRel=sample_ckb_dict_subRel)
        if counter_while_loop == 0:
            print("--- Finish MIPS and get encoded_cases: %s seconds ---" % (time.time() - start_time))
        counter_while_loop += 1
    print('Finished!')



if __name__ == "__main__":
    main()
