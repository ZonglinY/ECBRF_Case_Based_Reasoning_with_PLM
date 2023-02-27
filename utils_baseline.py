import os, copy
import numpy as np
import json
import torch
import torch.nn.functional as F
from nltk.tokenize import word_tokenize, RegexpTokenizer

# np.random.seed(10)
np.random.seed(11)


# INPUT
#   train/val/test_set: [[text, "", "positive/negative", label, line_id]]
def get_data_lines_using_sentimentSentence_dataset_for_retriever(train_set, val_set, test_set, splitted_data_dir="./Data/sentiment/splitted/"):
    def get_data_lines_from_one_split_set(data_set, data_type):
        assert data_type == "train" or data_type == 'eval' or data_type == 'test'
        data_write_dir = os.path.join(splitted_data_dir, data_type+"_lines.txt")
        processed_data_to_write = []
        for cur_id in range(len(data_set)):
            assert len(data_set[cur_id]) == 5
            cur_e1, cur_rel, cur_e2, cur_label, cur_line_id = data_set[cur_id]
            cur_text = cur_rel + '\t' + cur_e1 + '\t' + cur_e2 + '\n'
            processed_data_to_write.append(cur_text)
        with open(data_write_dir, 'w') as f:
            f.writelines(processed_data_to_write)
    get_data_lines_from_one_split_set(train_set, "train")
    get_data_lines_from_one_split_set(val_set, "eval")
    get_data_lines_from_one_split_set(test_set, "test")


# INPUT:
#   if_add_e2Rel: whether from (e1, label, id) to (e1, rel, e2, label, id), where rel and e2 are both ""
# OUTPUT:
#    train_set/val_set/test_set: [[text, label, line_id], ...]
def load_sentiment_data(splitted_data_dir="./Data/sentiment/splitted/", if_add_e2Rel=False):
    with open(os.path.join(splitted_data_dir, 'train.json'), 'r') as f:
        train_set = json.load(f)
    with open(os.path.join(splitted_data_dir, 'val.json'), 'r') as f:
        val_set = json.load(f)
    with open(os.path.join(splitted_data_dir, 'test.json'), 'r') as f:
        test_set = json.load(f)

    if if_add_e2Rel:
        def add_e2Rel(data_set):
            for cur_id in range(len(data_set)):
                assert len(data_set[cur_id]) == 3
                cur_label = data_set[cur_id][1]
                if cur_label == 0:
                    cur_label_text = "negative"
                elif cur_label == 1:
                    cur_label_text = "positive"
                else:
                    raise Exception("cur_label: ", cur_label)
                # as there's no relation
                data_set[cur_id].insert(1, "")
                # e2 is the expected generation for classification
                data_set[cur_id].insert(2, cur_label_text)
            return data_set
        train_set = add_e2Rel(train_set)
        val_set = add_e2Rel(val_set)
        test_set = add_e2Rel(test_set)

    print("len(train_set): ", len(train_set))
    print("len(val_set): ", len(val_set))
    print("len(test_set): ", len(test_set))
    print("train_set[:10]", train_set[:10])
    return train_set, val_set, test_set


# INPUT
#   force_split_id: only used when there's already saved subset with certain subset_selection;
#                      when force_split_id is speficied, len(split_size_list) should be 1, and force_split_id is the subset_selection for the new subset
# FUNCTION
#   to split the train set of sentiment sentence classification dataset, obtain the subset (and its corresponding index in full set) for further experiment
def sentiment_train_subset_obtainer(root_data_dir="./Data/sentiment/splitted/", split_size_list=[60, 200, 600], force_split_id=None):
    train_set, val_set, test_set = load_sentiment_data(root_data_dir, if_add_e2Rel=True)
    len_train = len(train_set)
    assert len_train > max(split_size_list)
    # split_size_id: the id of split_size in split_size_list
    def select_and_save_subset(data_set, split_size, split_size_id, data_type):
        full_index = np.arange(0, len(data_set), 1)
        np.random.shuffle(full_index)
        # print("full_index: ", full_index)
        subset_index_shuffle = full_index[:split_size]
        subset_index_sorted = sorted(subset_index_shuffle)
        data_subset = [data_set[idex] for idex in subset_index_sorted]
        # print("subset_index_sorted: ", subset_index_sorted)
        exitsing_files = os.listdir(root_data_dir)
        if "{}_subset_{}_index.npy".format(data_type, split_size_id) not in exitsing_files and \
            "{}_subset_{}_data.npy".format(data_type, split_size_id) not in exitsing_files:
            with open(os.path.join(root_data_dir, "{}_subset_{}_index.npy".format(data_type, split_size_id)), 'wb') as f:
                np.save(f, subset_index_sorted)
            with open(os.path.join(root_data_dir, "{}_subset_{}_data.npy".format(data_type, split_size_id)), 'wb') as f:
                np.save(f, data_subset)
        else:
            raise Exception('{}_subset_{}_index.npy or {}_subset_{}_data.npy already existing in {}'.format(data_type, split_size_id, data_type, split_size_id, root_data_dir))
    split_size_list = sorted(split_size_list)
    if force_split_id != None:
        assert len(split_size_list) == 1
        select_and_save_subset(train_set, split_size_list[0], force_split_id, "train")
    else:
        for split_size_id, split_size in enumerate(split_size_list):
            select_and_save_subset(train_set, split_size, split_size_id, "train")





# INPUT:
#   train_set/val_set/test_set: [[text, label, line_id], ...]
#   bow_dimension_setup: an integer
#   if_CDH_input: if no, raw bow is raw bow; else raw bow is the difference between raw bow and most similar case's raw bow
#   root_data_dir: when be used when if_CDH_input == True; used to collect the most similar cases' ids
# FUNCTION
#   lower case + bog of word feature + whitening
# OUTPUT:
#   processed_train_set, processed_val_set, processed_test_set: [whitened bow features tensor, label tensor, line_id tensor]
def preprocess_sentiment_dataset_as_NNInput(args, train_set, val_set, test_set, bow_dimension_setup, root_data_dir=""):
    # when if_CDH == True; this function needs root_data_dir
    if args.if_CDH:
        assert root_data_dir != ""
    tokenizer = RegexpTokenizer(r'\w+')
    processed_train_set, processed_val_set, processed_test_set = [], [], []
    ## bag of words feature extraction
    def get_tokenizedText_and_bowTokensCountDict(data_set):
        # data_set_text: [[text 0], ...]
        # data_set_text_tokenized: [[tokenized text 0], ...]
        data_set_text, data_set_text_tokenized = [], []
        bow_tokens_count_dict = {}
        for cur_id in range(len(data_set)):
            cur_text = data_set[cur_id][0].lower()
            cur_text_tokenized = tokenizer.tokenize(cur_text)
            data_set_text.append(cur_text)
            data_set_text_tokenized.append(cur_text_tokenized)
            for token in cur_text_tokenized:
                if token not in bow_tokens_count_dict:
                    bow_tokens_count_dict[token] = 1
                else:
                    bow_tokens_count_dict[token] += 1
        return data_set_text, data_set_text_tokenized, bow_tokens_count_dict
    train_text, train_text_tokenized, train_bow_tokens_count_dict = get_tokenizedText_and_bowTokensCountDict(train_set)
    val_text, val_text_tokenized, val_bow_tokens_count_dict = get_tokenizedText_and_bowTokensCountDict(val_set)
    test_text, test_text_tokenized, test_bow_tokens_count_dict = get_tokenizedText_and_bowTokensCountDict(test_set)
    ## get word2id for BOW feature
    sorted_bow_tokens = [k for k, v in sorted(train_bow_tokens_count_dict.items(), key=lambda item: item[1], reverse=True)]
    sorted_bow_count = [v for k, v in sorted(train_bow_tokens_count_dict.items(), key=lambda item: item[1], reverse=True)]
    assert len(sorted_bow_tokens) == len(sorted_bow_count)
    word2id = {}
    if not len(sorted_bow_tokens) >= bow_dimension_setup:
        raise Exception("len(sorted_bow_tokens): {}; bow_dimension_setup: {}".format(len(sorted_bow_tokens), bow_dimension_setup))
    for cur_id in range(len(sorted_bow_tokens[:(bow_dimension_setup-1)])):
        word2id[sorted_bow_tokens[cur_id]] = cur_id
    word2id['<unk>'] = bow_dimension_setup - 1
    # print(sorted_bow_count[bow_dimension_setup-1])
    # print(len(sorted_bow_count))
    ## To find the words that count in BOW feature
    def get_raw_bow(tokenized_data_set, bow_dimension, word2id):
        raw_bow = torch.zeros((len(tokenized_data_set), bow_dimension))
        for cur_id, cur_text_tokenized in enumerate(tokenized_data_set):
            for cur_token in cur_text_tokenized:
                if cur_token in word2id:
                    cur_word_id = word2id[cur_token]
                else:
                    cur_word_id = word2id['<unk>']
                raw_bow[cur_id, cur_word_id] += 1
        return raw_bow
    # raw_bow: torch.matrix((len(tokenized_data_set), bow_dimension))
    train_raw_bow = get_raw_bow(train_text_tokenized, bow_dimension_setup, word2id)
    val_raw_bow = get_raw_bow(val_text_tokenized, bow_dimension_setup, word2id)
    test_raw_bow = get_raw_bow(test_text_tokenized, bow_dimension_setup, word2id)
    # if using Case Difference Heuristics, this code block will process (raw_bow) to (raw_bow - most_similar_case's raw_bow)
    if args.if_CDH:
        def get_CDH_raw_bow(args, raw_bow, train_raw_bow, root_data_dir, data_type, bow_dimension, data_set, train_set):
            assert data_type == 'train' or data_type == 'val' or data_type == 'test'
            assert len(train_raw_bow) == len(train_set)
            if data_type == 'train':
                assert raw_bow.size() == train_raw_bow.size()
            most_similar_ids = torch.load(os.path.join(args.most_similar_ids_data_dir, "{}_most_similar_id_matrix_full.pt".format(data_type)))
            repetitive_similar_ids = torch.load(os.path.join(args.root_data_dir, "{}_ids_that_retrieved_the_same_case.pt".format(data_type)))
            # assert most_similar_ids.size()[0] == raw_bow.size()[0]
            len_data = raw_bow.size()[0]
            if args.CDH_NN_label_method == 3:
                CDH_raw_bow = torch.zeros((len_data, bow_dimension*2))
            else:
                CDH_raw_bow = torch.zeros((len_data, bow_dimension))
            most_similar_train_cases = []
            # dict_subsetIndex2lineId_curDataSet
            dict_subsetIndex2lineId_curDataSet = {}
            for cur_id in range(len(data_set)):
                dict_subsetIndex2lineId_curDataSet[cur_id] = data_set[cur_id][2]
            # train_subset_existing_original_line_ids
            train_subset_existing_original_line_ids = []
            dict_lineId2subsetIndex_train = {}
            for cur_id in range(len(train_set)):
                train_subset_existing_original_line_ids.append(train_set[cur_id][2])
                dict_lineId2subsetIndex_train[train_set[cur_id][2]] = cur_id
            # find CDH_raw_bow
            for cur_id in range(len_data):
                cur_bow = raw_bow[cur_id]
                cur_data_originLineId = dict_subsetIndex2lineId_curDataSet[cur_id]
                if cur_data_originLineId in repetitive_similar_ids:
                    cur_similar_ids = most_similar_ids[cur_data_originLineId][1:].tolist()
                else:
                    cur_similar_ids = most_similar_ids[cur_data_originLineId].tolist()
                tmp_similar_id = 0
                while cur_similar_ids[tmp_similar_id] not in train_subset_existing_original_line_ids:
                    tmp_similar_id += 1
                    if tmp_similar_id == len(cur_similar_ids):
                        raise Exception("Failed to find tmp_similar_id", cur_id)
                most_similar_case_bow = train_raw_bow[dict_lineId2subsetIndex_train[cur_similar_ids[tmp_similar_id]]]
                if args.CDH_NN_label_method == 3:
                    # print("cur_bow.size(): ", cur_bow.size(), "most_similar_case_bow.size(): ", most_similar_case_bow.size())
                    cur_bow_difference = cur_bow - most_similar_case_bow
                    cur_context_bow = most_similar_case_bow
                    cur_concat_feature = torch.cat((cur_bow_difference, cur_context_bow), dim=0)
                    CDH_raw_bow[cur_id] = cur_concat_feature
                else:
                    CDH_raw_bow[cur_id] = cur_bow - most_similar_case_bow
                most_similar_train_cases.append(train_set[dict_lineId2subsetIndex_train[cur_similar_ids[tmp_similar_id]]])
            assert len(CDH_raw_bow) == len(most_similar_train_cases)
            return CDH_raw_bow, most_similar_train_cases
        train_raw_bow_non_CDH = copy.deepcopy(train_raw_bow)
        train_raw_bow, train_most_similar_train_cases = get_CDH_raw_bow(args, train_raw_bow, train_raw_bow_non_CDH, root_data_dir, "train", bow_dimension_setup, train_set, train_set)
        val_raw_bow, val_most_similar_train_cases = get_CDH_raw_bow(args, val_raw_bow, train_raw_bow_non_CDH, root_data_dir, "val", bow_dimension_setup, val_set, train_set)
        test_raw_bow, test_most_similar_train_cases = get_CDH_raw_bow(args, test_raw_bow, train_raw_bow_non_CDH, root_data_dir, "test", bow_dimension_setup, test_set, train_set)
        # print("train_raw_bow: ", train_raw_bow.max().item())
        # print("val_raw_bow: ", val_raw_bow.max().item())
        # print("test_raw_bow: ", test_raw_bow.max().item())

    ## whitening
    def get_whitened_bow(raw_bow, train_raw_bow, data_type):
        assert data_type == 'train' or data_type == 'val' or data_type == 'test'
        assert raw_bow.size()[1] == train_raw_bow.size()[1]
        # print("raw_bow: ", raw_bow)
        # print("raw_bow: ", raw_bow.max().item())
        mean_bow = torch.mean(train_raw_bow, dim=0)
        std_bow = torch.std(train_raw_bow, dim=0)
        set_std_bow = torch.tensor(list(set(std_bow.tolist())))
        min_not_zero_std_bow = torch.kthvalue(set_std_bow, torch.tensor(2))[0]
        # print("min_not_zero_std_bow: ", min_not_zero_std_bow)
        for cur_id in range(len(std_bow)):
            if std_bow[cur_id] == 0:
                # print("min_not_zero_std_bow: ", min_not_zero_std_bow)
                std_bow[cur_id] = min_not_zero_std_bow
        # print("std_bow.min(): ", std_bow.min())
        whitened_bow = raw_bow - mean_bow
        whitened_bow = whitened_bow / std_bow
        # print("whitened_bow: ", whitened_bow.max().item())
        assert whitened_bow.size() == raw_bow.size()
        if data_type == 'train' and args.subset_selection == -1:
            if not torch.abs(whitened_bow.mean(dim=0).mean() - 0) < 0.01:
                raise Exception("whitened_bow.mean(dim=0).mean(): ", whitened_bow.mean(dim=0).mean())
            if not torch.abs(whitened_bow.var(dim=0).mean() - 1) < 0.2:
                raise Exception("whitened_bow.var(dim=0).mean(): ", whitened_bow.var(dim=0).mean())
        return whitened_bow
    train_whitened_bow = get_whitened_bow(train_raw_bow, train_raw_bow, 'train')
    val_whitened_bow = get_whitened_bow(val_raw_bow, train_raw_bow, 'val')
    test_whitened_bow = get_whitened_bow(test_raw_bow, train_raw_bow, 'test')
    # print("train_whitened_bow: ", train_whitened_bow.max().item())
    # print("val_whitened_bow: ", val_whitened_bow.max().item())
    # print("test_whitened_bow: ", test_whitened_bow.max().item())
    assert train_whitened_bow.size() == train_raw_bow.size()
    assert val_whitened_bow.size() == val_raw_bow.size()
    assert test_whitened_bow.size() == test_raw_bow.size()
    ## get processed_data
    def get_processed_data(args, data_set, whitened_bow, root_data_dir=None, data_type=None, train_set=None, data_set_most_similar_train_cases=None):
        if args.if_CDH:
            assert data_type == 'train' or data_type == 'val' or data_type == 'test'
            assert train_set != None
            # most_similar_ids = torch.load(os.path.join(root_data_dir, "{}_most_similar_id_matrix.pt".format(data_type)))
            # assert len(data_set) == most_similar_ids.size()[0]
            assert len(data_set_most_similar_train_cases) == len(data_set)
        assert whitened_bow.size()[0] == len(data_set)
        data_len = len(data_set)
        if args.if_CDH and args.CDH_NN_label_method == 3:
            label_tensor = torch.zeros((data_len, 2))
        else:
            label_tensor = torch.zeros((data_len))
        line_id_tensor = torch.zeros((data_len))
        for cur_id in range(data_len):
            if args.if_CDH:
                # label_most_similar_case = train_set[most_similar_ids[cur_id][0]][1]
                label_most_similar_case = data_set_most_similar_train_cases[cur_id][1]
                assert label_most_similar_case == 0 or label_most_similar_case == 1
                label_cur_query = data_set[cur_id][1]
                assert label_cur_query == 0 or label_cur_query == 1
                if args.CDH_NN_label_method == 0:
                    if label_most_similar_case == label_cur_query:
                        cur_CDH_label = 0
                    else:
                        cur_CDH_label = 1
                    label_tensor[cur_id] = cur_CDH_label
                elif args.CDH_NN_label_method == 1 or args.CDH_NN_label_method == 2:
                    label_tensor[cur_id] = label_cur_query
                elif args.CDH_NN_label_method == 3:
                    label_tensor[cur_id][label_cur_query] = 1
                    if label_most_similar_case == label_cur_query:
                        label_tensor[cur_id][abs(1-label_cur_query)] = 0
                    else:
                        label_tensor[cur_id][abs(1-label_cur_query)] = -1
                else:
                    raise NotImplementedError
            else:
                label_tensor[cur_id] = data_set[cur_id][1]
            line_id_tensor[cur_id] = data_set[cur_id][2]
        # processed_data = [whitened_bow, F.one_hot(label_tensor.to(torch.int64), num_classes=2), line_id_tensor]
        if args.if_CDH and args.CDH_NN_label_method == 3:
            label_tensor = label_tensor.to(torch.float32)
        else:
            label_tensor = label_tensor.to(torch.int64)
        processed_data = [whitened_bow, label_tensor, line_id_tensor]
        return processed_data
    if args.if_CDH:
        processed_train_set = get_processed_data(args, train_set, train_whitened_bow, root_data_dir, "train", train_set, train_most_similar_train_cases)
        processed_val_set = get_processed_data(args, val_set, val_whitened_bow, root_data_dir, "val", train_set, val_most_similar_train_cases)
        processed_test_set = get_processed_data(args, test_set, test_whitened_bow, root_data_dir, "test", train_set, test_most_similar_train_cases)
    else:
        processed_train_set = get_processed_data(args, train_set, train_whitened_bow, root_data_dir, "train", train_set)
        processed_val_set = get_processed_data(args, val_set, val_whitened_bow, root_data_dir, "val", train_set)
        processed_test_set = get_processed_data(args, test_set, test_whitened_bow, root_data_dir, "test", train_set)
    return processed_train_set, processed_val_set, processed_test_set







# files in raw_data_root_dir should be only data files
def sentiment_labelled_sentence_data_split(raw_data_root_dir="./Data/sentiment/raw_data/", data_to_save_dir="./Data/sentiment/splitted/"):
    train_set, val_set, test_set = [], [], []
    data_files = os.listdir(raw_data_root_dir)
    ttl_pos_data, ttl_neg_data = [], []
    for df in data_files:
        data_file_full_addr = os.path.join(raw_data_root_dir, df)
        cur_pos_data, cur_neg_data = [], []
        with open(data_file_full_addr, 'r') as f:
            cur_lines = f.readlines()
            for cur_line in cur_lines:
                cur_text, cur_label = cur_line.strip().split("\t")
                cur_label = int(cur_label)
                # assert cur_label == 0 or cur_label == 1
                if cur_label == 0:
                    cur_neg_data.append(cur_text)
                elif cur_label == 1:
                    cur_pos_data.append(cur_text)
                else:
                    raise Exception
        assert len(cur_pos_data) == len(cur_neg_data)
        assert len(cur_pos_data) == 500
        ttl_pos_data += cur_pos_data
        ttl_neg_data += cur_neg_data
    assert len(ttl_pos_data) == 1500
    assert len(ttl_neg_data) == 1500
    np.random.shuffle(ttl_pos_data)
    np.random.shuffle(ttl_neg_data)

    for cur_id in range(len(ttl_pos_data)):
        if cur_id < 1000:
            train_set.append([ttl_pos_data[cur_id], 1, len(train_set)])
            train_set.append([ttl_neg_data[cur_id], 0, len(train_set)])
        elif cur_id < 1250:
            val_set.append([ttl_pos_data[cur_id], 1, len(val_set)])
            val_set.append([ttl_neg_data[cur_id], 0, len(val_set)])
        elif cur_id < 1500:
            test_set.append([ttl_pos_data[cur_id], 1, len(test_set)])
            test_set.append([ttl_neg_data[cur_id], 0, len(test_set)])
        else:
            raise Exception

    # print("len(train_set): ", len(train_set))
    # print("len(val_set): ", len(val_set))
    # print("len(test_set): ", len(test_set))
    print("train_set[:10]", train_set[:10])

    with open(os.path.join(data_to_save_dir, "train.json"), 'w') as f:
        json.dump(train_set, f)
    with open(os.path.join(data_to_save_dir, "val.json"), 'w') as f:
        json.dump(val_set, f)
    with open(os.path.join(data_to_save_dir, "test.json"), 'w') as f:
        json.dump(test_set, f)
