import os, argparse, copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, TensorDataset)
import nltk
nltk.download('punkt')
from utils_baseline import load_sentiment_data, preprocess_sentiment_dataset_as_NNInput, get_data_lines_using_sentimentSentence_dataset_for_retriever

# device = torch.device("cuda")
device = torch.device("cpu")

class Net(torch.nn.Module):
    def __init__(self, args, n_feature, n_hidden1=128, n_hidden2=64, n_output=2):
        super(Net, self).__init__()
        self.args = args
        self.hidden1 = torch.nn.Linear(n_feature, n_hidden1)   # hidden layer
        # torch.nn.init.xavier_uniform_(self.hidden1.weight)
        self.dropout1 = nn.Dropout(0.25)
        self.hidden2 = torch.nn.Linear(n_hidden1, n_hidden2)   # hidden layer
        # torch.nn.init.xavier_uniform_(self.hidden2.weight)
        self.dropout2 = nn.Dropout(0.25)
        self.out = torch.nn.Linear(n_hidden2, n_output)   # output layer
        # torch.nn.init.xavier_uniform_(self.out.weight)

    def forward(self, x):
        x = F.relu(self.hidden1(x))      # activation function for hidden layer
        x = self.dropout1(x)
        x = F.relu(self.hidden2(x))      # activation function for hidden layer
        x = self.dropout2(x)
        x = self.out(x)
        if self.args.if_CDH and self.args.CDH_NN_label_method == 3:
            output = F.tanh(x)
        else:
            output = F.log_softmax(x, dim=1)
        return output

# INPUT:
#   y_pred: [batch_size, 2]
#   label: [batch_size] (with label 0 or 1)
def get_accuracy(args, y_pred, label, batch_line_ids=None, train_set=None, data_set=None, data_type=None):
    assert y_pred.size()[0] == label.size()[0]
    assert y_pred.size()[0] == batch_line_ids.size()[0]
    len_label = label.size()[0]
    if args.if_CDH:
        assert batch_line_ids != None and train_set != None and args.root_data_dir != None and data_set != None
        assert data_type == "train" or data_type == "val" or data_type == "test"
        if args.if_random_retrieval == 0:
            most_similar_ids = torch.load(os.path.join(args.most_similar_ids_data_dir, "{}_most_similar_id_matrix_full.pt".format(data_type)))
        elif args.if_random_retrieval == 1:
            most_similar_ids = torch.load(os.path.join(args.most_similar_ids_data_dir, "{}_most_similar_id_matrix_full_randperm.pt".format(data_type)))
        else:
            raise NotImplementedError
        repetitive_similar_ids = torch.load(os.path.join(args.root_data_dir, "{}_ids_that_retrieved_the_same_case.pt".format(data_type)))

        # train_subset_existing_original_line_ids
        train_subset_existing_original_line_ids = []
        # dict_lineId2subsetIndex_train
        dict_lineId2subsetIndex_train = {}
        for cur_id in range(len(train_set)):
            train_subset_existing_original_line_ids.append(train_set[cur_id][2])
            dict_lineId2subsetIndex_train[train_set[cur_id][2]] = cur_id

        label_most_similar_case = []
        for cur_id in range(len_label):
            # get cur_most_similar_label
            cur_data_originLineId = batch_line_ids[cur_id]
            if batch_line_ids[cur_id] in repetitive_similar_ids:
                cur_similar_ids = most_similar_ids[cur_data_originLineId][1:].tolist()
            else:
                cur_similar_ids = most_similar_ids[cur_data_originLineId].tolist()
            tmp_similar_id = 0
            while cur_similar_ids[tmp_similar_id] not in train_subset_existing_original_line_ids:
                tmp_similar_id += 1
                if tmp_similar_id == len(cur_similar_ids):
                    raise Exception("Failed to find tmp_similar_id", cur_id)
            cur_most_similar_label = train_set[dict_lineId2subsetIndex_train[cur_similar_ids[tmp_similar_id]]][1]
            assert cur_most_similar_label == 0 or cur_most_similar_label == 1
            label_most_similar_case.append(cur_most_similar_label)
        # if data_type == 'test':
        #     print("batch_line_ids: ", batch_line_ids)
        #     print("label_most_similar_case: ", label_most_similar_case, len(label_most_similar_case))
    correct_cnt, ttl_count = 0, 0
    win_1, win_2 = 0, 0
    for cur_id in range(len_label):
        if args.if_CDH:
            if args.CDH_NN_label_method == 0 or args.CDH_NN_label_method == 1 or args.CDH_NN_label_method == 2:
                assert label[cur_id] == 0 or label[cur_id] == 1
                prob_0, prob_1 = torch.exp(y_pred[cur_id][0]), torch.exp(y_pred[cur_id][1])
                if not torch.abs(prob_0 + prob_1 - 1) <= 0.005:
                    raise Exception("prob_0: {}; prob_1: {}".format(prob_0, prob_1))
                prob_0 = prob_0 / (prob_0 + prob_1)
                prob_1 = prob_1 / (prob_0 + prob_1)
                if args.CDH_NN_label_method == 0:
                    if y_pred[cur_id][0] >= y_pred[cur_id][1]:
                        if label_most_similar_case[cur_id] == label[cur_id]:
                            correct_cnt += 1
                    else:
                        if label_most_similar_case[cur_id] != label[cur_id]:
                            correct_cnt += 1
                elif args.CDH_NN_label_method == 1:
                    if (y_pred[cur_id][0] >= y_pred[cur_id][1] and label[cur_id] == 0) or (y_pred[cur_id][0] <= y_pred[cur_id][1] and label[cur_id] == 1):
                        correct_cnt += 1
                elif args.CDH_NN_label_method == 2:
                    if label[cur_id] == 0:
                        if prob_0 >= args.threshold_for_CDH:
                            correct_cnt += 1
                            win_1 += 1
                        elif prob_0 >= (1-args.threshold_for_CDH) and label_most_similar_case[cur_id] == 0:
                            correct_cnt += 1
                            win_2 += 1
                    elif label[cur_id] == 1:
                        if prob_1 >= args.threshold_for_CDH:
                            correct_cnt += 1
                            win_1 += 1
                        elif prob_1 >= (1-args.threshold_for_CDH) and label_most_similar_case[cur_id] == 1:
                            correct_cnt += 1
                            win_2 += 1
                    else:
                        raise Exception
                else:
                    raise NotImplementedError
            elif args.CDH_NN_label_method == 3:
                assert len(label[cur_id].size()) == 1 and len(y_pred[cur_id].size()) == 1
                assert label[cur_id].shape[0] == 2 and y_pred[cur_id].shape[0] == 2
                retrieved_case_label = torch.zeros(2)
                retrieved_case_label[label_most_similar_case] = 1
                final_label = retrieved_case_label + y_pred[cur_id]
                if y_pred[cur_id][0] >= y_pred[cur_id][1]:
                    cur_pred_label = 0
                else:
                    cur_pred_label = 1
                if label[cur_id][cur_pred_label] == 1:
                    correct_cnt += 1
            else:
                raise NotImplementedError
        else:
            if (y_pred[cur_id][0] >= y_pred[cur_id][1] and label[cur_id] == 0) or (y_pred[cur_id][0] <= y_pred[cur_id][1] and label[cur_id] == 1):
                correct_cnt += 1
        ttl_count += 1
    assert ttl_count == len_label
    cur_accuracy = correct_cnt / ttl_count
    # print("win_1: ", win_1, "win_2: ", win_2)
    return cur_accuracy

def evaluate(args, dataloader, net, loss_func, train_set=None, data_set=None, data_type=None):
    correct_cnt, ttl_count = 0, 0
    accumulated_loss = 0
    for step, batch in enumerate(dataloader):
        batch_bow_features, batch_label, batch_line_ids = batch
        # if data_type == 'test':
        #     print("batch_line_ids: ", batch_line_ids)
        batch_line_ids = batch_line_ids.type(torch.LongTensor)
        len_batch = batch_bow_features.size()[0]
        batch_bow_features = batch_bow_features.to(device)
        batch_label = batch_label.to(device)

        y_pred = net(batch_bow_features)
        loss = loss_func(y_pred, batch_label)
        # optimizer.zero_grad()
        # loss.backward()
        # optimizer.step()

        cur_accuracy = get_accuracy(args, y_pred, batch_label, batch_line_ids=batch_line_ids, train_set=train_set, data_set=data_set, data_type=data_type)
        correct_cnt += cur_accuracy * len_batch
        accumulated_loss += loss * len_batch
        ttl_count += len_batch
    eval_accuracy = correct_cnt / ttl_count
    eval_loss = accumulated_loss / ttl_count
    return eval_loss, eval_accuracy



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bow_dimension_setup", type=int, default=2048)
    parser.add_argument("--train_batch_size", type=int, default=32)
    parser.add_argument("--dev_batch_size", type=int, default=32)
    parser.add_argument("--test_batch_size", type=int, default=32)
    # 1e-3
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    # 3e-4
    # 3e-5
    parser.add_argument("--learning_rate", type=float, default=3e-5)
    parser.add_argument("--num_total_epochs", type=int, default=200)
    parser.add_argument("--eval_step", type=int, default=200)
    parser.add_argument("--patience", type=int, default=5)
    # parser.add_argument("--root_data_dir_used", type=str, default="./Data/sentiment/splitted/Used_data/")
    parser.add_argument("--root_data_dir", type=str, default="./Data/sentiment/splitted/")
    parser.add_argument("--most_similar_ids_data_dir", type=str, default="../Datas/sentiment/")
    parser.add_argument("--if_CDH", type=int, default=0)
    parser.add_argument("--CDH_NN_label_method", type=int, default=3, help="only be useful when args.if_CDH==1; 0: '0' for not change label, '1' for change label; 1: '0' for should changed to '0', '1' for should changed to '1'; 2: prob for '0' larger than threshold then should change to 0; prob for '1' is the same; 3: use tanh activation as last layer's output and use MSE loss and input uses (retrieved case's bow concats bow difference).")
    parser.add_argument("--threshold_for_CDH", type=float, default=0.75, help="only be usedful when args.if_CDH==1 and args.CDH_NN_label_method==2; only when the prob is larger than threshold does the CDH method change its label.")
    parser.add_argument("--subset_selection", type=int, default=-1)
    parser.add_argument("--if_random_retrieval", type=int, default=0, help="only be usedful when args.if_CDH==1; 0: DPR retrieval; 1: random retrieval")
    args = parser.parse_args()

    assert args.if_CDH == 0 or args.if_CDH == 1
    assert args.CDH_NN_label_method == -1 or args.CDH_NN_label_method == 0 or args.CDH_NN_label_method == 1 or args.CDH_NN_label_method == 2 or args.CDH_NN_label_method == 3
    assert args.subset_selection >= -1 and args.subset_selection <= 3
    assert args.if_random_retrieval == 0 or args.if_random_retrieval == 1

    if args.subset_selection == 0:
        # len(sorted_bow_tokens): 387; we take half as bow_dimension
        args.bow_dimension_setup = 200
    elif args.subset_selection == 1:
        # len(sorted_bow_tokens): 882; we take half as bow_dimension
        args.bow_dimension_setup = 450
    elif args.subset_selection == 2:
        # len(sorted_bow_tokens): 1913; we take half as bow_dimension
        args.bow_dimension_setup = 1000
    elif args.subset_selection == 3:
        # len(sorted_bow_tokens): 154; we take half as bow_dimension
        args.bow_dimension_setup = 75

    if args.subset_selection == -1:
        train_set, val_set, test_set = load_sentiment_data(splitted_data_dir=args.root_data_dir, if_add_e2Rel=False)
    else:
        _, val_set, test_set = load_sentiment_data(splitted_data_dir=args.root_data_dir, if_add_e2Rel=False)
        with open(os.path.join(args.root_data_dir, "{}_subset_{}_data.npy".format("train", args.subset_selection)), 'rb') as f:
            # currently train_set: [[e1, rel, e2, label, line_id]]
            origin_train_set = np.load(f)
            # train_set -> [[e1, label, line_id]]
            train_set = []
            for cur_id in range(len(origin_train_set)):
                # print([origin_train_set[cur_id][0]], origin_train_set[cur_id][3:].tolist())
                cur_data = [origin_train_set[cur_id][0]] + [int(origin_train_set[cur_id][3])] + [int(origin_train_set[cur_id][4])]
                train_set.append(cur_data)
                if cur_id == 0:
                    print("cur_data: ", cur_data)
        print("len(train_set): ", len(train_set))
    ## only used for generate data lines in "./Data/sentiment/splitted/" to be used for retriever
    # get_data_lines_using_sentimentSentence_dataset_for_retriever(train_set, val_set, test_set, splitted_data_dir=args.root_data_dir)
    # raise Exception("data_lines writed for retriever.")

    # processed_train/val/test_set: [whitened bow features tensor, label tensor, line_id tensor]
    processed_train_set, processed_val_set, processed_test_set = preprocess_sentiment_dataset_as_NNInput(args, train_set, val_set, test_set, args.bow_dimension_setup, args.root_data_dir)

    train_data = TensorDataset(*processed_train_set)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

    eval_data = TensorDataset(*processed_val_set)
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.dev_batch_size)

    test_data = TensorDataset(*processed_test_set)
    test_sampler = SequentialSampler(test_data)
    test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=args.test_batch_size)

    if args.if_CDH and args.CDH_NN_label_method == 3:
        net = Net(args, n_feature=args.bow_dimension_setup*2)
    else:
        net = Net(args, n_feature=args.bow_dimension_setup)
    net.to(device)

    param_net = list(net.named_parameters())
    # print("param_net: ", param_net)
    no_decay = ["bias"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in param_net if not any(nd in n for nd in no_decay)], "weight_decay": args.weight_decay, 'lr': args.learning_rate},
        {"params": [p for n, p in param_net if any(nd in n for nd in no_decay)], "weight_decay": 0.0, 'lr': args.learning_rate}
        ]
    optimizer = torch.optim.Adam(optimizer_grouped_parameters, lr=args.learning_rate)
    if args.if_CDH and args.CDH_NN_label_method == 3:
        loss_func = torch.nn.MSELoss()
    else:
        loss_func = torch.nn.NLLLoss()

    best_val_loss = 1e10
    patience = args.patience
    best_NN = None
    for id_epoch in range(args.num_total_epochs):
        if patience == 0:
            break
        for step, batch in enumerate(train_dataloader):
            # prepare batch data
            batch_bow_features, batch_label, batch_line_ids = batch
            batch_line_ids = batch_line_ids.type(torch.LongTensor)
            batch_bow_features = batch_bow_features.to(device)
            batch_label = batch_label.to(device)
            # forward and backward
            y_pred = net(batch_bow_features)
            loss = loss_func(y_pred, batch_label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # show train step
            if step % args.eval_step == 0:
                cur_accuracy = get_accuracy(args, y_pred, batch_label, batch_line_ids, train_set, train_set, "train")
                # print("train_loss: {:.5f}; accuracy: {}".format(loss, cur_accuracy))
            # validation set
            if step % args.eval_step == 0:
                with torch.no_grad():
                    eval_loss, eval_accuracy = evaluate(args, eval_dataloader, net, loss_func, train_set=train_set, data_set=val_set, data_type="val")
                    if eval_loss < best_val_loss:
                        best_val_loss = eval_loss
                        patience = args.patience
                        best_NN = copy.deepcopy(net)
                    else:
                        patience -= 1
                        if patience == 0:
                            break
                    print("eval_loss: {:.5f}; eval_accuracy: {}; patience: {}".format(eval_loss, eval_accuracy, patience))

    test_loss, test_accuracy = evaluate(args, test_dataloader, best_NN, loss_func, train_set=train_set, data_set=test_set, data_type="test")
    print("test_loss: {:.5f}; test_accuracy: {}".format(test_loss, test_accuracy))
















if __name__ == "__main__":
    main()
