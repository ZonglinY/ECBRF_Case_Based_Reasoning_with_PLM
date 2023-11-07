import argparse, json, os
import torch
import numpy as np
from utils_baseline import load_sentiment_data



def main():
    parser = argparse.ArgumentParser()
    # parser.add_argument("--root_data_dir_used", type=str, default="./Data/sentiment/splitted/Used_data/")
    parser.add_argument("--root_data_dir", type=str, default="./Data/sentiment/splitted/")
    parser.add_argument("--most_similar_ids_data_dir", type=str, default="../Datas/sentiment/")
    parser.add_argument("--subset_selection", type=int, default=-1)
    parser.add_argument("--if_random_retrieval", type=int, default=0, help="0: DPR retrieval; 1: random retrieval")
    args = parser.parse_args()

    assert args.subset_selection >= -1 and args.subset_selection <= 3
    assert args.if_random_retrieval == 0 or args.if_random_retrieval == 1

    # train_set/val_set/test_set: [[text, label, line_id], ...]
    train_set, val_set, test_set = load_sentiment_data(splitted_data_dir=args.root_data_dir, if_add_e2Rel=False)
    if args.subset_selection == -1:
        train_subset_existing_original_line_ids = np.arange(0, len(train_set)-1, 1)
    else:
        with open(os.path.join(args.root_data_dir, "{}_subset_{}_data.npy".format("train", args.subset_selection)), 'rb') as f:
            train_subset_5items = np.load(f)
            # train_datasets: 3 items (e1, label, line_id)
            train_subset = []
            train_subset_existing_original_line_ids = []
            for cur_id in range(len(train_subset_5items)):
                cur_data = [train_subset_5items[cur_id][0]] + [int(train_subset_5items[cur_id][3])] + [int(train_subset_5items[cur_id][4])]
                train_subset.append(cur_data)
                train_subset_existing_original_line_ids.append(int(train_subset_5items[cur_id][4]))
    if args.if_random_retrieval == 0:
        most_similar_ids = torch.load(os.path.join(args.most_similar_ids_data_dir, "test_most_similar_id_matrix_full.pt"))
    elif args.if_random_retrieval == 1:
        most_similar_ids = torch.load(os.path.join(args.most_similar_ids_data_dir, "test_most_similar_id_matrix_full_randperm.pt"))
    else:
        raise NotImplementedError
    repetitive_similar_ids = torch.load(os.path.join(args.root_data_dir, "test_ids_that_retrieved_the_same_case.pt"))
    assert most_similar_ids.size()[0] == len(test_set)
    len_test = len(test_set)

    cnt_correct, cnt_all = 0, 0
    for cur_id in range(len_test):
        true_label = test_set[cur_id][1]
        assert true_label == 0 or true_label == 1
        if cur_id in repetitive_similar_ids:
            selectable_most_similar_ids = most_similar_ids[cur_id][1:]
        else:
            selectable_most_similar_ids = most_similar_ids[cur_id]
        # print("selectable_most_similar_ids: ", selectable_most_similar_ids)
        # print("train_subset_existing_original_line_ids: ", train_subset_existing_original_line_ids)
        tmp_id_to_pick = 0
        while selectable_most_similar_ids[tmp_id_to_pick].item() not in train_subset_existing_original_line_ids:
            tmp_id_to_pick += 1
            if tmp_id_to_pick == len(selectable_most_similar_ids):
                raise Exception("Failed to find tmp_id_to_pick", cur_id)
        # print("tmp_id_to_pick: ", tmp_id_to_pick)
        pred_label = train_set[selectable_most_similar_ids[tmp_id_to_pick]][1]
        assert pred_label == 0 or pred_label == 1
        if pred_label == true_label:
            cnt_correct += 1
        cnt_all += 1

    accuracy = cnt_correct / cnt_all
    print("accuracy: ", accuracy)







if __name__ == "__main__":
    main()
