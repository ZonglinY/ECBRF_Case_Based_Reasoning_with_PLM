#!/bin/bash
#SBATCH -J nrm1_retr
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --partition=DGXq
#SBATCH -w node19
#SBATCH --gres=gpu:1
#SBATCH --output /export/home/zonglin001/Out/twitter_bart_noRandMask_cbr_subset1_bs16_Seed106_retr.out

## with frozen retriever
# num_steps has no use for now
# sample_size for conceptnet should not be less than 5000
# simi_batch_size for conceptnet should be as low as possible
# --generator_model_type bart-base; gpt2-lmhead \
python -u ./comet-atomic-index-builder-v6-twoEmbedder-DPR-faster-May-TST.py  \
        --train_batch_size 32 --eval_batch_size 32 \
        --simi_batch_size 2 \
        --generator_model_type bart-base --retriever_model_type dpr \
        --n_doc 1 --num_cases 3 \
        --sample_size 200000 --if_only_embed_ckb_once \
        --rerank_selection 1 \
        --use_special_tokens_to_split_retrieved_cases --if_not_adding_special_relation_tokens \
        --dataStore_dir /export/home/zonglin001/Checkpoints/ \
        --output_dir /export/home/zonglin001/Checkpoints/twitter_bart_noRandMask_cbr_subset1_bs16_Seed106/     \
        --dataset_selection 7 --subset_selection 1 \
        --if_use_nshot_data --num_sample 1 \
        --if_reverse_order_demonstrations \
        # --random_retrieval \
        # --if_use_full_memory_store_while_subset --subset_selection_while_use_full_memory_store 3 \







## still support:
# --if_double_retrieval --filter_ratio 0.34 \
# --larger_range_to_select_retrieval_randomly_from 12 \
# --use_only_sub_rel_for_retrieval \
# --possibility_add_cur_tuple_to_its_retrieved_cases 0.85 \
# --if_use_relation_for_shakes \

## may or may not support:
# --use_obj_for_retrieval
# --model_type bert \
