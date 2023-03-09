#!/bin/bash
#SBATCH -J G0
#SBATCH -N 1
#SBATCH -n 2
#SBATCH --partition=RTXA6Kq
#SBATCH -w node09
#SBATCH --gres=gpu:1
#SBATCH --output /export/home2/zonglin001/Out/Gene_addedST_conceptnet_bart_comet_subset0_bs32.out


# --model_type gpt2-lmhead; bart-base
python -u /export/home2/zonglin001/ECBRF_Case_Based_Reasoning_with_PLM/generation.py \
            --use_special_tokens_to_split_retrieved_cases \
            --if_not_adding_special_relation_tokens \
            --rel_lang --num_cases 3 --BLEU_n 2 --if_val 0 \
            --output_dir /export/home2/zonglin001/Checkpoints/addedST_conceptnet_bart_comet_subset0_bs32/  \
            --model_type  bart-base \
            --dataset_selection 0  \
            --is_greedy \
            --if_ECBRF 1

            # --top_p 1.0 --top_k 50 --length_penalty 1.0 --temperature 1.0 \
            # --if_use_full_memory_store_while_subset --num_sample 3 \
            # --if_only_use_retrieved_target \



# --if_without_case --if_not_with_strt_mid_promp \
# --if_with_strt_mid_promp \
# --toy
# --add_separator
# --if_nltk_RM_stop_words 1
# --toy \
# --if_only_use_relation_and_retrieved_target \
# --is_greedy \
# --beam_size 10 \
