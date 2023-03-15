# ECBRF_Case_Based_Reasoning_with_PLM
This is the official implementation of the paper - [End-to-end Case-Based Reasoning for Commonsense Knowledge Base Completion](http://sentic.net/commonsense-knowledge-base-completion.pdf), which is accepted by EACL 2023 (main).

Here the code can be (almost) directly used on work stations that use slurm.  
If your work station does not use slurm, just overlook anything related to slurm and only run the python commands in shell scripts mentioned below.

In general, the code is designed to run in two steps.  
The first step is to obtain a checkpoint when test perplexity would be obtained (by command ```sbatch ECBRF_generator```, ```sbatch ECBRF_retriever```, or ```sbatch COMET``` illustrated below);  
the second step is to run generator.py to obtain BLEU for the existing checkpoint (by command ```sbatch generation```).

# Step 1

## To run the code on ECBRF

First, besides adjusting slurm parameters, please adjust *#SBATCH --output*, *--dataStore_dir*, and *--output_dir* in ```ECBRF_generator``` and ```ECBRF_retriever```.  
*--dataStore_dir* is the file dir that stores data (such as all checkpoints), and *--output_dir* is to store files (e.g., checkpoint) for the current experiment.  
Here *--dataStore_dir* and *--output_dir* should be the same for both ```ECBRF_generator``` and ```ECBRF_retriever```.

Then, 

```sbatch ECBRF_generator```  
```sbatch ECBRF_retriever```  

## To run the code on COMET

Similarly adjust *#SBATCH --output*, *--dataStore_dir*, and *--output_dir* in ```COMET```, and then 

```sbatch COMET```

## Parser arguments

*SBATCH --output*: file address to store the print output  
*--dataStore_dir*: file address to store some shared files that can be reused by other experiments using this repo  
*--output_dir*: where the checkpoint files for current experiment are saved  
*--model_type*: bart-base or gpt2-lmhead  
*--subset_selection*: -1 --- full train set; 0 ~ 6 --- 5 shot ~ 320 shot train set (to run few-shot experiments, please run full set experiment first, otherwise a small exception would occur);  
*--dataset_selection*: 0 --- ConceptNet; 1 --- ATOMIC  
*--prob_randomly_mask_demonstrations*: to adjust probability for *random mask*, by default is 0.3  

## Other Notes

1. *--dataStore_dir* can be set to the same file addr for all experiments using this repo, since it stores some files that can be possibly reused by all experiments using this repo  
2. ECBRF involves inter-process communication (here the two processes are generator and retriever), so it is important to make sure *--output_dir* (the checkpoint dir) is empty for both ```ECBRF_generator``` and ```ECBRF_retriever``` before the experiments. You can also assure that by always running ```ECBRF_generator``` first, and then (maybe after 10 seconds) to run ```ECBRF_retriever``` (since ```ECBRF_generator``` will automatically delete all files in *--output_dir* if these files do not correspond to a finished experiment).


# Step 2

First adjust *SBATCH --output*, *--output_dir*, *--model_type*, *--dataset_selection*, *--if_ECBRF* accordingly, and then 

```sbatch generation```

## Parser arguments

*SBATCH --output*: file address to store the print output  
*--output_dir*: the same as used in step 1, where the checkpoint are saved  
*--model_type*: bart-base or gpt2-lmhead  
*--dataset_selection*: 0 --- ConceptNet; 1 --- ATOMIC  
*--if_ECBRF*: 0 --- To run COMET baseline; 1 --- To run ECBRF  


## About this code

The code is designed to dynamically retrieve cases to support updating the retriever during finetuning. However, during my very preliminary experiments, I find that at least with a relatively small number of retrieved documents (e.g., 4), updating the retriever only leads to comparable results and is significantly slower. So I do not try to update the retriever in the following experiments, and slightly update the code for faster running. Currently, the code can support updating the retriever with small modifications. If it is in need, I would continue to refine the code to support updating the retriever.


<!-- ## About this paper

The main experiments of this paper that can show the effectiveness of ECBRF were first obtained at the end of 2020. Later I decided to enable it with dynamic retrieval to also update the retriever, which took me much time since I wrote it from scratch and did not use existing packages for the retriever. If this paper could be published earlier, it might have made more contributions to the field.  
We appreciate the reviewers and meta-reviewer for this paper in EACL 2023, who recognize the contribution of this paper. -->
