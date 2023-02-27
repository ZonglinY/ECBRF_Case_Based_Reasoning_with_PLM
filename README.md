# ECBRF_Case_Based_Reasoning_with_PLM
The github repo for [End-to-end Case-Based Reasoning for Commonsense Knowledge Base Completion](http://sentic.net/commonsense-knowledge-base-completion.pdf), accepted by EACL 2023 (main).

Here the code can be (almost) directly used on work stations that use slurm.  
If your work station does not use slurm, just overlook anything related to slurm and only run the python commands in shell scripts mentioned below.

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


## About this code

The code is designed to dynamically retrieve cases to support updating the retriever during finetuning. However, during my very preliminary experiments, I find that at least with a relatively small number of retrieved documents (e.g., 4), updating the retriever only leads to comparable results and is significantly slower. So I do not try to update the retriever in the following experiments, and slightly update the code for faster running. Currently, the code can support updating the retriever with small modifications. If it is in need, I would continue to refine the code to support updating the retriever.



