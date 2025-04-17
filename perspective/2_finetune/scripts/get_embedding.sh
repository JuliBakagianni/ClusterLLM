# ===== original embedding =====
for dataset in arxiv-clustering-s2s reddit-clustering biorxiv-clustering-s2s  stackexchange-clustering medrxiv-clustering-s2s
do
    CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python get_embedding.py \
        --model_name hkunlp/instructor-large \
        --scale default \
        --task_name $dataset \
        --data_path ../../datasets/${dataset}/.jsonl \
        --result_file ../../datasets/${dataset}_embeds.hdf5 \
        --measure
done

# ===== with checkpoint =====
# dataset=banking77
# scale=small
# checkpoint_path=CHECKPOINT_PATH
# CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python get_embedding.py \
#     --model_name hkunlp/instructor-large \
#     --checkpoint $checkpoint_path \
#     --scale $scale \
#     --task_name $dataset \
#     --data_path ../../datasets/${dataset}/${scale}.jsonl \
#     --result_file ${checkpoint_path}/${scale}_embeds.hdf5 \
#     --measure \
#     --overwrite
