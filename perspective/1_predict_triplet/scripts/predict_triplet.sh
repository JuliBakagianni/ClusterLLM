export OPENAI_API_KEY="OPENAI_API_KEY"
for dataset in arxiv-clustering-s2s reddit-clustering biorxiv-clustering-s2s  stackexchange-clustering medrxiv-clustering-s2s
do
    link_path=sampled_triplet_results/${dataset}_embed=instructor_s=small_m=1024_d=67.0_sf_choice_seed=100.json
    # link_path=sampled_triplet_results/${dataset}_embed=instructor_s=large_m=1024_d=67.0_sf_choice_seed=100.json
    OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python predict_triplet.py \
        --dataset $dataset \
        --data_path $link_path \
        --openai_org "OPENAI_ORG" \
        --model_name gpt-3.5-turbo-0301 \
        --temperature 0
done

