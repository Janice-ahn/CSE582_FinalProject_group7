gpu=$1
batch=$2
model=$3
lr=$4  # 5e-4 for small, base and large; 1e-4 for 3b

export out_dir="./out/cls/${model}"
export data_dir="./data"
export label2id_dir="./data/categories2id.json"
export cache_dir="./cache"

export CUDA_VISIBLE_DEVICES=$gpu
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export TRANSFORMERS_CACHE=${cache_dir}/huggingface
export CUDA_LAUNCH_BLOCKING="0"

export epoch=5
export lr_cls=3e-3

python main.py \
    --model_name_or_path ${model} \
    --do_train \
    --do_predict \
    --train_file ${data_dir}/train.csv \
    --test_file ${data_dir}/test.csv \
    --per_device_train_batch_size ${batch} \
    --per_device_eval_batch_size ${batch} \
    --cache_dir ${cache_dir} \
    --output_dir ${out_dir} \
    --learning_rate ${lr} \
    --learning_rate_cls ${lr_cls} \
    --num_train_epochs ${epoch} \
    --evaluation_strategy epoch \
    --seed 42 \
    --max_seq_length 1024 \
    --classifier_dropout 0.2 \
    --label_column_name label
