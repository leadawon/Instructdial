export WANDB_PROJECT=instructdial
cd scripts
deepspeed --include localhost:0,1,2,3,5,6,7 ./run_train.py \
    --model_name_or_path prakharz/DIAL-BART0 \
    --do_train \
    --do_eval \
    --train_file text2textfiles/dawon_train.json \
    --validation_file text2textfiles/dawon_test.json \
    --text_column prompt \
    --target_column output \
    --output_dir ./tmp/outmodel_m1 \
    --per_device_train_batch_size=3 \
    --per_device_eval_batch_size=3 \
    --gradient_accumulation_steps 12 \
    --learning_rate 5e-05 \
    --overwrite_output_dir \
    --predict_with_generate \
    --gradient_checkpointing \
    --save_total_limit 3\
    --deepspeed ds-config.json \
    --evaluation_strategy steps\
    --num_train_epochs 3\
    --fp16 \
    --load_best_model_at_end\
    --metric_for_best_model f1 \
    --save_steps 100\
    --eval_steps 100\
    --logging_steps 25
