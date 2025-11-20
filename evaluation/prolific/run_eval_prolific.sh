CATEGORY=general # animals, dance, general, pedestrian, sports
VIS_SAVE_PATH="results/VideoLISA-prolific-${CATEGORY}"

# Step-2: run evaluation
set -x
python evaluation/prolific/eval_prolific.py \
  --annotation /weka/oe-training-default/mm-olmo/video_datasets/prolific/video_text_queries_filtered_111025_val_category_${CATEGORY}/annotation/largest_center/ \
  --video_dir /weka/oe-training-default/mm-olmo/video_datasets/prolific/video_text_queries_filtered_111025_val/JPEGImages/ \
  --predictions_dir $VIS_SAVE_PATH \
  --save_name $VIS_SAVE_PATH"result.json"

