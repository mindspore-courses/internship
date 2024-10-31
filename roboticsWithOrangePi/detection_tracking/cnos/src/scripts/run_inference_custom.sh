export CAD_PATH=/home/yaxun/FoundationPose/demo_data/realDex/mesh/air_duster.obj
export OUTPUT_DIR=/home/yaxun/FoundationPose/cnos/realDex_template
export RGB_PATH=/home/yaxun/FoundationPose/demo_data/realDex/00temp/0.jpg
# export RGB_PATH=/home/yaxun/FoundationPose/demo_data/realDex/rgb/73.jpg
python -m src.scripts.inference_custom --template_dir $OUTPUT_DIR --rgb_path $RGB_PATH --stability_score_thresh 0.5