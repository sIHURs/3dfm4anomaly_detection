classes=(
    # "rubberduck"
    # "binderclip2"
    # "bowl_upright"
    # "box"
    # "can"
    "charger"
    # "cup1_upright"
    # "cup2_upright"
    # "gluebottle"
    # "spoon_upright"
    # "tennisball"
    # "phonecase2"

    # "binderclip"  # 3dgs mcmc traninig is wrong
    # "cup2_upright2"
    # "cup2_upright3"
    # "phonecase"
    # "gluebottle2"
    # "spraybottle2" # also
)

python utils/msk_count_eval.py \
  --root_birefnet /home/wangyifa/tmp/3dfm4anomaly_detection/data/Anomaly_refine_msk_birefnet_infer1024 \
  --classes "${classes[@]}" \
  --min_area 3

# python utils/msk_count_eval.py \
#   --root_birefnet /home/wangyifa/tmp/3dfm4anomaly_detection/data/Anomaly_refine_msk_birefnet_infer1024 \
#   --classes "${classes[@]}" \
#   --fix \
#   --min_area 3