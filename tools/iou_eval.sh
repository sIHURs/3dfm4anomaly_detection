
classes=(
    "rubberduck"
    "binderclip2"
    "bowl_upright"  #-> bowl
    "box"
    "can"
    "charger"
    "cup1_upright"  #-> cup1
    "cup2_upright"  #-> cup2
    "gluebottle"
    "spoon_upright" #-> spoon
    "tennisball"
    "phonecase2"

    "binderclip"
    "cup2_upright2" #-> cup3
    "cup2_upright3" #-> cup4
    "phonecase"
    "gluebottle2"
    "spraybottle2" 
)

# python utils/msk_iou_eval.py \
#     --root /home/wangyifa/tmp/3dfm4anomaly_detection/data/Anomaly_refine_msk \
#     --root_birefnet /home/wangyifa/tmp/3dfm4anomaly_detection/data/Anomaly_refine_msk_birefnet_infer1024 \
#     --classes "${classes[@]}" \
#     --tol 5 \
#     # --save_white_msk \

# new
python utils/msk_iou_eval.py \
    --root /home/wangyifa/tmp/3dfm4anomaly_detection/data/Anomaly_refine_msk \
    --root_birefnet /home/wangyifa/tmp/3dfm4anomaly_detection/data/Anomaly_refine_msk_birefnet_infer1024 \
    --classes "${classes[@]}" \
    --tol 5 \
    --save_plots