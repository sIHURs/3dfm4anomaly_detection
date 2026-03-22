# chmod +x tools/birefnet_msk.sh

# DEVICE=cuda INFER_SIZE=1024 THRESH=128 \
# ./tools/birefnet_msk.sh \
#     /home/wangyifa/tmp/3dfm4anomaly_detection/data/Anomaly_refine_nonmsk \
#     /home/wangyifa/tmp/3dfm4anomaly_detection/data/Anomaly_refine_msk_birefnet_infer1024 \
#     /home/wangyifa/tmp/3dfm4anomaly_detection/utils/birefnet_process.py


chmod +x tools/birefnet_msk_improved.sh

DEVICE=cuda INFER_SIZE=1024 THRESH=128 KEEP_SOFT=1 FILL_HOLES=1 \
MASKED_SUBDIR="." \
./tools/birefnet_msk_improved.sh \
  /home/wangyifa/tmp/3dfm4anomaly_detection/data/Anomaly_refine_nonmsk \
  /home/wangyifa/tmp/3dfm4anomaly_detection/data/Anomaly_refine_msk_birefnet_refine_infer1024_fp32 \
  /home/wangyifa/tmp/3dfm4anomaly_detection/utils/birefnet_process_improved.py