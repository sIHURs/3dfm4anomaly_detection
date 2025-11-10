```bash
colmap model_converter \
    --input_path /path/to/sparse/0 \
    --output_path /path/to/sparse_txt \
    --output_type TXT

colmap model_merger \
    --input_path1 /path/to/model1 \
    --input_path2 /path/to/model2 \
    --output_path /path/to/merged_model

colmap model_analyzer --path /path/to/sparse/0
```