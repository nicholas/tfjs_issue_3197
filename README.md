```
pip install tensorflowjs==1.7.4

tensorflowjs_converter \
  --input_format=tf_saved_model \
  --output_format=tfjs_graph_model \
  --weight_shard_size_bytes=10485760 \
  --skip_op_check \
  assets/exported_container \
  assets/converted_tfjs
```