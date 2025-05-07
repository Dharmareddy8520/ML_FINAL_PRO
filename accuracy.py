import os
import tensorflow as tf
from object_detection.utils import config_util
from object_detection.utils import label_map_util
from object_detection.metrics import coco_evaluation
from object_detection.protos import pipeline_pb2
from tensorboard.backend.event_processing import event_accumulator
from google.protobuf import text_format

# Define paths
MODEL_NAME = 'my_ssd_mobnet'  # Change to 'my_ssd_vgg16' for VGG16
paths = {
    'APIMODEL_PATH': os.path.join('Tensorflow', 'models'),
    'ANNOTATION_PATH': os.path.join('Tensorflow', 'workspace', 'annotations'),
    'CHECKPOINT_PATH': os.path.join('Tensorflow', 'workspace', 'models', MODEL_NAME),
}
files = {
    'PIPELINE_CONFIG': os.path.join('Tensorflow', 'workspace', 'models', MODEL_NAME, 'pipeline.config'),
    'LABELMAP': os.path.join('Tensorflow', 'workspace', 'annotations', 'label_map.pbtxt')
}

# Verify pipeline.config exists
if not os.path.exists(files['PIPELINE_CONFIG']):
    raise FileNotFoundError(f"Pipeline config not found at {files['PIPELINE_CONFIG']}. Ensure Step 4 of the original code was run.")

# Load pipeline config as a dictionary
config_dict = config_util.get_configs_from_pipeline_file(files['PIPELINE_CONFIG'])

# Convert dictionary to protobuf message
pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
with tf.io.gfile.GFile(files['PIPELINE_CONFIG'], 'r') as f:
    proto_str = f.read()
    text_format.Merge(proto_str, pipeline_config)

# Ensure COCO metrics are enabled
eval_config = pipeline_config.eval_config
eval_config.metrics_set.clear()  # Clear existing metrics
eval_config.metrics_set.append('coco_detection_metrics')  # Add COCO metrics

# Write updated config to a temporary file
eval_dir = os.path.join(paths['CHECKPOINT_PATH'], 'eval')
if not os.path.exists(eval_dir):
    os.makedirs(eval_dir)
temp_config_path = os.path.join(eval_dir, 'temp_eval.config')
with tf.io.gfile.GFile(temp_config_path, 'w') as f:
    f.write(text_format.MessageToString(pipeline_config))

# Run evaluation
TRAINING_SCRIPT = os.path.join(paths['APIMODEL_PATH'], 'research', 'object_detection', 'model_main_tf2.py')
command = f"python {TRAINING_SCRIPT} --model_dir={paths['CHECKPOINT_PATH']} " \
          f"--pipeline_config_path={temp_config_path} --checkpoint_dir={paths['CHECKPOINT_PATH']} " \
          f"--eval_dir={eval_dir}"
print(f"Running evaluation: {command}")
os.system(command)

# Load TensorBoard events to extract metrics
ea = event_accumulator.EventAccumulator(eval_dir)
ea.Reload()

# Define metrics to extract
metrics = {
    'mAP@0.5IOU': 'Precision/mAP@0.5IOU',
    'Precision': 'Precision/)' ,
    'Recall': 'Recall/AR@1'
}

# Print and save results
output_file = os.path.join(eval_dir, 'evaluation_results.txt')
with open(output_file, 'w') as f:
    for metric_name, metric_key in metrics.items():
        if metric_key in ea.Tags()['scalars']:
            values = ea.Scalars(metric_key)
            if values:
                value = values[-1].value
                print(f"{metric_name}: {value:.4f}")
                f.write(f"{metric_name}: {value:.4f}\n")
    
    # Extract per-class AP
    for class_name in ['Fall-Detected', 'Gloves', 'Goggles', 'Hardhat', 'Ladder', 'Mask',
                       'NO-Gloves', 'NO-Goggles', 'NO-Hardhat', 'NO-Mask', 'NO-Safety Vest',
                       'Person', 'Safety Cone', 'Safety Vest']:
        ap_key = f'Precision/mAP@0.5IOU/{class_name}'
        if ap_key in ea.Tags()['scalars']:
            ap_values = ea.Scalars(ap_key)
            if ap_values:
                ap = ap_values[-1].value
                print(f"AP@0.5IOU/{class_name}: {ap:.4f}")
                f.write(f"AP@0.5IOU/{class_name}: {ap:.4f}\n")

print(f"Results saved to {output_file}")