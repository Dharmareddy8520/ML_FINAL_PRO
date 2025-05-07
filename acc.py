import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from object_detection.utils import config_util, label_map_util
from object_detection.builders import model_builder
from object_detection.metrics import coco_evaluation
from object_detection.core import standard_fields as fields

# Define paths
MODEL_NAME = 'my_ssd_mobnet'
paths = {
    'APIMODEL_PATH': os.path.join('Tensorflow', 'models'),
    'ANNOTATION_PATH': os.path.join('Tensorflow', 'workspace', 'annotations'),
    'CHECKPOINT_PATH': os.path.join('Tensorflow', 'workspace', 'models', MODEL_NAME),
}
files = {
    'PIPELINE_CONFIG': os.path.join('Tensorflow', 'workspace', 'models', MODEL_NAME, 'pipeline.config'),
    'LABELMAP': os.path.join('Tensorflow', 'workspace', 'annotations', 'label_map.pbtxt'),
    'TEST_RECORD': os.path.join('Tensorflow', 'workspace', 'annotations', 'test.record'),
}

# Create evaluation directory
eval_dir = os.path.join(paths['CHECKPOINT_PATH'], 'eval')
os.makedirs(eval_dir, exist_ok=True)

# Load pipeline config and build model
configs = config_util.get_configs_from_pipeline_file(files['PIPELINE_CONFIG'])
model_config = configs['model']
detection_model = model_builder.build(model_config=model_config, is_training=False)

# Restore checkpoint
ckpt_path = os.path.join(paths['CHECKPOINT_PATH'], 'ckpt-11')
if not os.path.exists(ckpt_path + '.index'):
    raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(ckpt_path).expect_partial()

# Load label map
category_index = label_map_util.create_category_index_from_labelmap(files['LABELMAP'])
class_names = [category_index[i]['name'] for i in sorted(category_index.keys())]
num_classes = len(class_names)

# Verify label map
print(f"Number of classes: {num_classes}")
print(f"Class names: {class_names}")

# Define detection function
@tf.function
def detect_fn(image):
    print(f"Before preprocess - Image dtype: {image.dtype}, Shape: {image.shape}")
    normalized_image = tf.cast(image, tf.float32)
    normalized_image = (2.0 / 255.0) * normalized_image - 1.0
    print(f"After manual normalization - Image dtype: {normalized_image.dtype}, Shape: {normalized_image.shape}")
    true_image_shapes = tf.constant([[320, 320, 3]], dtype=tf.int32)
    prediction_dict = detection_model.predict(normalized_image, true_image_shapes)
    detections = detection_model.postprocess(prediction_dict, true_image_shapes)
    return detections

# Load test dataset
dataset = tf.data.TFRecordDataset(files['TEST_RECORD'])
feature_description = {
    'image/encoded': tf.io.FixedLenFeature([], tf.string),
    'image/object/class/label': tf.io.VarLenFeature(tf.int64),
    'image/object/bbox/xmin': tf.io.VarLenFeature(tf.float32),
    'image/object/bbox/ymin': tf.io.VarLenFeature(tf.float32),
    'image/object/bbox/xmax': tf.io.VarLenFeature(tf.float32),
    'image/object/bbox/ymax': tf.io.VarLenFeature(tf.float32),
}

def _parse_function(example_proto):
    example = tf.io.parse_single_example(example_proto, feature_description)
    image = tf.image.decode_jpeg(example['image/encoded'], channels=3)
    image = tf.cast(image, tf.uint8)
    image = tf.image.resize(image, [320, 320], method=tf.image.ResizeMethod.BILINEAR)
    image = tf.cast(image, tf.uint8)
    labels = tf.sparse.to_dense(example['image/object/class/label'])
    bboxes = tf.stack([
        tf.sparse.to_dense(example['image/object/bbox/xmin']),
        tf.sparse.to_dense(example['image/object/bbox/ymin']),
        tf.sparse.to_dense(example['image/object/bbox/xmax']),
        tf.sparse.to_dense(example['image/object/bbox/ymax'])
    ], axis=1)
    return image, labels, bboxes

dataset = dataset.map(_parse_function)

# Initialize COCO evaluator
coco_evaluator = coco_evaluation.CocoDetectionEvaluator(
    categories=[{'id': i, 'name': category_index[i]['name']} for i in range(1, num_classes+1)]
)

# Collect ground truths, predictions, and COCO metrics
ground_truths = []
predictions = []
score_threshold = 0.1  # Keep low to capture more detections
image_id = 0

for image, labels, bboxes in dataset:
    input_tensor = tf.expand_dims(image, 0)
    detections = detect_fn(input_tensor)
    
    detection_scores = detections['detection_scores'][0].numpy()
    detection_classes = detections['detection_classes'][0].numpy().astype(np.int64)
    detection_boxes = detections['detection_boxes'][0].numpy()
    
    print(f"Image {image_id}: Num detections: {len(detection_scores)}, Max score: {np.max(detection_scores):.4f}")
    
    # Filter valid detections for COCO evaluation
    valid_indices = detection_scores >= score_threshold
    coco_pred_classes = detection_classes[valid_indices]
    coco_pred_boxes = detection_boxes[valid_indices]
    coco_pred_scores = detection_scores[valid_indices]
    
    # Convert to 0-based for confusion matrix
    cm_pred_classes = coco_pred_classes - 1
    labels = labels.numpy() - 1  # Convert to 0-based indexing
    
    # Pad/truncate predictions for confusion matrix to match ground truth labels
    if len(cm_pred_classes) == 0:
        cm_pred_classes = np.array([-1] * len(labels))  # -1 for no detection
    elif len(cm_pred_classes) < len(labels):
        cm_pred_classes = np.pad(cm_pred_classes, (0, len(labels) - len(cm_pred_classes)), constant_values=-1)
    elif len(cm_pred_classes) > len(labels):
        cm_pred_classes = cm_pred_classes[:len(labels)]
    
    ground_truths.extend(labels.tolist())
    predictions.extend(cm_pred_classes.tolist())
    
    # Add ground truth to COCO evaluator
    coco_evaluator.add_single_ground_truth_image_info(
        image_id=str(image_id),
        groundtruth_dict={
            fields.InputDataFields.groundtruth_boxes: bboxes.numpy(),
            fields.InputDataFields.groundtruth_classes: labels + 1,  # Convert back to 1-based
            fields.InputDataFields.groundtruth_difficult: np.zeros(len(labels), dtype=np.int32)
        }
    )
    
    # Add detections to COCO evaluator
    coco_evaluator.add_single_detected_image_info(
        image_id=str(image_id),
        detections_dict={
            fields.DetectionResultFields.detection_boxes: coco_pred_boxes,
            fields.DetectionResultFields.detection_scores: coco_pred_scores,
            fields.DetectionResultFields.detection_classes: coco_pred_classes  # Already 1-based
        }
    )
    
    image_id += 1

# Compute COCO metrics
coco_metrics = coco_evaluator.evaluate()

# Compute confusion matrix and per-class metrics
cm = confusion_matrix(ground_truths, predictions, labels=range(-1, num_classes))
precision, recall, f1, _ = precision_recall_fscore_support(ground_truths, predictions, labels=range(num_classes), zero_division=0)

# Save results
output_file = os.path.join(eval_dir, 'evaluation_results.txt')
with open(output_file, 'w') as f:
    f.write("COCO Metrics:\n")
    for metric_name, value in coco_metrics.items():
        f.write(f"{metric_name}: {value:.4f}\n")
        print(f"{metric_name}: {value:.4f}")

    f.write("\nPer-Class AP@0.5IOU:\n")
    per_class_ap = {}
    for i, class_name in enumerate(class_names):
        ap_key = f'AP50:{i+1}'
        if ap_key in coco_metrics:
            ap = coco_metrics[ap_key]
            per_class_ap[class_name] = ap
            f.write(f"AP@0.5IOU/{class_name}: {ap:.4f}\n")
            print(f"AP@0.5IOU/{class_name}: {ap:.4f}")

    f.write("\nPer-Class Metrics (Excluding 'No Detection'):\n")
    for i, class_name in enumerate(class_names):
        f.write(f"{class_name}:\n")
        f.write(f"  Precision: {precision[i]:.4f}\n")
        f.write(f"  Recall: {recall[i]:.4f}\n")
        f.write(f"  F1-Score: {f1[i]:.4f}\n")
        print(f"{class_name} - Precision: {precision[i]:.4f}, Recall: {recall[i]:.4f}, F1-Score: {f1[i]:.4f}")

# Plot and save confusion matrix
plt.figure(figsize=(12, 10))
cm_labels = ['No Detection'] + class_names
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=cm_labels, yticklabels=cm_labels)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Ground Truth')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
cm_plot_path = os.path.join(eval_dir, 'confusion_matrix.png')
plt.savefig(cm_plot_path, bbox_inches='tight')
plt.close()
print(f"Confusion matrix saved to {cm_plot_path}")

# Plot per-class AP
plt.figure(figsize=(10, 6))
ap_values = [per_class_ap.get(cn, 0) for cn in class_names]
plt.bar(class_names, ap_values, color='skyblue')
plt.title('Per-Class AP@0.5IOU')
plt.xlabel('Class')
plt.ylabel('AP@0.5IOU')
plt.xticks(rotation=45, ha='right')
ap_plot_path = os.path.join(eval_dir, 'per_class_ap.png')
plt.savefig(ap_plot_path, bbox_inches='tight')
plt.close()
print(f"Per-class AP plot saved to {ap_plot_path}")

# Plot per-class precision, recall, F1-score
fig, ax = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
ax[0].bar(class_names, precision, color='lightcoral')
ax[0].set_title('Per-Class Precision')
ax[0].set_ylabel('Precision')
ax[1].bar(class_names, recall, color='lightgreen')
ax[1].set_title('Per-Class Recall')
ax[1].set_ylabel('Recall')
ax[2].bar(class_names, f1, color='lightblue')
ax[2].set_title('Per-Class F1-Score')
ax[2].set_ylabel('F1-Score')
ax[2].set_xlabel('Class')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
metrics_plot_path = os.path.join(eval_dir, 'per_class_metrics.png')
plt.savefig(metrics_plot_path, bbox_inches='tight')
plt.close()
print(f"Per-class metrics plot saved to {metrics_plot_path}")

print(f"All results saved to {output_file}")