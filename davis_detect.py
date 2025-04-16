import os
import cv2
import torch
import torchvision
from torchvision.transforms import functional as F
from PIL import Image
import numpy as np

from resnet_backbone import get_faster_rcnn_model

# Set paths
DAVIS_PATH = "DAVIS/JPEGImages/480p"
OUTPUT_PATH = "outputs/"
MODEL_PATH = "frcnn_davis_epoch_2.pth"  # Change this to your latest model checkpoint

os.makedirs(OUTPUT_PATH, exist_ok=True)

def load_model(model_path):
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    
    # Get number of classes from the class mapping
    num_classes = len(checkpoint['class_mapping']) + 1
    
    # Initialize model
    model = get_faster_rcnn_model(num_classes)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, checkpoint['class_mapping']

def detect_and_draw(frame, model, class_mapping):
    # Convert frame to PIL Image
    frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    # Transform image
    image_tensor = F.to_tensor(frame_pil).unsqueeze(0)
    
    # Get predictions
    with torch.no_grad():
        prediction = model(image_tensor)[0]
    
    # Print all scores before filtering
    all_scores = prediction['scores'].numpy()
    if len(all_scores) > 0:
        print("\nAll confidence scores:")
        print(f"Min score: {all_scores.min():.3f}")
        print(f"Max score: {all_scores.max():.3f}")
        print(f"Mean score: {all_scores.mean():.3f}")
    
    # Get results above threshold
    keep = prediction['scores'] > 0.1
    boxes = prediction['boxes'][keep].numpy()
    labels = prediction['labels'][keep].numpy()
    scores = prediction['scores'][keep].numpy()
    
    print(f"\nDetections above threshold: {len(boxes)}")
    
    # Create a copy of the frame for drawing
    result_frame = frame.copy()
    
    # Reverse class mapping for display
    rev_class_mapping = {v: k for k, v in class_mapping.items()}
    
    # Draw boxes and labels
    for box, label, score in zip(boxes, labels, scores):
        box = box.astype(np.int32)
        class_name = rev_class_mapping.get(label.item(), "Unknown")
        print(f"Detected {class_name} with confidence {score:.3f}")
        
        # Draw rectangle with thicker lines
        cv2.rectangle(result_frame, 
                     (box[0], box[1]), 
                     (box[2], box[3]),
                     (0, 255, 0), 3)
        
        # Prepare label text
        label_text = f"{class_name}: {score:.2f}"
        
        # Get text size
        (text_width, text_height), baseline = cv2.getTextSize(
            label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
        )
        
        # Draw label background
        cv2.rectangle(result_frame,
                     (box[0], box[1] - text_height - 10),
                     (box[0] + text_width, box[1]),
                     (0, 255, 0), -1)
        
        # Draw label text in black
        cv2.putText(result_frame, 
                    label_text,
                    (box[0], box[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 0),
                    2)
    
    return result_frame

def run_on_davis():
    print("Loading model...")
    model, class_mapping = load_model(MODEL_PATH)
    print("Model loaded successfully")
    print(f"Class mapping: {class_mapping}")
    
    # Get list of all video sequences
    sequences = sorted([d for d in os.listdir(DAVIS_PATH) if os.path.isdir(os.path.join(DAVIS_PATH, d))])
    # Use only half of the sequences
    sequences = sequences[:len(sequences)//2]
    
    for video_name in sequences:
        video_path = os.path.join(DAVIS_PATH, video_name)
        print(f"\nProcessing video: {video_name}")
        output_video_path = os.path.join(OUTPUT_PATH, video_name)
        os.makedirs(output_video_path, exist_ok=True)
        
        # Get only jpg files
        frame_files = sorted([f for f in os.listdir(video_path) if f.endswith('.jpg')])
        total_frames = len(frame_files)
        
        print(f"Found {total_frames} frames")
        
        # Track confidence scores for this sequence
        sequence_scores = []
        
        for i, frame_name in enumerate(frame_files):
            print(f"\nProcessing frame {i+1}/{total_frames}: {frame_name}")
            frame_path = os.path.join(video_path, frame_name)
            frame = cv2.imread(frame_path)
            
            if frame is None:
                print(f"Error reading frame: {frame_name}")
                continue
            
            # Detect objects and draw on frame
            result = detect_and_draw(frame, model, class_mapping)
            
            # Save result
            out_path = os.path.join(output_video_path, frame_name)
            cv2.imwrite(out_path, result)
        
        print(f"Saved results to: {output_video_path}")
        
        if sequence_scores:
            avg_score = sum(sequence_scores) / len(sequence_scores)
            print(f"Average confidence score for sequence: {avg_score:.3f}")

if __name__ == "__main__":
    run_on_davis()
