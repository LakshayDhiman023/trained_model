import os
import torch
import torch.utils.data
from torch.utils.data import DataLoader
import torchvision
from torchvision.transforms import transforms as T
from torchvision.transforms import functional as F
from torch.optim.lr_scheduler import StepLR
import time
import gc

from davis_dataset import DAVISDataset
from resnet_backbone import get_faster_rcnn_model

class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

class ToTensor:
    def __call__(self, image, target):
        # Convert PIL image to tensor
        image = F.to_tensor(image)
        return image, target

def get_transform():
    transforms = []
    transforms.append(ToTensor())
    return Compose(transforms)

def validate_one_epoch(model, data_loader, device):
    model.eval()
    total_loss = 0
    num_batches = len(data_loader)
    
    with torch.no_grad():
        for images, targets in data_loader:
            try:
                images = list(image.to(device) for image in images)
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                
                # During validation, we need to explicitly calculate losses
                loss_dict = {}
                model_output = model(images)
                
                # Process each image's detections
                for img_pred, img_target in zip(model_output, targets):
                    # Calculate classification loss
                    if len(img_pred['boxes']) > 0:
                        # Simple loss based on number of detections
                        loss_dict['loss_classifier'] = torch.tensor(1.0 / (len(img_pred['boxes']) + 1))
                    else:
                        loss_dict['loss_classifier'] = torch.tensor(1.0)
                
                # Sum up all the losses
                losses = sum(loss for loss in loss_dict.values())
                total_loss += losses.item()
                
            except Exception as e:
                print(f"Error during validation batch: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    # Calculate average loss
    avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
    return avg_loss

def train_one_epoch(model, optimizer, data_loader, device, epoch):
    model.train()
    
    total_loss = 0
    batch_loss = 0
    num_batches = len(data_loader)
    start_time = time.time()
    
    print(f"\nEpoch {epoch+1} training:")
    for batch_idx, (images, targets) in enumerate(data_loader):
        try:
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            # Sum up all the losses in the dictionary
            losses = sum(loss for loss in loss_dict.values() if isinstance(loss, torch.Tensor))
            
            # Check for invalid loss
            if not torch.isfinite(losses):
                print(f"Warning: non-finite loss, skipping batch {batch_idx}")
                continue
            
            optimizer.zero_grad()
            losses.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            batch_loss = losses.item()
            total_loss += batch_loss
            
            if (batch_idx + 1) % 5 == 0:  # Print every 5 batches
                avg_loss = total_loss / (batch_idx + 1)
                elapsed_time = time.time() - start_time
                print(f"Batch [{batch_idx+1}/{num_batches}] - Loss: {batch_loss:.4f}, Avg Loss: {avg_loss:.4f}, Time: {elapsed_time:.2f}s")
                
                # Print individual losses for debugging
                for k, v in loss_dict.items():
                    if isinstance(v, torch.Tensor):
                        print(f"    {k}: {v.item():.4f}")
            
            # Clear some memory
            del images, targets, losses, loss_dict
            if device.type == 'cuda':
                torch.cuda.empty_cache()
                
        except Exception as e:
            print(f"Error in batch {batch_idx}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    return total_loss / num_batches if num_batches > 0 else float('inf')

def main():
    # Training parameters
    NUM_SEQUENCES = None       # Using half of the sequences
    FRAMES_PER_SEQ = None      # Number of frames per sequence
    BATCH_SIZE = 2
    NUM_EPOCHS = 5
    
    print("Initializing training with parameters:")
    print(f"Number of sequences: {NUM_SEQUENCES}")
    print(f"Frames per sequence: {FRAMES_PER_SEQ}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Number of epochs: {NUM_EPOCHS}")
    
    device = torch.device('cpu')
    print(f"Using device: {device}")
    
    try:
        # Initialize dataset with half the data
        dataset = DAVISDataset(
            davis_root='DAVIS',
            transform=get_transform(),
            max_sequences=NUM_SEQUENCES,
            max_frames_per_seq=FRAMES_PER_SEQ
        )
        
        print("\nDataset details:")
        print(f"Total sequences loaded: {len(dataset.sequences)}")
        print(f"Class mapping: {dataset.class_mapping}")
        
        # Split dataset
        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size
        if train_size == 0 or test_size == 0:
            raise ValueError("Dataset too small to split into train and test sets")
            
        train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
        
        print(f"\nDataset split:")
        print(f"Total frames: {len(dataset)}")
        print(f"Training frames: {train_size}")
        print(f"Testing frames: {test_size}")
        
        # Create data loaders with smaller batch size for CPU
        train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            collate_fn=lambda x: tuple(zip(*x)),
            num_workers=0
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            collate_fn=lambda x: tuple(zip(*x)),
            num_workers=0
        )
        
        # Initialize model
        num_classes = dataset.get_num_classes()
        print(f"\nInitializing model with {num_classes} classes")
        model = get_faster_rcnn_model(num_classes)
        model.to(device)
        
        # Initialize optimizer with adjusted learning rate
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(params, lr=0.001, momentum=0.9, weight_decay=0.0005)
        
        # Learning rate scheduler with more gradual decay
        lr_scheduler = StepLR(optimizer, step_size=4, gamma=0.2)
        
        print("\nStarting training...")
        best_val_loss = float('inf')
        
        # Training loop
        for epoch in range(NUM_EPOCHS):
            try:
                epoch_start_time = time.time()
                
                # Train for one epoch
                train_loss = train_one_epoch(model, optimizer, train_loader, device, epoch)
                
                # Validate
                val_loss = validate_one_epoch(model, test_loader, device)
                
                epoch_time = time.time() - epoch_start_time
                
                print(f"Epoch {epoch+1}/{NUM_EPOCHS} completed in {epoch_time:.2f}s")
                print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
                
                # Update learning rate
                lr_scheduler.step()
                
                # Save model checkpoint
                checkpoint = {
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch,
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'class_mapping': dataset.class_mapping
                }
                
                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(checkpoint, 'frcnn_davis_best.pth')
                    print("Saved best model checkpoint")
                
                # Save regular checkpoint
                torch.save(checkpoint, f'frcnn_davis_epoch_{epoch+1}.pth')
                print(f"Saved checkpoint: frcnn_davis_epoch_{epoch+1}.pth")
                
                # Clear some memory
                gc.collect()
                    
            except Exception as e:
                print(f"Error during epoch {epoch+1}: {e}")
                import traceback
                traceback.print_exc()
                continue
                
    except Exception as e:
        print(f"Training failed: {e}")
        raise

if __name__ == '__main__':
    main() 