import torch
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator

class ResNetBackbone(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Load pretrained ResNet50 model
        resnet = torchvision.models.resnet50(weights='ResNet50_Weights.DEFAULT')
        
        # Remove the last two layers (avg pool and fc)
        layers = list(resnet.children())[:-2]
        self.backbone = torch.nn.Sequential(*layers)
        
        # Freeze the layers
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        # Set the output channels
        self.out_channels = 2048  # ResNet50's last layer output channels
    
    def forward(self, x):
        x = self.backbone(x)
        return {'0': x}  # Return in format expected by FasterRCNN

def get_faster_rcnn_model(num_classes):
    # Get the backbone
    backbone = ResNetBackbone()
    
    # Define the anchor generator
    anchor_generator = AnchorGenerator(
        sizes=((32, 64, 128, 256, 512),),
        aspect_ratios=((0.5, 1.0, 2.0),)
    )
    
    # Define the ROI pooler
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(
        featmap_names=['0'],
        output_size=7,
        sampling_ratio=2
    )
    
    # Create the FasterRCNN model
    model = FasterRCNN(
        backbone=backbone,
        num_classes=num_classes,
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=roi_pooler,
        min_size=800,
        max_size=1333,
        rpn_pre_nms_top_n_train=2000,
        rpn_pre_nms_top_n_test=1000,
        rpn_post_nms_top_n_train=2000,
        rpn_post_nms_top_n_test=1000,
        rpn_nms_thresh=0.7,
        rpn_fg_iou_thresh=0.7,
        rpn_bg_iou_thresh=0.3,
        box_score_thresh=0.05,
        box_nms_thresh=0.5,
        box_detections_per_img=100,
        box_fg_iou_thresh=0.5,
        box_bg_iou_thresh=0.5
    )
    
    return model
