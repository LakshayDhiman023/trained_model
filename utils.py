import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn

# Load pretrained model
model = fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()
