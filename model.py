import torch 
import torchvision
import pytorch_lightning as pl
import segmentation_models_pytorch as smp

class get_model(torch.nn.Module):
    def __init__(self,
                 encoder="resnet34"
                 ) -> None:
        super().__init__()
        self.model = smp.Unet(
                            encoder_name=encoder,        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
                            encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
                            in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
                            # classes=3,                      # model output channels (number of classes in your dataset))
                 )

    def forward(self, img):
        outputs = self.model(img)
        return outputs