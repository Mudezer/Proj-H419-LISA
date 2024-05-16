import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

class DoubleConv(nn.Module):
    '''
    This class is used to create the double convolutional layer for the UNET model.
    We have two 2D convolutional layers with a kernel size of 3 and stride (step) of 1 and padding of 1.
    The bias is set to False as the 2D batch normalization layer is added after each convolutional layer.
    The final layer is a ReLU activation function in order to have non-zero values.
    '''
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            ## padding of 1 allows to retain the same spatial dimensions as the input feature map
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),  # kernel size = 3, stride = 1, padding = 1
            ## batch normalization normalizes the input layer by adjusting and scaling the activations
            ## stabilize and speed up the learning process
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True), # inplace = True allows to modify the input directly, without allocating additional output (save memory)
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class UNET(nn.Module):
    '''
    This class is used to create the UNET model.
    
    '''
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512],
                 ):
        super(UNET, self).__init__()
        # create the contracting path list
        self.downs = nn.ModuleList()
        # create the expansive path list
        self.ups = nn.ModuleList()
        # pooling layer to reduce the spatial dimensions of the input feature map,
        # reduce the dimension by a factor of 2
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2) # kernel size = 2 (window size), stride = 2 (step size)

        ## building the contracting path
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        ## building the expansive path
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2)
            )
            self.ups.append(DoubleConv(feature * 2, feature))
        ## bottom part is a simple double convolutional layer with the ouptput = input*2
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

        # final conv is a simple 2D convolutional layer with kernel size of 1 and stride of 1
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)
        # output a pixel-wise segmentaion map which indicates the class of each pixel

    def forward(self, x):
        # list storing the "copy and crop" from the unet architecture
        # the skip connections are used to concatenate the output of the downsampling path with the output of the upsampling path
        # they are crucial for the network to use both high-level and detailed spatial information to better localize and
        # delineate the object boundaries in the segmentation task
        skip_connections = []
        for down in self.downs:
            x = down(x)
            skip_connections.append(x) # append the output of the downsampling path
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx // 2]

            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx + 1](concat_skip)

        return self.final_conv(x)


def test():
    ''''
    Simple test function to see if the model is working and if the output is of the same size as the input.
    Allows as well to see if the model is able to run on the device.
    '''''
    device = getDevice()
    # x= torch.randn((3, 3, 1280, 1918)).to(device) # to big for the device
    x = torch.randn((3, 3, 640, 960)).to(device)  # batch size = 3, 1 channel,  image
    model = UNET(in_channels=3, out_channels=1).to(device)
    preds = model(x)
    print(preds.shape)
    print(x.shape)

def getDevice():
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    print(f"Using {device} device")
    return device


if __name__ == "__main__":

    test()