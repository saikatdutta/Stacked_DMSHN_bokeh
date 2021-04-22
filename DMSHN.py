import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        #Conv1
        self.layer1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1)
            )
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1)
            )
        #Conv2
        self.layer5 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.layer6 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1)
            )
        self.layer7 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1)
            )
        #Conv3
        self.layer9 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.layer10 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1)
            )
        self.layer11 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1)
            )
        
    def forward(self, x):
        #Conv1
        x = self.layer1(x)
        x = self.layer2(x) + x
        x = self.layer3(x) + x
        #Conv2
        x = self.layer5(x)
        x = self.layer6(x) + x
        x = self.layer7(x) + x
        #Conv3
        x = self.layer9(x)    
        x = self.layer10(x) + x
        x = self.layer11(x) + x 
        return x

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()        
        # Deconv3
        self.layer13 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1)
            )
        self.layer14 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1)
            )
        self.layer16 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        #Deconv2
        self.layer17 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1)
            )
        self.layer18 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1)
            )
        self.layer20 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        #Deconv1
        self.layer21 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1)
            )
        self.layer22 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1)
            )
        self.layer24 = nn.Conv2d(32, 3, kernel_size=3, padding=1)
        
    def forward(self,x):        
        #Deconv3
        x = self.layer13(x) + x
        x = self.layer14(x) + x
        x = self.layer16(x)                
        #Deconv2
        x = self.layer17(x) + x
        x = self.layer18(x) + x
        x = self.layer20(x)
        #Deconv1
        x = self.layer21(x) + x
        x = self.layer22(x) + x
        x = self.layer24(x)
        return x

class DMSHN(nn.Module):
    def __init__(self):
        super(DMSHN, self).__init__()
        self.encoder_lv1 = Encoder()
        self.encoder_lv2 = Encoder()
        self.encoder_lv3 = Encoder()

        self.decoder_lv1 = Decoder()
        self.decoder_lv2 = Decoder()
        self.decoder_lv3 = Decoder()

    def forward(self,images_lv1):
        H = images_lv1.size(2)
        W = images_lv1.size(3)

        images_lv2 = F.interpolate(images_lv1, scale_factor = 0.5, mode = 'bilinear')
        images_lv3 = F.interpolate(images_lv2, scale_factor = 0.5, mode = 'bilinear')

        feature_lv3 = self.encoder_lv3(images_lv3)
        residual_lv3 = self.decoder_lv3(feature_lv3)
        out_lv3 = images_lv3 + residual_lv3 

        residual_lv3 = F.interpolate(residual_lv3, scale_factor=2, mode= 'bilinear')
        feature_lv3 = F.interpolate(feature_lv3, scale_factor=2, mode= 'bilinear')
        feature_lv2 = self.encoder_lv2(images_lv2 + residual_lv3)
        residual_lv2 = self.decoder_lv2(feature_lv2 + feature_lv3)
        out_lv2 = images_lv2 + residual_lv2

        residual_lv2 = F.interpolate(residual_lv2, scale_factor=2, mode= 'bilinear')
        feature_lv2 = F.interpolate(feature_lv2, scale_factor=2, mode= 'bilinear')
        feature_lv1 = self.encoder_lv1(images_lv1 + residual_lv2)
        bokeh_image = self.decoder_lv1(feature_lv1 + feature_lv2)


        return bokeh_image

'''
model =  DMSHN().cuda()
inp = torch.randn(1,3,1024,1024).cuda()
y = model(inp)
print (y.shape)
'''
