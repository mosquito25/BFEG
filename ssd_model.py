from src.res50_backbone import resnet50
from torch import nn, Tensor
import torch
from torch.jit.annotations import Optional, List, Dict, Tuple, Module
from src.utils import dboxes300_coco, Encoder, PostProcess
from torchvision.utils import save_image
import random
import vgg
from src.SSIM import SSIM
from src.Unet_CBA3 import UNetCBA3
from src.Attation import CBAM
from src.fa import Fa
import torch.nn.functional as F
from src.CBA_parts import PALayer,CALayer
from src.CBA_parts3 import DeformConv2d

class Backbone(nn.Module):
    def __init__(self, pretrain_path=None):
        super(Backbone, self).__init__()
        net = resnet50()
        self.out_channels = [1024, 512, 512, 256, 256, 256]

        if pretrain_path is not None:
            net.load_state_dict(torch.load(pretrain_path))

        self.conv1=nn.Sequential(*list(net.children())[:3])
        self.conv2=nn.Sequential(*list(net.children())[3:5])
        self.conv3 = nn.Sequential(*list(net.children())[5:7])
        conv3_block1 = self.conv3[-1][0]
        conv3_block1.conv1.stride = (1, 1)
        conv3_block1.conv2.stride = (1, 1)
        conv3_block1.downsample[0].stride = (1, 1)
    def forward(self, x):
        featurelist = [] #300*300*3   150*150*64   75*75*256   38*38*1024
        featurelist.append(x)
        x=self.conv1(x)
        featurelist.append(x)
        x=self.conv2(x)
        featurelist.append(x)
        x=self.conv3(x)
        featurelist.append(x)
        return featurelist


class SSD300(nn.Module):
    def __init__(self, backbone=None, num_classes=21):
        super(SSD300, self).__init__()
        if backbone is None:
            raise Exception("backbone is None")
        if not hasattr(backbone, "out_channels"):
            raise Exception("the backbone not has attribute: out_channel")
        #backbone you 300 ,150 75, 38
        self.feature_extractor = backbone
        self.unetrec=UNetCBA3()
        self.num_classes = num_classes
        # out_channels = [1024, 512, 512, 256, 256, 256] for resnet50
        self._build_additional_features(self.feature_extractor.out_channels)
        self.num_defaults = [4, 6, 6, 6, 4, 4]
        location_extractors = []
        confidence_extractors = []

        # out_channels = [1024, 512, 512, 256, 256, 256] for resnet50
        for nd, oc in zip(self.num_defaults, self.feature_extractor.out_channels):
            # nd is number_default_boxes, oc is output_channel
            location_extractors.append(nn.Conv2d(oc, nd * 4, kernel_size=3, padding=1))
            confidence_extractors.append(nn.Conv2d(oc, nd * self.num_classes, kernel_size=3, padding=1))

        self.loc = nn.ModuleList(location_extractors)
        self.conf = nn.ModuleList(confidence_extractors)
        self._init_weights()

        default_box = dboxes300_coco()
        self.compute_loss = Loss(default_box)
        self.recloss = recloss()
        self.encoder = Encoder(default_box)
        self.postprocess = PostProcess(default_box)

        self.down1=nn.Sequential(
            nn.MaxPool2d(4),
            nn.Conv2d(3,256,kernel_size=3,padding=1),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(4),  #19
            nn.Conv2d(256,512,kernel_size=2,padding=1),
            nn.LeakyReLU(inplace=True)
        )

        self.down2=nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(512,512,kernel_size=2,padding=1),
            nn.LeakyReLU(inplace=True)
        )



        self.Connv1=nn.Conv2d(1024,512,kernel_size=3,padding=1)
        self.Connv2 =nn.Conv2d(1024, 512, kernel_size=3, padding=1)

    def _build_additional_features(self, input_size):
        additional_blocks = []
        middle_channels = [256, 256, 128, 128, 128]
        for i, (input_ch, output_ch, middle_ch) in enumerate(zip(input_size[:-1], input_size[1:], middle_channels)):
            padding, stride = (1, 2) if i < 3 else (0, 1)
            layer = nn.Sequential(
                nn.Conv2d(input_ch, middle_ch, kernel_size=1, bias=False),
                nn.BatchNorm2d(middle_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(middle_ch, output_ch, kernel_size=3, padding=padding, stride=stride, bias=False),
                nn.BatchNorm2d(output_ch),
                nn.ReLU(inplace=True),
            )
            additional_blocks.append(layer)
        self.additional_blocks = nn.ModuleList(additional_blocks)
    def _init_weights(self):
        layers = [*self.additional_blocks, *self.loc, *self.conf]
        for layer in layers:
            for param in layer.parameters():
                if param.dim() > 1:
                    nn.init.xavier_uniform_(param)
    def bbox_view(self, features, loc_extractor, conf_extractor):
        locs = []
        confs = []
        for f, l, c in zip(features, loc_extractor, conf_extractor):
            locs.append(l(f).view(f.size(0), 4, -1))
            confs.append(c(f).view(f.size(0), self.num_classes, -1))

        locs, confs = torch.cat(locs, 2).contiguous(), torch.cat(confs, 2).contiguous()
        return locs, confs

    def forward(self, image, clear,targets=None):
        xx = self.feature_extractor(image)
        rec=self.unetrec(xx)
        x=xx[3]

        detection_features = torch.jit.annotate(List[Tensor], [])  # [x]
        detection_features.append(x)
        for i,layer in enumerate(self.additional_blocks):
            x = layer(x)
            if i==0:
                xx_rec=self.down1(rec)
                diffY = xx_rec.size()[2] - x.size()[2]
                diffX = xx_rec.size()[3] - x.size()[3]
                x1 = F.pad(x, [diffX // 2, diffX - diffX // 2,diffY // 2, diffY - diffY // 2])
                x = torch.cat([xx_rec, x1], dim=1)
                x=self.Connv1(x)

            if i==1:
                xx_rec=self.down2(xx_rec)
                diffY = xx_rec.size()[2] - x.size()[2]
                diffX = xx_rec.size()[3] - x.size()[3]
                x1 = F.pad(x, [diffX // 2, diffX - diffX // 2,diffY // 2, diffY - diffY // 2])
                x = torch.cat([xx_rec, x1], dim=1)
                x=self.Connv2(x)
            detection_features.append(x)
        locs, confs = self.bbox_view(detection_features, self.loc, self.conf)

        if self.training:
            if targets is None:
                raise ValueError("In training mode, targets should be passed")
            bboxes_out = targets['boxes']
            bboxes_out = bboxes_out.transpose(1, 2).contiguous()
            labels_out = targets['labels']
            loss1,loss2,loss3=self.recloss(rec,clear)
            loss = self.compute_loss(locs, confs, bboxes_out, labels_out)
            return {"total_losses": 1.2*loss+(loss1+loss2+loss3),"objectloss":loss,"l1loss":loss1,"l2loss":loss2,"per loss":loss3}
        results = self.postprocess(locs, confs)
        return results,rec


class Loss(nn.Module):

    def __init__(self, dboxes):
        super(Loss, self).__init__()
        self.scale_xy = 1.0 / dboxes.scale_xy  # 10
        self.scale_wh = 1.0 / dboxes.scale_wh  # 5

        self.location_loss = nn.SmoothL1Loss(reduction='none')
        # [num_anchors, 4] -> [4, num_anchors] -> [1, 4, num_anchors]
        self.dboxes = nn.Parameter(dboxes(order="xywh").transpose(0, 1).unsqueeze(dim=0),
                                   requires_grad=False)

        self.confidence_loss = nn.CrossEntropyLoss(reduction='none')

    def _location_vec(self, loc):
        gxy = self.scale_xy * (loc[:, :2, :] - self.dboxes[:, :2, :]) / self.dboxes[:, 2:, :]  # Nx2x8732
        gwh = self.scale_wh * (loc[:, 2:, :] / self.dboxes[:, 2:, :]).log()  # Nx2x8732
        return torch.cat((gxy, gwh), dim=1).contiguous()

    def forward(self, ploc, plabel, gloc, glabel):

        mask = torch.gt(glabel, 0)
        pos_num = mask.sum(dim=1)
        vec_gd = self._location_vec(gloc)
        loc_loss = self.location_loss(ploc, vec_gd).sum(dim=1)
        loc_loss = (mask.float() * loc_loss).sum(dim=1)
        con = self.confidence_loss(plabel, glabel)

        con_neg = con.clone()
        con_neg[mask] = 0.0

        _, con_idx = con_neg.sort(dim=1, descending=True)
        _, con_rank = con_idx.sort(dim=1)
        neg_num = torch.clamp(3 * pos_num, max=mask.size(1)).unsqueeze(-1)
        neg_mask = torch.lt(con_rank, neg_num)
        con_loss = (con * (mask.float() + neg_mask.float())).sum(dim=1)

        total_loss = loc_loss + con_loss
        num_mask = torch.gt(pos_num, 0).float()
        pos_num = pos_num.float().clamp(min=1e-6)
        ret = (total_loss * num_mask / pos_num).mean(dim=0)
        return ret

class recloss(nn.Module):
    def __init__(self):
        super(recloss,self).__init__()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.l1loss=nn.L1Loss().to(device)
        self.mse=nn.MSELoss().to(device)
        dtype=torch.cuda.FloatTensor
        self.recvgg= vgg.Vgg16().type(dtype)
        self.ssim=SSIM().to(device)
    def forward(self,rec,clear):
        loss1=10*self.l1loss(rec,clear)
        rec_vgg=self.recvgg(rec)[3]
        clear_vgg=self.recvgg(clear)[3]

        loss2=self.mse(rec_vgg,clear_vgg)
        loss3=1-self.ssim(rec,clear)
        return loss1,loss2,loss3



