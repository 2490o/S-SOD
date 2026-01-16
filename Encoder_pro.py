import torch
import torch.nn as nn
import pvt_v2
import torchvision.models as models
from torch.nn import functional as F
from singe_prototype import Singe_prototype
# from model import convnext_small as create_model
def convblock(in_, out_, ks, st, pad):
    return nn.Sequential(
        nn.Conv2d(in_, out_, ks, st, pad),
        nn.BatchNorm2d(out_),
        nn.ReLU(inplace=True)
    )

class CA(nn.Module):
    def __init__(self, in_ch):
        super(CA, self).__init__()
        self.avg_weight = nn.AdaptiveAvgPool2d(1)
        self.max_weight = nn.AdaptiveMaxPool2d(1)
        self.fus = nn.Sequential(
            nn.Conv2d(in_ch, in_ch // 2, 1, 1, 0),
            nn.ReLU(),
            nn.Conv2d(in_ch // 2, in_ch, 1, 1, 0),
        )
        self.c_mask = nn.Sigmoid()

    def forward(self, x):
        avg_map_c = self.avg_weight(x)
        max_map_c = self.max_weight(x)
        c_mask = self.c_mask(torch.add(self.fus(avg_map_c), self.fus(max_map_c)))
        return torch.mul(x, c_mask)

class R_CLCM(nn.Module):
    def __init__(self, in_1, in_2, in_3, in_4):
        super(R_CLCM, self).__init__()
        self.ca1 = CA(in_1)
        self.ca2 = CA(in_2)
        self.ca3 = CA(in_3)
        self.ca4 = CA(in_4)
        # self.p1 = Singe_prototype(64, 20)
        # self.p2 = Singe_prototype(128, 20)
        # self.p3 = Singe_prototype(320, 20)
        self.p4 = Singe_prototype(64, 20)
        self.p4_512 = Singe_prototype(512, 20)
        self.conv_r1 = convblock(in_1, in_1, 1, 1, 0)
        self.conv_r2 = convblock(in_2, in_1, 3, 1, 1)
        self.conv_r3 = convblock(in_3, in_1, 3, 1, 1)
        self.conv_r4 = convblock(in_4, in_1, 3, 1, 1)

        self.conv_n1 = nn.Conv2d(in_1, in_1 // 4, 1, 1, 0)
        self.conv_n2 = nn.Conv2d(in_1, in_1 // 4, 1, 1, 0)
        self.conv_n3 = nn.Conv2d(in_1, in_1 // 4, 1, 1, 0)
        self.conv_n4 = nn.Conv2d(in_1, in_1 // 4, 1, 1, 0)

        self.conv_3_r = convblock(in_1 // 4, in_1, 1, 1, 0)
        self.conv_2_r = convblock(in_1 // 4, in_1, 1, 1, 0)
        self.conv_1_r = convblock(in_1 // 4, in_1, 1, 1, 0)

        self.conv_out1 = nn.Conv2d(in_1, in_1, 1, 1, 0)
        self.conv_out2 = nn.Conv2d(in_1, in_2, 3, 1, 1)
        self.conv_out3 = nn.Conv2d(in_1, in_3, 3, 1, 1)
        self.softmax = nn.Softmax(dim=-1)
        self.gam3 = nn.Parameter(torch.zeros(1))
        self.gam2 = nn.Parameter(torch.zeros(1))
        self.gam1 = nn.Parameter(torch.zeros(1))

    def forward(self, x1, x2, x3, x4):
        r1 = self.conv_r1(self.ca1(x1))
        r2 = self.conv_r2(self.ca2(x2))
        r3 = self.conv_r3(self.ca3(x3))
        r4 = self.conv_r4(self.ca4(x4))

        r4 = self.p4(r4)

        b, c, h1, w1 = r1.size()
        b, c, h2, w2 = r2.size()
        b, c, h3, w3 = r3.size()
        b, c, h4, w4 = r4.size()

        r_4 = self.conv_n4(r4).view(b, -1, h4 * w4)  # b, c, l4
        r_4_t = r_4.permute(0, 2, 1)  # b, l4, c
        r_3 = self.conv_n3(r3).view(b, -1, h3 * w3)  # b, c, l3

        r_4_3 = torch.bmm(r_4_t, r_3)  # b, l4, l3
        att_r_4_3 = self.softmax(r_4_3)
        r_3_4 = torch.bmm(r_4, att_r_4_3)  # b, c, l3
        r_3_in = r_3_4 + r_3  # b, c, l3

        r_3_in_t = r_3_in.permute(0, 2, 1)  # b, l3, c
        r_2 = self.conv_n2(r2).view(b, -1, h2 * w2)  # b, c, l2

        r_3_2 = torch.bmm(r_3_in_t, r_2)  # b, l3, l2
        att_r_3_2 = self.softmax(r_3_2)
        r_2_3 = torch.bmm(r_3_in, att_r_3_2)  # b, c, l2
        r_2_in = r_2_3 + r_2

        r_2_in_t = r_2_in.permute(0, 2, 1)  # b, l2, c
        r_1 = self.conv_n1(r1).view(b, -1, h1 * w1)  # b, c, l1

        r_2_1 = torch.bmm(r_2_in_t, r_1)  # b, l2, l1
        att_r_2_1 = self.softmax(r_2_1)
        r_1_2 = torch.bmm(r_2_in, att_r_2_1)  # b, c, l1
        r_1_in = r_1_2 + r_1

        r_3_out = self.conv_3_r(r_3_in.view(b, -1, h3, w3))
        out_r3 = self.gam3 * r_3_out + r3
        r_2_out = self.conv_2_r(r_2_in.view(b, -1, h2, w2))
        out_r2 = self.gam2 * r_2_out + r2
        r_1_out = self.conv_1_r(r_1_in.view(b, -1, h1, w1))
        out_r1 = self.gam1 * r_1_out + r1

        return self.conv_out1(out_r1), self.conv_out2(out_r2), self.conv_out3(out_r3)

    class R_CLCM(nn.Module):
        def __init__(self, in_1, in_2, in_3, in_4):
            super(R_CLCM, self).__init__()
            self.ca1 = CA(in_1)
            self.ca2 = CA(in_2)
            self.ca3 = CA(in_3)
            self.ca4 = CA(in_4)
            # self.p1 = Singe_prototype(64, 20)
            # self.p2 = Singe_prototype(128, 20)
            # self.p3 = Singe_prototype(320, 20)
            self.p4 = Singe_prototype(64, 20)
            self.p4_512 = Singe_prototype(512, 20)
            self.conv_r1 = convblock(in_1, in_1, 1, 1, 0)
            self.conv_r2 = convblock(in_2, in_1, 3, 1, 1)
            self.conv_r3 = convblock(in_3, in_1, 3, 1, 1)
            self.conv_r4 = convblock(in_4, in_1, 3, 1, 1)

            self.conv_n1 = nn.Conv2d(in_1, in_1 // 4, 1, 1, 0)
            self.conv_n2 = nn.Conv2d(in_1, in_1 // 4, 1, 1, 0)
            self.conv_n3 = nn.Conv2d(in_1, in_1 // 4, 1, 1, 0)
            self.conv_n4 = nn.Conv2d(in_1, in_1 // 4, 1, 1, 0)

            self.conv_3_r = convblock(in_1 // 4, in_1, 1, 1, 0)
            self.conv_2_r = convblock(in_1 // 4, in_1, 1, 1, 0)
            self.conv_1_r = convblock(in_1 // 4, in_1, 1, 1, 0)

            self.conv_out1 = nn.Conv2d(in_1, in_1, 1, 1, 0)
            self.conv_out2 = nn.Conv2d(in_1, in_2, 3, 1, 1)
            self.conv_out3 = nn.Conv2d(in_1, in_3, 3, 1, 1)
            self.softmax = nn.Softmax(dim=-1)
            self.gam3 = nn.Parameter(torch.zeros(1))
            self.gam2 = nn.Parameter(torch.zeros(1))
            self.gam1 = nn.Parameter(torch.zeros(1))

        def forward(self, x1, x2, x3, x4):
            r1 = self.conv_r1(self.ca1(x1))
            r2 = self.conv_r2(self.ca2(x2))
            r3 = self.conv_r3(self.ca3(x3))
            r4 = self.conv_r4(self.ca4(x4))

            r4 = self.p4(r4)

            b, c, h1, w1 = r1.size()
            b, c, h2, w2 = r2.size()
            b, c, h3, w3 = r3.size()
            b, c, h4, w4 = r4.size()

            r_4 = self.conv_n4(r4).view(b, -1, h4 * w4)  # b, c, l4
            r_4_t = r_4.permute(0, 2, 1)  # b, l4, c
            r_3 = self.conv_n3(r3).view(b, -1, h3 * w3)  # b, c, l3

            r_4_3 = torch.bmm(r_4_t, r_3)  # b, l4, l3
            att_r_4_3 = self.softmax(r_4_3)
            r_3_4 = torch.bmm(r_4, att_r_4_3)  # b, c, l3
            r_3_in = r_3_4 + r_3  # b, c, l3

            r_3_in_t = r_3_in.permute(0, 2, 1)  # b, l3, c
            r_2 = self.conv_n2(r2).view(b, -1, h2 * w2)  # b, c, l2

            r_3_2 = torch.bmm(r_3_in_t, r_2)  # b, l3, l2
            att_r_3_2 = self.softmax(r_3_2)
            r_2_3 = torch.bmm(r_3_in, att_r_3_2)  # b, c, l2
            r_2_in = r_2_3 + r_2

            r_2_in_t = r_2_in.permute(0, 2, 1)  # b, l2, c
            r_1 = self.conv_n1(r1).view(b, -1, h1 * w1)  # b, c, l1

            r_2_1 = torch.bmm(r_2_in_t, r_1)  # b, l2, l1
            att_r_2_1 = self.softmax(r_2_1)
            r_1_2 = torch.bmm(r_2_in, att_r_2_1)  # b, c, l1
            r_1_in = r_1_2 + r_1

            r_3_out = self.conv_3_r(r_3_in.view(b, -1, h3, w3))
            out_r3 = self.gam3 * r_3_out + r3
            r_2_out = self.conv_2_r(r_2_in.view(b, -1, h2, w2))
            out_r2 = self.gam2 * r_2_out + r2
            r_1_out = self.conv_1_r(r_1_in.view(b, -1, h1, w1))
            out_r1 = self.gam1 * r_1_out + r1

            return self.conv_out1(out_r1), self.conv_out2(out_r2), self.conv_out3(out_r3)

class R_CLCM2(nn.Module):
    def __init__(self, in_1, in_2, in_3, in_4):
        super(R_CLCM2, self).__init__()
        self.ca1 = CA(in_1)
        self.ca2 = CA(in_2)
        self.ca3 = CA(in_3)
        self.ca4 = CA(in_4)
        # self.p1 = Singe_prototype(64, 20)
        # self.p2 = Singe_prototype(128, 20)
        # self.p3 = Singe_prototype(320, 20)
        self.p4 = Singe_prototype(64, 20)
        self.p4_512 = Singe_prototype(512, 20)
        self.conv_r1 = convblock(in_1, in_1, 1, 1, 0)
        self.conv_r2 = convblock(in_2, in_1, 3, 1, 1)
        self.conv_r3 = convblock(in_3, in_1, 3, 1, 1)
        self.conv_r4 = convblock(in_4, in_1, 3, 1, 1)

        self.conv_n1 = nn.Conv2d(in_1, in_1 // 4, 1, 1, 0)
        self.conv_n2 = nn.Conv2d(in_1, in_1 // 4, 1, 1, 0)
        self.conv_n3 = nn.Conv2d(in_1, in_1 // 4, 1, 1, 0)
        self.conv_n4 = nn.Conv2d(in_1, in_1 // 4, 1, 1, 0)

        self.conv_3_r = convblock(in_1 // 4, in_1, 1, 1, 0)
        self.conv_2_r = convblock(in_1 // 4, in_1, 1, 1, 0)
        self.conv_1_r = convblock(in_1 // 4, in_1, 1, 1, 0)

        self.conv_out1 = nn.Conv2d(in_1, in_1, 1, 1, 0)
        self.conv_out2 = nn.Conv2d(in_1, in_2, 3, 1, 1)
        self.conv_out3 = nn.Conv2d(in_1, in_3, 3, 1, 1)
        self.softmax = nn.Softmax(dim=-1)
        self.gam3 = nn.Parameter(torch.zeros(1))
        self.gam2 = nn.Parameter(torch.zeros(1))
        self.gam1 = nn.Parameter(torch.zeros(1))

    def forward(self, x1, x2, x3, x4):
        r1 = self.conv_r1(self.ca1(x1))
        r2 = self.conv_r2(self.ca2(x2))
        r3 = self.conv_r3(self.ca3(x3))
        r4 = self.conv_r4(self.ca4(x4))

        r4 = self.p4_512(r4)

        b, c, h1, w1 = r1.size()
        b, c, h2, w2 = r2.size()
        b, c, h3, w3 = r3.size()
        b, c, h4, w4 = r4.size()

        r_4 = self.conv_n4(r4).view(b, -1, h4 * w4)  # b, c, l4
        r_4_t = r_4.permute(0, 2, 1)  # b, l4, c
        r_3 = self.conv_n3(r3).view(b, -1, h3 * w3)  # b, c, l3

        r_4_3 = torch.bmm(r_4_t, r_3)  # b, l4, l3
        att_r_4_3 = self.softmax(r_4_3)
        r_3_4 = torch.bmm(r_4, att_r_4_3)  # b, c, l3
        r_3_in = r_3_4 + r_3  # b, c, l3

        r_3_in_t = r_3_in.permute(0, 2, 1)  # b, l3, c
        r_2 = self.conv_n2(r2).view(b, -1, h2 * w2)  # b, c, l2

        r_3_2 = torch.bmm(r_3_in_t, r_2)  # b, l3, l2
        att_r_3_2 = self.softmax(r_3_2)
        r_2_3 = torch.bmm(r_3_in, att_r_3_2)  # b, c, l2
        r_2_in = r_2_3 + r_2

        r_2_in_t = r_2_in.permute(0, 2, 1)  # b, l2, c
        r_1 = self.conv_n1(r1).view(b, -1, h1 * w1)  # b, c, l1

        r_2_1 = torch.bmm(r_2_in_t, r_1)  # b, l2, l1
        att_r_2_1 = self.softmax(r_2_1)
        r_1_2 = torch.bmm(r_2_in, att_r_2_1)  # b, c, l1
        r_1_in = r_1_2 + r_1

        r_3_out = self.conv_3_r(r_3_in.view(b, -1, h3, w3))
        out_r3 = self.gam3 * r_3_out + r3
        r_2_out = self.conv_2_r(r_2_in.view(b, -1, h2, w2))
        out_r2 = self.gam2 * r_2_out + r2
        r_1_out = self.conv_1_r(r_1_in.view(b, -1, h1, w1))
        out_r1 = self.gam1 * r_1_out + r1

        return self.conv_out1(out_r1), self.conv_out2(out_r2), self.conv_out3(out_r3)

class Transformer(nn.Module):
    def __init__(self, backbone, pretrained=None):
        super().__init__()
        self.encoder = getattr(pvt_v2, backbone)()
        if pretrained:
            checkpoint = torch.load('/home/zzh/SOD/SOD/pvt_v2_b3.pth', map_location='cpu')
            if 'model' in checkpoint:
                checkpoint_model = checkpoint['model']
            else:
                checkpoint_model = checkpoint
            state_dict = self.encoder.state_dict()
            for k in ['head.weight', 'head.bias', 'head_dist.weight', 'head_dist.bias']:
                if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                    print(f"Removing key {k} from pretrained checkpoint")
                    del checkpoint_model[k]
            self.encoder.load_state_dict(checkpoint_model, strict=False)

def Encoder():
    model = Transformer('pvt_v2_b3', pretrained=True)
    return model

class Mnet(nn.Module):
    def __init__(self):
        super(Mnet, self).__init__()
        model = Encoder()
        self.rgb_net = model.encoder
        self.t_net = model.encoder
        self.R_CLCM_1 = R_CLCM(64, 128, 320, 512)
        self.R_CLCM_2 = R_CLCM2(512, 320, 128, 64)
        self.conv_3 = convblock(320 * 2, 320, 3, 1, 1)
        self.conv_2 = convblock(128 * 2, 128, 3, 1, 1)
        self.sig = nn.Sigmoid()
        self.p1 = Singe_prototype(512, 20)
        self.up_4 = convblock(512, 64, 3, 1, 1)
        self.up_3 = convblock(320, 64, 3, 1, 1)
        self.up_2 = convblock(128, 64, 3, 1, 1)
        self.up_1 = convblock(64, 64, 1, 1, 0)

        self.conv_s1 = convblock(64 * 3, 64, 3, 1, 1)
        self.score1 = nn.Conv2d(64, 1, 1, 1, 0)

        self.conv_s2 = convblock(64 * 3, 64, 3, 1, 1)
        self.score2 = nn.Conv2d(64, 1, 1, 1, 0)

        self.score = nn.Conv2d(2, 1, 1, 1, 0)

    def forward(self, imgs):
        img_1, img_2, img_3, img_4 = self.rgb_net(imgs)

        out_r1234, out_r234, out_r34 = self.R_CLCM_1(img_1, img_2, img_3, img_4)
        out_r4321, out_r321, out_r21 = self.R_CLCM_2(img_4, img_3, img_2, img_1)
        out_r3421 = self.conv_3(torch.cat((out_r321 + out_r34, out_r321 * out_r34), 1))
        out_r2341 = self.conv_2(torch.cat((out_r21 + out_r234, out_r21 * out_r234), 1))

        clf_4 = self.up_4(F.interpolate(out_r4321, out_r1234.size()[2:], mode='bilinear', align_corners=True))
        clf_3 = self.up_3(F.interpolate(out_r3421, out_r1234.size()[2:], mode='bilinear', align_corners=True))
        clf_2 = self.up_2(F.interpolate(out_r2341, out_r1234.size()[2:], mode='bilinear', align_corners=True))
        clf_1 = self.up_1(out_r1234)

        s_clf1 = self.conv_s1(torch.cat((clf_4, clf_3, clf_2), 1))
        score_2 = self.score1(F.interpolate(s_clf1, (384, 384), mode='bilinear', align_corners=True))
        s_clf2 = self.conv_s2(torch.cat((s_clf1, clf_1, clf_2), 1))
        score_1 = self.score2(F.interpolate(s_clf2, (384, 384), mode='bilinear', align_corners=True))

        score = self.score(torch.cat((score_1 + torch.mul(score_1, self.sig(score_2)),
                                      score_2 + torch.mul(score_2, self.sig(score_1))), 1))

        return score, score_1, score_2, self.sig(score)



