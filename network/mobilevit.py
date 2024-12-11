import torch
import torch.nn as nn



def conv_1x1_bn(inp, oup, padding=0):
    return nn.Sequential(
        nn.Conv2d(inp, oup, kernel_size=1, stride=1, padding=padding, bias=False),
        nn.BatchNorm2d(oup),
        nn.SiLU()
    )


def conv_nxn_bn(inp, oup, kernel_size=3, stride=1, padding=1):
    return nn.Sequential(
        nn.Conv2d(inp, oup, kernel_size, stride, padding=padding, bias=False),
        nn.BatchNorm2d(oup),
        nn.SiLU()
    )


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b p n (h d) -> b p h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b p h n d -> b p n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads, dim_head, dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout))
            ]))
    
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class MV2Block(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=2, stride=1, padding=0, expansion=16, skip_con=True):
        super().__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(in_ch * expansion)
        #self.use_res_connect = self.stride == 1 and in_ch == out_ch
        self.skip_con = skip_con

        if expansion == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(in_ch, hidden_dim, kernel_size=kernel_size, stride=stride, padding=padding, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(),
                # pw-linear
                nn.Conv2d(hidden_dim, out_ch, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(in_ch, hidden_dim, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=kernel_size, stride=stride, padding=padding, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(),
                # pw-linear
                nn.Conv2d(hidden_dim, out_ch, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_ch),
            )


    def forward(self, x):
        if self.skip_con:
            return x + self.conv(x)
        else:
            x = self.conv(x)

            return x


class MobileViTBlock(nn.Module):
    def __init__(self, dim, depth, channel, kernel_size, patch_size, mlp_dim, dropout=0.):
        super().__init__()
        self.ph, self.pw = patch_size

        self.conv1 = conv_nxn_bn(channel, channel, kernel_size=3)
        self.conv2 = conv_1x1_bn(channel, dim)

        self.transformer = Transformer(dim, depth, 4, 8, mlp_dim, dropout)

        self.conv3 = conv_1x1_bn(dim, channel)
        self.conv4 = conv_nxn_bn(2 * channel, channel, kernel_size=3)
    
    def forward(self, x):
        y = x.clone()
        #print("MobileViTBlock start")
        # Local representations
        #print(f"Transformer input : {x.size()}")
        x = self.conv1(x)
        #print(f"Transformer after nxn : {x.size()}")
        x = self.conv2(x)
        #print(f"Transformer after 1x1 : {x.size()}")

        # Global representations
        _, _, h, w = x.shape
        x = rearrange(x, 'b d (h ph) (w pw) -> b (ph pw) (h w) d', ph=self.ph, pw=self.pw)
        x = self.transformer(x)
        x = rearrange(x, 'b (ph pw) (h w) d -> b d (h ph) (w pw)', h=h//self.ph, w=w//self.pw, ph=self.ph, pw=self.pw)

        # Fusion
        x = self.conv3(x)
        x = torch.cat((x, y), 1)
        x = self.conv4(x)
        #print("MobileViTBlock end")

        return x


class MobileViT(nn.Module):
    def __init__(self, image_size, dims, channels, num_classes, expansion=4, kernel_size=3, patch_size=(2, 2)):
        super().__init__()
        ih, iw = image_size
        ph, pw = patch_size
        assert ih % ph == 0 and iw % pw == 0

        L = [2, 4, 3]

        self.conv1 = conv_nxn_bn(3, channels[0], stride=2)

        self.mv2 = nn.ModuleList([])
        self.mv2.append(MV2Block(channels[0], channels[1], 1, expansion))
        self.mv2.append(MV2Block(channels[1], channels[2], 2, expansion))
        self.mv2.append(MV2Block(channels[2], channels[3], 1, expansion))
        self.mv2.append(MV2Block(channels[2], channels[3], 1, expansion))   # Repeat
        self.mv2.append(MV2Block(channels[3], channels[4], 2, expansion))
        self.mv2.append(MV2Block(channels[5], channels[6], 2, expansion))
        self.mv2.append(MV2Block(channels[7], channels[8], 2, expansion))
        
        self.mvit = nn.ModuleList([])
        self.mvit.append(MobileViTBlock(dims[0], L[0], channels[5], kernel_size, patch_size, int(dims[0]*2)))
        self.mvit.append(MobileViTBlock(dims[1], L[1], channels[7], kernel_size, patch_size, int(dims[1]*4)))
        self.mvit.append(MobileViTBlock(dims[2], L[2], channels[9], kernel_size, patch_size, int(dims[2]*4)))

        self.conv2 = conv_1x1_bn(channels[-2], channels[-1])

        self.pool = nn.AvgPool2d(ih//32, 1)
        self.fc = nn.Linear(channels[-1], num_classes, bias=False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.mv2[0](x)

        x = self.mv2[1](x)
        x = self.mv2[2](x)
        x = self.mv2[3](x)      # Repeat

        x = self.mv2[4](x)
        x = self.mvit[0](x)

        x = self.mv2[5](x)
        x = self.mvit[1](x)

        x = self.mv2[6](x)
        x = self.mvit[2](x)
        x = self.conv2(x)

        x = self.pool(x).view(-1, x.shape[1])
        x = self.fc(x)
        return x


def mobilevit_xxs():
    dims = [64, 80, 96]
    channels = [16, 16, 24, 24, 48, 48, 64, 64, 80, 80, 320]
    return MobileViT((256, 256), dims, channels, num_classes=1000, expansion=2)


def mobilevit_xs():
    dims = [96, 120, 144]
    channels = [16, 32, 48, 48, 64, 64, 80, 80, 96, 96, 384]
    return MobileViT((256, 256), dims, channels, num_classes=1000)


def mobilevit_s():
    dims = [144, 192, 240]
    channels = [16, 32, 64, 64, 96, 96, 128, 128, 160, 160, 640]
    return MobileViT((256, 256), dims, channels, num_classes=1000)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    img = torch.randn(5, 3, 256, 256)

    vit = mobilevit_xxs()
    out = vit(img)
    print(out.shape)
    print(count_parameters(vit))

    vit = mobilevit_xs()
    out = vit(img)
    print(out.shape)
    print(count_parameters(vit))

    vit = mobilevit_s()
    out = vit(img)
    print(out.shape)
    print(count_parameters(vit))



#-------------------------------------------------------
#             　EPINET-Dを本格的に実装する
#-------------------------------------------------------

class DWSCBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=2, stride=1, padding=0, skip_con=True):
        super().__init__()
        self.stride = stride
        assert stride in [1, 2]

        self.skip_con = skip_con

        # Define each layer separately
        self.depthwise_conv1 = nn.Conv2d(in_ch, in_ch, kernel_size=kernel_size, stride=stride, padding=padding, groups=in_ch, bias=False)
        self.pointwise_conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=False)
        self.relu1 = nn.ReLU()
        
        self.depthwise_conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding, groups=out_ch, bias=False)
        self.pointwise_conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=False)
        self.batch_norm = nn.BatchNorm2d(out_ch)
        self.relu2 = nn.ReLU()

    def forward(self, x):
        #print("Input size:", x.size())
        
        x = self.depthwise_conv1(x)
        #print("After depthwise_conv1:", x.size())
        
        x = self.pointwise_conv1(x)
        #print("After pointwise_conv1:", x.size())
        
        x = self.relu1(x)
        #print("After relu1:", x.size())
        
        x = self.depthwise_conv2(x)
        #print("After depthwise_conv2:", x.size())
        
        x = self.pointwise_conv2(x)
        #print("After pointwise_conv2:", x.size())
        
        x = self.batch_norm(x)
        #print("After batch_norm:", x.size())
        
        x = self.relu2(x)
        #print("After relu2:", x.size())

        return x