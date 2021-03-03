import torch
import torch.nn as nn
import torchsummary

from layers import TransformerEncoder

class ViTSmall(nn.Module):
    def __init__(self, in_c:int=3, num_classes:int=10, img_size:int=32, patch:int=8, dropout:float=0., head:int=8):
        super(ViTSmall, self).__init__()
        hidden=384

        self.patch = patch # number of patches in one row(or col)
        self.patch_size = img_size//self.patch
        f = (img_size//self.patch)**2*3 # 48 # patch vec length

        self.emb = nn.Linear(f, hidden) # (b, n, f)
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden))
        self.pos_emb = nn.Parameter(torch.randn(1, (self.patch**2)+1, hidden))
        self.enc = nn.Sequential(
            TransformerEncoder(hidden, dropout=dropout, head=head),
            TransformerEncoder(hidden, dropout=dropout, head=head),
            TransformerEncoder(hidden, dropout=dropout, head=head),
            TransformerEncoder(hidden, dropout=dropout, head=head),
            TransformerEncoder(hidden, dropout=dropout, head=head),
            TransformerEncoder(hidden, dropout=dropout, head=head),
            TransformerEncoder(hidden, dropout=dropout, head=head)
        )
        self.fc = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, num_classes) # for cls_token
        )


    def forward(self, x):
        out = self._to_words(x)
        out = torch.cat([self.cls_token.repeat(out.size(0),1,1), self.emb(out)],dim=1)
        out = out + self.pos_emb
        out = self.enc(out)
        out = out[:,0]
        out = self.fc(out)
        return out

    def _to_words(self, x):
        """
        (b, c, h, w) -> (b, n, f)
        """
        out = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size).permute(0,2,3,4,5,1).contiguous()
        out = out.view(x.size(0), self.patch**2 ,-1)
        return out


if __name__ == "__main__":
    b,c,h,w = 4, 3, 32, 32
    x = torch.randn(b, c, h, w)
    net = ViTSmall(c, 10, h, 16, dropout=0.1)
    out = net(x)
    # out.mean().backward()
    torchsummary.summary(net, (c,h,w))
    print(out.shape)
    