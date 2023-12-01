from tinyvit import TinyViT
import torch

checkpoint = torch.load('nn/tiny_vit_5m_22kto1k_distill.pth')

model = TinyViT(
    img_size=224,
    num_classes=1000,
    embed_dims=[64, 128, 160, 320],
    num_heads=[ 2, 4, 5, 10 ],
    depths=[ 2, 2, 6, 2 ]
)

model.load_state_dict(checkpoint['model'])
model.eval()

input = torch.randn(1, 3, 224, 224)
output = model.custom_forward_features(input)
print(output.shape)

