import lpips
import torch


# LPIPS github link: https://github.com/richzhang/PerceptualSimilarity

loss_fn_alex = lpips.LPIPS(net='alex')  # best forward scores
loss_fn_vgg = lpips.LPIPS(net='vgg')  # closer to "traditional" perceptual loss, when used for optimization

img0 = torch.zeros(1, 3, 64, 64)  # image should be RGB, IMPORTANT: normalized to [-1,1]
img1 = torch.ones(1, 3, 64, 64)
d_alex = loss_fn_alex(img0, img1)
d_vgg = loss_fn_vgg(img0, img1)

print("LPIPS score: d_alex: {}, d_vgg: {}".format(d_alex, d_vgg))
