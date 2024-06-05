# ComfyUI-OmDet
本节点主要是基于[OmDet](https://github.com/om-ai-lab/OmDet)，实现目标检测功能功能,并且输出相关的图片、mask和labelme格式的JSON信息。

![image](/docs/workflow.png)

# 切换语言
- [English](README.md)
- [简体中文](readme/README.zh_CN.md)

# 节点
|名称                          |描述                              |
|------------------------------|---------------------------------|
|Apply OmDet             |默认，使用OmDet模型，自动下载模型   |

# 需要安装的依赖
```
pip install -r requirements.txt
```

# 模型
本节点会自动根据你选择的语言下载对应模型
```
ComfyUI
    models
        OmDet
            OmDet-Turbo_tiny_SWIN_T.pth
            ViT-B-16.pt
```
