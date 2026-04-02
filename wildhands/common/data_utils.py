"""
This file contains functions that are used to perform data augmentation.
"""
import cv2
import numpy as np
import torch
# from loguru import logger

#在其次坐标系下，通过这个3×3的仿射矩阵把将原始图片中的“手部区域（Bounding Box）
# ”裁剪、缩放并选择性旋转，最终映射到神经网络所需的固定输入分辨率

def get_transform(center, scale, res, rot=0):
    """Generate transformation matrix."""
    h = 200 * scale
    t = np.zeros((3, 3))
    t[0, 0] = float(res[1]) / h
    t[1, 1] = float(res[0]) / h
    t[0, 2] = res[1] * (-float(center[0]) / h + 0.5)
    t[1, 2] = res[0] * (-float(center[1]) / h + 0.5)
    t[2, 2] = 1
    if not rot == 0:
        rot = -rot  # To match direction of rotation from cropping
        rot_mat = np.zeros((3, 3))
        rot_rad = rot * np.pi / 180
        sn, cs = np.sin(rot_rad), np.cos(rot_rad)
        rot_mat[0, :2] = [cs, -sn]
        rot_mat[1, :2] = [sn, cs]
        rot_mat[2, 2] = 1
        # Need to rotate around center
        t_mat = np.eye(3)
        t_mat[0, 2] = -res[1] / 2
        t_mat[1, 2] = -res[0] / 2
        t_inv = t_mat.copy()
        t_inv[:2, 2] *= -1
        ## 矩阵连乘：初始变换 -> 移到中心 -> 旋转 -> 移回中心
        t = np.dot(t_inv, np.dot(rot_mat, np.dot(t_mat, t)))
    return t

#这是坐标变换函数，将像素坐标从原图映射到裁剪后的图像坐标系（反之亦然）。
# 输入：
# - pt: 像素坐标 [x, y]
# - center, scale, res, rot: 与 get_transform 相同
# - invert: 是否反向变换（从输出坐标→原图坐标）
def transform(pt, center, scale, res, invert=0, rot=0):
    """Transform pixel location to different reference."""
    t = get_transform(center, scale, res, rot=rot)
    if invert:
        t = np.linalg.inv(t)
    new_pt = np.array([pt[0] - 1, pt[1] - 1, 1.0]).T#-1是因为像素坐标是从1开始
    new_pt = np.dot(t, new_pt)
    return new_pt[:2].astype(int) + 1

#把坐标轴旋转后，原来点的坐标在新坐标系中的表示
def rotate_2d(pt_2d, rot_rad):
    x = pt_2d[0]
    y = pt_2d[1]
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)
    xx = x * cs - y * sn
    yy = x * sn + y * cs
    return np.array([xx, yy], dtype=np.float32)



# 核心思想：建立3对对应点 → 求仿射变换
# 源图像 (src)                    目标图像 (dst)
#     ↑                              ↑ 
#   src[0] = 中心点                 dst[0] = 输出图像中心
#   src[1] = 中心+向下(旋转后)      dst[1] = 中心+向下
#   src[2] = 中心+向右(旋转后)      dst[2] = 中心+向右
# 流程：
# 1. 先缩放：src_w = src_width * scale
# 2. 计算旋转后的向下/向右方向向量
# 3. 用 cv2.getAffineTransform 求变换矩阵
# 用途：将原图中以 (c_x, c_y) 为中心、经过旋转缩放的区域，变换到固定尺寸 dst_width × dst_height 的输出图像中心。
# inv=True：返回逆变换（从输出坐标反推原图坐标），用于将关键点坐标映射回原图。
def gen_trans_from_patch_cv(
    c_x, c_y, src_width, src_height, dst_width, dst_height, scale, rot, inv=False
):
    # augment size with scale
    src_w = src_width * scale
    src_h = src_height * scale
    src_center = np.array([c_x, c_y], dtype=np.float32)

    # augment rotation
    rot_rad = np.pi * rot / 180
    src_downdir = rotate_2d(np.array([0, src_h * 0.5], dtype=np.float32), rot_rad)
    src_rightdir = rotate_2d(np.array([src_w * 0.5, 0], dtype=np.float32), rot_rad)

    dst_w = dst_width
    dst_h = dst_height
    dst_center = np.array([dst_w * 0.5, dst_h * 0.5], dtype=np.float32)
    dst_downdir = np.array([0, dst_h * 0.5], dtype=np.float32)
    dst_rightdir = np.array([dst_w * 0.5, 0], dtype=np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = src_center
    src[1, :] = src_center + src_downdir
    src[2, :] = src_center + src_rightdir

    dst = np.zeros((3, 2), dtype=np.float32)
    dst[0, :] = dst_center
    dst[1, :] = dst_center + dst_downdir
    dst[2, :] = dst_center + dst_rightdir

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    trans = trans.astype(np.float32)
    return trans


# #作用：
# 从输入图像中根据边界框提取并变换得到一个图像块（patch）。
# 具体流程：
# 1. 根据 bbox（边界框）、scale（缩放）、rot（旋转）生成仿射变换矩阵
# 2. 对原图进行高斯模糊（抗锯齿）
# 3. 使用 warpAffine 将图像变换到目标尺寸 out_shape
# 4. 额外生成逆变换矩阵 inv_trans
# 输出：
# - img_patch: 变换后的图像块
# - trans: 正向仿射变换矩阵
# - inv_trans: 逆变换矩阵（可用于将关键点等从 patch 坐标映射回原图）
# 典型用途： 目标检测/姿态估计中的数据增强，或从大图中裁剪局部区域进行后续处理。
def generate_patch_image(
    cvimg,
    bbox,
    scale,
    rot,
    out_shape,
    interpl_strategy,
    gauss_kernel=5,
    gauss_sigma=8.0,
):
    img = cvimg.copy()

    bb_c_x = float(bbox[0])
    bb_c_y = float(bbox[1])
    bb_width = float(bbox[2])
    bb_height = float(bbox[3])

    trans = gen_trans_from_patch_cv(
        bb_c_x, bb_c_y, bb_width, bb_height, out_shape[1], out_shape[0], scale, rot
    )

    # anti-aliasing
    blur = cv2.GaussianBlur(img, (gauss_kernel, gauss_kernel), gauss_sigma)
    img_patch = cv2.warpAffine(
        blur, trans, (int(out_shape[1]), int(out_shape[0])), flags=interpl_strategy
    )
    #深度学习模型输入一般是 float32，需要显式转换
    #高斯模糊和变换过程中可能会改变数据类型，加上这行确保类型统一
    img_patch = img_patch.astype(np.float32)
    inv_trans = gen_trans_from_patch_cv(
        bb_c_x,
        bb_c_y,
        bb_width,
        bb_height,
        out_shape[1],
        out_shape[0],
        scale,
        rot,
        inv=True,
    )

    return img_patch, trans, inv_trans


# 数据增强参数生成函数。
# 输入：
# - is_train: 是否在训练模式
# - flip_prob: 翻转概率
# - noise_factor: 噪声因子
# - rot_factor: 旋转因子
# - scale_factor: 缩放因子
# 输出 augm_dict： 包含以下增强参数
# - flip: 1=翻转，0=不翻转
# - pn: 每个通道的噪声乘子（3个值）
# - rot: 旋转角度
# - sc: 缩放因子
# 训练时：
# - 按概率翻转
# - 像素噪声：各通道乘以 1-noise, 1+noise 间的随机数
# - 旋转：随机角度（正态分布，受 rot_factor 限制）
# - 缩放：随机缩放（正态分布，均值1，受 scale_factor 限制）
def augm_params(is_train, flip_prob, noise_factor, rot_factor, scale_factor):
    """Get augmentation parameters."""
    flip = 0  # flipping
    pn = np.ones(3)  # per channel pixel-noise
    rot = 0  # rotation
    sc = 1  # scaling
    if is_train:
        # We flip with probability 1/2
        if np.random.uniform() <= flip_prob:
            flip = 1
            # assert False, "Flipping not supported"

        # Each channel is multiplied with a number
        # in the area [1-opt.noiseFactor,1+opt.noiseFactor]
        pn = np.random.uniform(1 - noise_factor, 1 + noise_factor, 3)

        # The rotation is a number in the area [-2*rotFactor, 2*rotFactor]
        rot = min(
            2 * rot_factor,
            max(
                -2 * rot_factor,
                np.random.randn() * rot_factor,
            ),
        )

        # The scale is multiplied with a number
        # in the area [1-scaleFactor,1+scaleFactor]
        sc = min(
            1 + scale_factor,
            max(
                1 - scale_factor,
                np.random.randn() * scale_factor + 1,
            ),
        )
        # but it is zero with probability 3/5
        if np.random.uniform() <= 0.6:
            rot = 0

    augm_dict = {}
    augm_dict["flip"] = flip
    augm_dict["pn"] = pn
    augm_dict["rot"] = rot
    augm_dict["sc"] = sc
    return augm_dict


# RGB图像预处理函数。
# 作用：
# 1. 裁剪：根据 bbox_dim 和 sc 计算裁剪尺寸，用 generate_patch_image 从原图提取并旋转/缩放到 img_res × img_res
# 2. 像素噪声：各通道乘以 pn[0/1/2]（来自增强参数）
# 3. 归一化：转为 float32 → 转置 (H,W,C) → (C,H,W) → 除以 255 归一化到 0,1
# 输出： 形状为 (3, img_res, img_res) 的张量，供神经网络输入
def rgb_processing(is_train, rgb_img, center, bbox_dim, augm_dict, img_res):
    rot = augm_dict["rot"]
    sc = augm_dict["sc"]
    pn = augm_dict["pn"]
    scale = sc * bbox_dim

    crop_dim = int(scale * 200)
    # faster cropping!!
    rgb_img = generate_patch_image(
        rgb_img,
        [center[0], center[1], crop_dim, crop_dim],
        1.0,
        rot,
        [img_res, img_res],
        cv2.INTER_CUBIC,
    )[0]

    # in the rgb image we add pixel noise in a channel-wise manner
    rgb_img[:, :, 0] = np.minimum(255.0, np.maximum(0.0, rgb_img[:, :, 0] * pn[0]))
    rgb_img[:, :, 1] = np.minimum(255.0, np.maximum(0.0, rgb_img[:, :, 1] * pn[1]))
    rgb_img[:, :, 2] = np.minimum(255.0, np.maximum(0.0, rgb_img[:, :, 2] * pn[2]))
    rgb_img = np.transpose(rgb_img.astype("float32"), (2, 0, 1)) / 255.0
    return rgb_img

# 掩码（mask）预处理函数，与 rgb_processing 类似但有区别：
# 相同点：
# - 裁剪、旋转、缩放到 img_res × img_res
# - 归一化到 0,1
# 不同点：
# - 使用 generate_patch_image_clean（而非 generate_patch_image，应该是无高斯模糊版本）
# - 使用 cv2.INTER_NEAREST（最近邻插值，保持掩码边界清晰）
# - 不添加像素噪声：注释掉了噪声部分，因为掩码是离散标签，加噪声无意义
# 输出： 形状为 (C, img_res, img_res) 的掩码张量，C 是掩码通道数
# 虽然参数名字叫 rgb_img，但在这个特定的 mask_processing 函数里，
# 它输入的不是那张色彩丰富的原始照片，而是一张已经做好了标注的“分割掩码图”（Mask）。
def mask_processing(is_train, rgb_img, center, bbox_dim, augm_dict, img_res):
    rot = augm_dict["rot"]
    sc = augm_dict["sc"]
    pn = augm_dict["pn"]
    scale = sc * bbox_dim

    crop_dim = int(scale * 200)
    # faster cropping!!
    rgb_img = generate_patch_image_clean(
        rgb_img,
        [center[0], center[1], crop_dim, crop_dim],
        1.0,
        rot,
        [img_res, img_res],
        cv2.INTER_NEAREST,
    )[0]

    # no noise for mask
    # rgb_img[:, :, 0] = np.minimum(255.0, np.maximum(0.0, rgb_img[:, :, 0] * pn[0]))
    # rgb_img[:, :, 1] = np.minimum(255.0, np.maximum(0.0, rgb_img[:, :, 1] * pn[1]))
    # rgb_img[:, :, 2] = np.minimum(255.0, np.maximum(0.0, rgb_img[:, :, 2] * pn[2]))
    rgb_img = np.transpose(rgb_img.astype("float32"), (2, 0, 1)) / 255.0
    return rgb_img

# 深度图预处理函数，与 mask_processing 几乎相同：
# 这里输入的是图片中每个像素的深度信息，也就是离相机的距离
# 区别：
# - 不归一化：没有除以 255，直接返回原始像素值
# - 没有 transpose
# 输出： 处理后的深度图（形状 H×W 或 C×H×W）
def depth_processing(is_train, rgb_img, center, bbox_dim, augm_dict, img_res):
    rot = augm_dict["rot"]
    sc = augm_dict["sc"]
    pn = augm_dict["pn"]
    scale = sc * bbox_dim

    crop_dim = int(scale * 200)
    # faster cropping!!
    rgb_img = generate_patch_image_clean(
        rgb_img,
        [center[0], center[1], crop_dim, crop_dim],
        1.0,
        rot,
        [img_res, img_res],
        cv2.INTER_NEAREST,
    )[0]

    # no noise for mask
    # rgb_img[:, :, 0] = np.minimum(255.0, np.maximum(0.0, rgb_img[:, :, 0] * pn[0]))
    # rgb_img[:, :, 1] = np.minimum(255.0, np.maximum(0.0, rgb_img[:, :, 1] * pn[1]))
    # rgb_img[:, :, 2] = np.minimum(255.0, np.maximum(0.0, rgb_img[:, :, 2] * pn[2]))
    return rgb_img

# 这个函数不是给模型算的，而是为了统一考试格式。如果你要向 ARCTIC 官网提交你的预测结果，
# 你必须把预测值通过这个函数转换到 1000px 的空间里，否则你的误差会大得离谱
# 这函数输入的 kp2d 是原图上标注的 2D 关键点坐标（通常是手部关节点或人脸关键点），
# 经过裁剪区域变换后，输出对应于 generate_patch_image 裁剪出图像块中的坐标。
def transform_kp2d(kp2d, bbox):
    # bbox: (cx, cy, scale) in the original image space
    # scale is normalized
    assert isinstance(kp2d, np.ndarray)
    assert len(kp2d.shape) == 2
    cx, cy, scale = bbox
    s = 200 * scale  # to px
    cap_dim = 1000  # px
    factor = cap_dim / (1.5 * s)
    kp2d_cropped = np.copy(kp2d)
    kp2d_cropped[:, 0] -= cx - 1.5 / 2 * s
    kp2d_cropped[:, 1] -= cy - 1.5 / 2 * s
    kp2d_cropped[:, 0] *= factor
    kp2d_cropped[:, 1] *= factor
    return kp2d_cropped

# 核心作用：当你在训练时对 RGB 图像进行了随机旋转（rot）和缩放（sc）增强后，这个函数确保 2D 关键点
# 的标签（Ground Truth）也经历完全相同的旋转和缩放。
def j2d_processing(kp, center, bbox_dim, augm_dict, img_res):
    """Process gt 2D keypoints and apply all augmentation transforms."""
    scale = augm_dict["sc"] * bbox_dim
    rot = augm_dict["rot"]

    nparts = kp.shape[0]
    for i in range(nparts):
        kp[i, 0:2] = transform(
            kp[i, 0:2] + 1,
            center,
            scale,
            [img_res, img_res],
            rot=rot,
        )
    # convert to normalized coordinates
    kp = normalize_kp2d_np(kp, img_res)
    kp = kp.astype("float32")
    return kp


def pose_processing(pose, augm_dict):
    """Process SMPL theta parameters  and apply all augmentation transforms."""
    rot = augm_dict["rot"]
    # rotation or the pose parameters
    pose[:3] = rot_aa(pose[:3], rot)
    # flip the pose parameters
    # (72),float
    pose = pose.astype("float32")
    return pose

#这个函数的作用是：把一个“轴角表示”的旋转 aa，再额外绕 z 轴旋转 rot 度，返回更新后的轴角结果。
def rot_aa(aa, rot):
    """Rotate axis angle parameters."""
    # pose parameters
    R = np.array(
        [
            [np.cos(np.deg2rad(-rot)), -np.sin(np.deg2rad(-rot)), 0],
            [np.sin(np.deg2rad(-rot)), np.cos(np.deg2rad(-rot)), 0],
            [0, 0, 1],
        ]
    )
    # find the rotation of the body 
    per_rdg, _ = cv2.Rodrigues(aa)
    # apply the global rotation to the global orientation
    resrot, _ = cv2.Rodrigues(np.dot(R, per_rdg))
    aa = (resrot.T)[0]
    return aa

#反归一化
def denormalize_images(images):
    images = images * torch.tensor([0.229, 0.224, 0.225], device=images.device).reshape(
        1, 3, 1, 1
    )
    images = images + torch.tensor([0.485, 0.456, 0.406], device=images.device).reshape(
        1, 3, 1, 1
    )
    return images

# 这个函数的核心作用是：
# 避免程序因为某一张图片坏掉直接崩掉
# 用一个占位图继续让后面的流程跑下去
# 同时用 True/False 告诉调用方这张图到底是不是正常读到的
def read_img(img_fn, dummy_shape):
    try:
        cv_img = _read_img(img_fn)
    except:
        # logger.warning(f"Unable to load {img_fn}")
        cv_img = np.zeros(dummy_shape, dtype=np.float32)
        return cv_img, False
    return cv_img, True


def _read_img(img_fn):
    img = cv2.cvtColor(cv2.imread(img_fn), cv2.COLOR_BGR2RGB)
    return img.astype(np.float32)


#坐标范围大致从 [0, img_res] 映射到 [-1, 1]，或者反过来映射回来
# 它们的区别可以直接记成两组：
# numpy 版：
# normalize_kp2d_np
# unnormalize_2d_kp
# torch 版：
# normalize_kp2d
# unormalize_kp2d
# 再加一个维度区别：
# 前三个函数都保留了第 3 维信息或要求输入带第 3 维
# 最后一个 unormalize_kp2d 只接受 ... x 2，只管坐标，不管置信度
def normalize_kp2d_np(kp2d: np.ndarray, img_res):
    assert kp2d.shape[1] == 3
    kp2d_normalized = kp2d.copy()
    kp2d_normalized[:, :2] = 2.0 * kp2d[:, :2] / img_res - 1.0
    return kp2d_normalized


def unnormalize_2d_kp(kp_2d_np: np.ndarray, res):
    assert kp_2d_np.shape[1] == 3
    kp_2d = np.copy(kp_2d_np)
    kp_2d[:, :2] = 0.5 * res * (kp_2d[:, :2] + 1)
    return kp_2d


def normalize_kp2d(kp2d: torch.Tensor, img_res):
    assert len(kp2d.shape) == 3
    kp2d_normalized = kp2d.clone()
    kp2d_normalized[:, :, :2] = 2.0 * kp2d[:, :, :2] / img_res - 1.0
    return kp2d_normalized


def unormalize_kp2d(kp2d_normalized: torch.Tensor, img_res):
    assert len(kp2d_normalized.shape) == 3
    assert kp2d_normalized.shape[2] == 2
    kp2d = kp2d_normalized.clone()
    kp2d = 0.5 * img_res * (kp2d + 1)
    return kp2d

# 弱透视分支：直接在最终 img_res × img_res 输入坐标系里造一个默认相机
# 真实内参分支：从原图内参出发，经过裁剪和平移、再经过 resize，得到最终 img_res × img_res 输入坐标系下的内参
#弱透视参数
def get_wp_intrix(fixed_focal: float, img_res):
    # consruct weak perspective on patch
    camera_center = np.array([img_res // 2, img_res // 2])
    intrx = torch.zeros([3, 3])
    intrx[0, 0] = fixed_focal
    intrx[1, 1] = fixed_focal
    intrx[2, 2] = 1.0
    intrx[0, -1] = camera_center[0]
    intrx[1, -1] = camera_center[1]
    return intrx

# 这个函数的作用是：在做裁剪、缩放数据增强之后，计算“这张 patch 对应的新相机内参矩阵 intrx”。
# 它主要解决一个问题：
# 原始相机内参通常是针对整张原图的
# 但训练时我们常常会从原图里裁一个 bbox，再 resize 成固定大小 img_res
# 这样一来，原来的内参就不再直接适用了，需要换算成 patch 坐标系下的新内参
#分为两种情况，第一种不使用真实内参，使用一个编造的弱透视参数
#第二种，使用内参，并根据裁剪对内参进行调整，如果我们只是把一张已有 patch 从 400 像素重新采样到 224 像素，那么以像素为单位的焦距一定跟着变小
def get_aug_intrix(
    intrx, fixed_focal: float, img_res, use_gt_k, bbox_cx, bbox_cy, scale
):
    """
    This function returns camera intrinsics under scaling.
    If use_gt_k, the GT K is used, but scaled based on the amount of scaling in the patch.
    Else, we construct an intrinsic camera with a fixed focal length and fixed camera center.
    """

    if not use_gt_k:
        # consruct weak perspective on patch
        intrx = get_wp_intrix(fixed_focal, img_res)
    else:
        # update the GT intrinsics (full image space)
        # such that it matches the scale of the patch

        dim = scale * 200.0  # bbox size
        k_scale = float(img_res) / dim  # resized_dim / bbox_size in full image space
        """
        # x1 and y1: top-left corner of bbox
        intrinsics after data augmentation
        fx' = k*fx
        fy' = k*fy
        cx' = k*(cx - x1)
        cy' = k*(cy - y1)
        """
        intrx[0, 0] *= k_scale  # k*fx
        intrx[1, 1] *= k_scale  # k*fy
        intrx[0, 2] -= bbox_cx - dim / 2.0 #减去patch的左上角坐标原点，也就是计算相机主点在patch坐标系中的坐标
        intrx[1, 2] -= bbox_cy - dim / 2.0
        intrx[0, 2] *= k_scale
        intrx[1, 2] *= k_scale
    return intrx

# 输入：
# - cvimg: 原始图片
# - bbox: 边界框 [cx, cy, w, h]（中心点坐标 + 宽高）
# - scale: 缩放比例（让裁剪区域比bbox稍大）
# - rot: 旋转角度
# - out_shape: 输出图像尺寸
# - interpl_strategy: 插值策略
# 作用：
# 1. 根据 bbox 和 scale 计算仿射变换矩阵 trans
# 2. 用 cv2.warpAffine 裁剪并对齐图片到指定尺寸（仿射变换）
# 3. 计算逆变换矩阵 inv_trans（用于后续把结果坐标映射回原图）# 典型用途： 把带旋转、尺度变化的手部区域裁剪成统一大小，便于后续网络处理。
def generate_patch_image_clean(
    cvimg,
    bbox,
    scale,
    rot,
    out_shape,
    interpl_strategy,
    gauss_kernel=5,
    gauss_sigma=8.0,
):
    img = cvimg.copy()

    bb_c_x = float(bbox[0])
    bb_c_y = float(bbox[1])
    bb_width = float(bbox[2])
    bb_height = float(bbox[3])

    trans = gen_trans_from_patch_cv(
        bb_c_x, bb_c_y, bb_width, bb_height, out_shape[1], out_shape[0], scale, rot
    )

    img_patch = cv2.warpAffine(
        img, trans, (int(out_shape[1]), int(out_shape[0])), flags=interpl_strategy
    )
    img_patch = img_patch.astype(np.float32)
    inv_trans = gen_trans_from_patch_cv(
        bb_c_x,
        bb_c_y,
        bb_width,
        bb_height,
        out_shape[1],
        out_shape[0],
        scale,
        rot,
        inv=True,
    )

    return img_patch, trans, inv_trans

#也是数据增强，加一个裁剪框移动的干扰，只做了一个裁剪图平移的干扰，范围是[-0.2w,0.2w]
def jitter_bbox(bbox, s_stdev=0.5, t_stdev=0.2):
    if bbox is None: return bbox
    x0, y0, w, h = bbox
    center = np.array([x0 + w / 2, y0 + h / 2]) 
    ori_size = np.array([w, h])

    jitter_s = np.exp(np.random.rand() * s_stdev * 2 - s_stdev)
    new_size = ori_size #* jitter_s

    jitter_t = np.random.rand(2) * t_stdev * 2 - t_stdev
    jitter_t = ori_size * jitter_t
    new_center = center + jitter_t

    new_x0 = new_center[0] - new_size[0] / 2
    new_y0 = new_center[1] - new_size[1] / 2

    new_bbox = np.array([new_x0, new_y0, new_size[0], new_size[1]]).astype(np.float32)
    return new_bbox 

#增加一列1，齐次坐标
def pad_jts2d(jts):
    num_jts = jts.shape[0]
    jts_pad = np.ones((num_jts, 3))
    jts_pad[:, :2] = jts
    return jts_pad