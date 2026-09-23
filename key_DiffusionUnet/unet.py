import math
import torch
from torch import nn
import torch.nn.functional as F
from inspect import isfunction


def exists(x):
    """检查变量是否存在"""
    return x is not None


def default(val, d):
    """如果val存在则返回val，否则返回默认值d"""
    if exists(val):
        return val
    return d() if isfunction(d) else d


# PositionalEncoding
class PositionalEncoding(nn.Module):
    """位置编码：将时间步信息编码为向量表示"""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, noise_level):
        """
        输入: noise_level shape [batch_size] - 噪声级别/时间步
        输出: encoding shape [batch_size, inner_channel] - 位置编码向量
        """
        count = self.dim // 2
        step = torch.arange(count, dtype=noise_level.dtype,
                            device=noise_level.device) / count
        encoding = noise_level.unsqueeze(
            1) * torch.exp(-math.log(1e4) * step.unsqueeze(0))
        encoding = torch.cat(
            [torch.sin(encoding), torch.cos(encoding)], dim=-1)
        return encoding


# FeatureWiseAffine
class FeatureWiseAffine(nn.Module):
    """特征仿射变换：使用时间嵌入对特征进行调制"""

    def __init__(self, in_channels, out_channels, use_affine_level=False):
        super(FeatureWiseAffine, self).__init__()
        self.use_affine_level = use_affine_level
        self.noise_func = nn.Sequential(
            nn.Linear(in_channels, out_channels * (1 + self.use_affine_level))
        )

    def forward(self, x, noise_embed):
        """
        输入:
            x shape [batch, channels, height, width] - 特征图
            noise_embed shape [batch, in_channels] - 时间嵌入向量
        输出: x shape [batch, out_channels, height, width] - 调制后的特征图
        """
        batch = x.shape[0]
        if self.use_affine_level:
            # 生成缩放因子gamma和偏移因子beta
            gamma, beta = self.noise_func(noise_embed).view(
                batch, -1, 1, 1).chunk(2, dim=1)
            x = (1 + gamma) * x + beta  # 仿射变换: (1+γ)*x + β
        else:
            # 简单的特征加法
            x = x + self.noise_func(noise_embed).view(batch, -1, 1, 1)
        return x


# Swish
class Swish(nn.Module):
    """Swish激活函数：x * sigmoid(x)"""

    def forward(self, x):
        return x * torch.sigmoid(x)


# Upsample
class Upsample(nn.Module):
    """上采样模块：最近邻上采样 + 卷积"""

    def __init__(self, dim):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv = nn.Conv2d(dim, dim, 3, padding=1)

    def forward(self, x):
        """
        输入: x shape [batch, dim, height, width]
        输出: x shape [batch, dim, height*2, width*2] - 上采样后的特征图
        """
        return self.conv(self.up(x))


# Downsample
class Downsample(nn.Module):
    """下采样模块：步长为2的卷积实现下采样"""

    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, 3, 2, 1)

    def forward(self, x):
        """
        输入: x shape [batch, dim, height, width]
        输出: x shape [batch, dim, height//2, width//2] - 下采样后的特征图
        """
        return self.conv(x)


# Block
class Block(nn.Module):
    """基础卷积块：GroupNorm + Swish + Dropout + Conv2d"""

    def __init__(self, dim, dim_out, groups=32, dropout=0, stride=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.GroupNorm(groups, dim),
            Swish(),
            nn.Dropout(dropout) if dropout != 0 else nn.Identity(),
            nn.Conv2d(dim, dim_out, 3, stride=stride, padding=1)
        )

    def forward(self, x):
        return self.block(x)


# 注意力模块
class SEBlock(nn.Module):
    """轻量级通道注意力 - Squeeze-and-Excitation"""

    def __init__(self, channels, reduction=16):
        super().__init__()
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)  # 全局平均池化
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),  # 降维
            Swish(),
            nn.Linear(channels // reduction, channels),  # 升维
            nn.Sigmoid()  # 激活到0-1范围
        )

    def forward(self, x):
        """
        输入: x shape [batch, channels, height, width]
        输出: x shape [batch, channels, height, width] - 通道重加权后的特征
        流程: 全局平均池化 → 全连接层 → Sigmoid → 通道权重相乘
        """
        batch, channels, _, _ = x.shape
        y = self.global_avgpool(x).view(batch, channels)  # 压缩空间维度
        y = self.fc(y).view(batch, channels, 1, 1)  # 生成通道权重
        return x * y  # 通道重加权


class SpatialAttention(nn.Module):
    """轻量级空间注意力"""

    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels // 4, 1),  # 通道降维
            Swish(),
            nn.Conv2d(channels // 4, 1, 1),  # 空间注意力图
            nn.Sigmoid()  # 激活到0-1范围
        )

    def forward(self, x):
        """
        输入: x shape [batch, channels, height, width]
        输出: x shape [batch, channels, height, width] - 空间重加权后的特征
        流程: 卷积降维 → 生成空间注意力图 → 空间权重相乘
        """
        attn = self.conv(x)  # 生成空间注意力图 [B, 1, H, W]
        return x * attn  # 空间重加权


# CSPAM模块 - 基于CSPNet思想
class CSPAM(nn.Module):
    """跨阶段部分注意力模块 - 核心创新模块"""

    def __init__(self, query_dim, key_dims, value_dims, split_ratio=0.5, reduction=16):
        super().__init__()
        # CSP分割比例：将特征分成处理路径和捷径路径
        self.split_ratio = split_ratio
        self.split_dim = int(query_dim * split_ratio)

        # 多尺度融合层：为每个SAR特征尺度创建投影层
        self.cross_scale_fusion = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(key_dim, self.split_dim, 1),  # 1x1卷积投影
                nn.GroupNorm(8, self.split_dim),
                Swish()
            ) for key_dim in key_dims  # 对应5个尺度的SAR特征
        ])

        # 双重注意力机制
        self.channel_attn = SEBlock(self.split_dim, reduction)  # 通道注意力
        self.spatial_attn = SpatialAttention(self.split_dim)  # 空间注意力

        # 最终融合卷积
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(query_dim, query_dim, 3, padding=1),
            nn.GroupNorm(16, query_dim),
            Swish()
        )

    def forward(self, query, keys, values):
        """
        CSPAM前向传播流程：
        1. CSP分割 → 2. 多尺度融合 → 3. 双重注意力 → 4. CSP合并 → 5. 最终融合

        输入:
            query: U-Net当前层特征 [B, C_q, H, W]
            keys: 多尺度SAR条件特征列表 [5个尺度，每个是[B, C_k, H_k, W_k]]
            values: 多尺度SAR条件特征列表
        输出: modulated_query shape [B, C_q, H, W] - 调制后的特征
        """
        batch_size, _, h, w = query.shape

        # 1. CSP分割：将特征分成处理路径和捷径路径
        part1, part2 = torch.split(query, [self.split_dim, query.shape[1] - self.split_dim], dim=1)
        # part1: 处理路径，参与多尺度融合和注意力
        # part2: 捷径路径，直接传递到输出

        # 2. 多尺度特征融合 (只对part1操作)
        fused_part1 = 0
        for i, (key, fusion_layer) in enumerate(zip(keys, self.cross_scale_fusion)):
            # 调整条件特征尺寸以匹配查询特征
            if key.shape[-2:] != (h, w):
                key = F.adaptive_avg_pool2d(key, (h, w))

            # 投影并融合：将SAR特征投影到查询空间并累加
            projected_key = fusion_layer(key)
            fused_part1 = fused_part1 + projected_key

        # 平均融合：对多尺度特征求平均
        fused_part1 = fused_part1 / len(keys)

        # 3. 与原始part1融合 + 双重注意力
        enhanced_part1 = part1 + fused_part1  # 残差连接
        attended_part1 = self.channel_attn(enhanced_part1)  # 通道重校准
        attended_part1 = self.spatial_attn(attended_part1)  # 空间重校准

        # 4. CSP合并：将处理后的part1和捷径part2合并
        merged = torch.cat([attended_part1, part2], dim=1)

        # 5. 最终融合
        output = self.fusion_conv(merged)

        return output + query  # 残差连接：保持梯度流


class EfficientCSPAM(nn.Module):
    """高效版CSPAM - 选择性多尺度融合，节省内存"""

    def __init__(self, query_dim, key_dims, value_dims, split_ratio=0.5):
        super().__init__()
        self.split_ratio = split_ratio
        self.split_dim = int(query_dim * split_ratio)

        # 只选择3个关键尺度进行融合 (节省内存)
        # 跳过最浅层c1(噪声较多)和最深层c5(过于抽象)
        self.selected_scales = [1, 2, 3]  # 使用c2, c3, c4

        # 只为选中的尺度创建融合层
        self.fusion_layers = nn.ModuleList([
            nn.Conv2d(key_dims[i], self.split_dim, 1)  # 1x1投影
            for i in self.selected_scales
        ])

        # 简化注意力机制：只使用通道注意力
        self.channel_attn = SEBlock(self.split_dim, reduction=8)

        # 最终融合
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(query_dim, query_dim, 3, padding=1),
            nn.GroupNorm(16, query_dim),
            Swish()
        )

    def forward(self, query, keys, values):
        """
        高效版CSPAM前向传播：选择性融合关键尺度
        """
        batch_size, _, h, w = query.shape

        # 1. CSP分割
        part1, part2 = torch.split(query, [self.split_dim, query.shape[1] - self.split_dim], dim=1)

        # 2. 选择性多尺度融合：只处理选中的尺度
        fused_part1 = 0
        for i, scale_idx in enumerate(self.selected_scales):
            key = keys[scale_idx]
            if key.shape[-2:] != (h, w):
                key = F.adaptive_avg_pool2d(key, (h, w))

            projected_key = self.fusion_layers[i](key)
            fused_part1 = fused_part1 + projected_key

        # 平均融合
        fused_part1 = fused_part1 / len(self.selected_scales)

        # 3. 注意力机制：只使用通道注意力
        enhanced_part1 = part1 + fused_part1
        attended_part1 = self.channel_attn(enhanced_part1)

        # 4. CSP合并
        merged = torch.cat([attended_part1, part2], dim=1)
        output = self.fusion_conv(merged)

        return output + query  # 残差连接


# ResnetBlock with CSPAM support
class ResnetBlock(nn.Module):
    """残差块：集成CSPAM模块的条件特征融合"""

    def __init__(self, dim, dim_out, noise_level_emb_dim=None, dropout=0,
                 use_affine_level=False, norm_groups=32, cspam_module=None, condition_ch=3):
        super().__init__()
        self.cspam_module = cspam_module  # CSPAM模块

        # 条件特征投影层 - 新增：将条件特征投影到正确的通道数
        self.condition_proj = nn.Conv2d(condition_ch, dim_out, 1) if condition_ch != dim_out else nn.Identity()

        # 时间嵌入的特征调制
        self.noise_func = FeatureWiseAffine(
            noise_level_emb_dim, dim_out, use_affine_level)

        # 条件特征注入 - 修改：现在输入通道数匹配了
        self.c_func = nn.Conv2d(dim_out, dim_out, 1)

        # 两个卷积块
        self.block1 = Block(dim, dim_out, groups=norm_groups)
        self.block2 = Block(dim_out, dim_out, groups=norm_groups, dropout=dropout)

        # 残差连接
        self.res_conv = nn.Conv2d(
            dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb, c, multi_scale_conditions=None):
        """
        残差块前向传播：
        1. 第一个卷积块 → 2. 时间调制 → 3. CSPAM融合 → 4. 第二个卷积块 → 5. 条件注入 → 6. 残差连接

        输入:
            x: 输入特征 [B, C, H, W]
            time_emb: 时间嵌入 [B, C_t]
            c: 条件特征 [B, C_c, H, W] - 现在会被投影到正确通道数
            multi_scale_conditions: 多尺度条件特征列表
        输出: h shape [B, C_out, H, W]
        """
        h = self.block1(x)  # 第一个卷积块
        h = self.noise_func(h, time_emb)  # 时间特征调制

        # 如果提供了CSPAM模块，进行跨尺度特征融合
        if self.cspam_module is not None and multi_scale_conditions is not None:
            h = self.cspam_module(h, multi_scale_conditions, multi_scale_conditions)

        h = self.block2(h)  # 第二个卷积块

        # 投影条件特征到正确通道数，然后注入
        c_projected = self.condition_proj(c)

        # 调整条件特征的空间尺寸以匹配当前特征图
        if c_projected.shape[-2:] != h.shape[-2:]:
            c_projected = F.interpolate(c_projected, size=h.shape[-2:], mode='bilinear', align_corners=False)

        h = self.c_func(c_projected) + h  # 条件特征注入

        return h + self.res_conv(x)  # 残差连接


# SelfAttention
class SelfAttention(nn.Module):
    """自注意力机制：捕捉特征内部的长程依赖关系"""

    def __init__(self, in_channel, n_head=1, norm_groups=32):
        super().__init__()
        self.n_head = n_head

        self.norm = nn.GroupNorm(norm_groups, in_channel)
        self.qkv = nn.Conv2d(in_channel, in_channel * 3, 1, bias=False)  # QKV投影
        self.out = nn.Conv2d(in_channel, in_channel, 1)  # 输出投影

    def forward(self, input, t=None, save_flag=None, file_num=None):
        """
        自注意力前向传播：
        1. 归一化 → 2. QKV投影 → 3. 注意力计算 → 4. 加权求和 → 5. 输出投影 → 6. 残差连接
        """
        batch, channel, height, width = input.shape
        n_head = self.n_head
        head_dim = channel // n_head

        norm = self.norm(input)  # 归一化
        qkv = self.qkv(norm).view(batch, n_head, head_dim * 3, height, width)
        query, key, value = qkv.chunk(3, dim=2)  # 分割QKV

        # 注意力计算：QK^T / sqrt(d)
        attn = torch.einsum(
            "bnchw, bncyx -> bnhwyx", query, key
        ).contiguous() / math.sqrt(channel)
        attn = attn.view(batch, n_head, height, width, -1)
        attn = torch.softmax(attn, -1)  # Softmax归一化
        attn = attn.view(batch, n_head, height, width, height, width)

        # 加权求和：注意力权重 × Value
        out = torch.einsum("bnhwyx, bncyx -> bnchw", attn, value).contiguous()
        out = self.out(out.view(batch, channel, height, width))  # 输出投影

        return out + input  # 残差连接


# ResnetBlocWithAttn with CSPAM support
class ResnetBlocWithAttn(nn.Module):
    """带注意力的残差块：集成自注意力和CSPAM"""

    def __init__(self, dim, dim_out, *, noise_level_emb_dim=None, norm_groups=32,
                 dropout=0, with_attn=False, size=256, cspam_module=None, condition_ch=3):
        super().__init__()
        self.with_attn = with_attn

        # 残差块（包含CSPAM）- 传递 condition_ch
        self.res_block = ResnetBlock(
            dim, dim_out, noise_level_emb_dim, norm_groups=norm_groups,
            dropout=dropout, cspam_module=cspam_module, condition_ch=condition_ch)

        # 可选的自注意力
        if with_attn:
            self.attn = SelfAttention(dim_out, norm_groups=norm_groups)

    def forward(self, x, time_emb, c, multi_scale_conditions=None, t=0, save_flag=False, file_i=0):
        """
        前向传播：残差块 → (可选)自注意力
        """
        x = self.res_block(x, time_emb, c, multi_scale_conditions)  # 残差块
        if self.with_attn:
            x = self.attn(x, t=t, save_flag=save_flag, file_num=file_i)  # 自注意力
        return x


# ResBlock_normal
class ResBlock_normal(nn.Module):
    """标准残差块：用于CPEN编码器"""

    def __init__(self, dim, dim_out, dropout=0, norm_groups=32):
        super().__init__()
        self.block1 = Block(dim, dim_out, groups=norm_groups)
        self.block2 = Block(dim_out, dim_out, groups=norm_groups, dropout=dropout)
        self.res_conv = nn.Conv2d(
            dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x):
        b, c, h, w = x.shape
        h = self.block1(x)
        h = self.block2(h)
        return h + self.res_conv(x)  # 残差连接


# Dual-stream CPEN with spatial and frequency branches
class CPEN(nn.Module):
    """条件编码网络：双分支（空间+频率）从SAR图像提取多尺度特征"""

    def __init__(self, inchannel=1):
        super(CPEN, self).__init__()
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)  # 平均池化下采样

        # 空间分支（原始结构）
        self.E1 = nn.Sequential(
            nn.Conv2d(inchannel, 64, kernel_size=3, padding=1),  # 64通道
            Swish()
        )

        self.E2 = nn.Sequential(
            ResBlock_normal(64, 128, dropout=0, norm_groups=16),  # 128通道
            ResBlock_normal(128, 128, dropout=0, norm_groups=16),
        )

        self.E3 = nn.Sequential(
            ResBlock_normal(128, 256, dropout=0, norm_groups=16),  # 256通道
            ResBlock_normal(256, 256, dropout=0, norm_groups=16),
        )

        self.E4 = nn.Sequential(
            ResBlock_normal(256, 512, dropout=0, norm_groups=16),  # 512通道
            ResBlock_normal(512, 512, dropout=0, norm_groups=16),
        )

        self.E5 = nn.Sequential(
            ResBlock_normal(512, 512, dropout=0, norm_groups=16),
            ResBlock_normal(512, 1024, dropout=0, norm_groups=16),  # 1024通道
        )

        # 频率分支：处理FFT幅度谱
        self.F1 = nn.Sequential(
            nn.Conv2d(inchannel, 64, kernel_size=3, padding=1),
            Swish()
        )
        self.F2 = nn.Sequential(
            ResBlock_normal(64, 128, dropout=0, norm_groups=16),
            ResBlock_normal(128, 128, dropout=0, norm_groups=16)
        )
        self.F3 = nn.Sequential(
            ResBlock_normal(128, 256, dropout=0, norm_groups=16),
            ResBlock_normal(256, 256, dropout=0, norm_groups=16)
        )
        self.F4 = nn.Sequential(
            ResBlock_normal(256, 512, dropout=0, norm_groups=16),
            ResBlock_normal(512, 512, dropout=0, norm_groups=16)
        )
        self.F5 = nn.Sequential(
            ResBlock_normal(512, 512, dropout=0, norm_groups=16),
            ResBlock_normal(512, 1024, dropout=0, norm_groups=16)
        )

    def _fft_logmag(self, x):
        """
        计算FFT对数幅度谱
        输入: x [B, C, H, W]
        输出: 对数幅度谱 [B, C, H, W]
        """
        # 转换为灰度图进行FFT
        if x.shape[1] > 1:
            x_gray = x.mean(dim=1, keepdim=True)
        else:
            x_gray = x

        # 计算FFT
        fft = torch.fft.fft2(x_gray, norm='ortho')  # 复数张量 [B, 1, H, W]
        mag = torch.abs(fft)

        # 对数幅度（稳定性处理）
        log_mag = torch.log1p(mag)

        # 复制通道以匹配原始输入通道数
        if x.shape[1] > 1:
            log_mag = log_mag.repeat(1, x.shape[1], 1, 1)

        return log_mag

    def forward(self, x):
        """
        CPEN前向传播：双分支提取多尺度特征
        输出:
            spatial_features: 5个尺度的空间特征 [c1, c2, c3, c4, c5]
            freq_features: 5个尺度的频率特征 [f1, f2, f3, f4, f5]
        """
        # 空间分支
        x1 = self.E1(x)  # [B, 64, H, W]
        x2 = self.pool(x1)
        x2 = self.E2(x2)  # [B, 128, H/2, W/2]

        x3 = self.pool(x2)
        x3 = self.E3(x3)  # [B, 256, H/4, W/4]

        x4 = self.pool(x3)
        x4 = self.E4(x4)  # [B, 512, H/8, W/8]

        x5 = self.pool(x4)
        x5 = self.E5(x5)  # [B, 1024, H/16, W/16]

        # 频率分支（处理对数幅度谱）
        mag = self._fft_logmag(x)
        f1 = self.F1(mag)  # [B, 64, H, W]
        f2 = self.pool(f1)
        f2 = self.F2(f2)  # [B, 128, H/2, W/2]

        f3 = self.pool(f2)
        f3 = self.F3(f3)  # [B, 256, H/4, W/4]

        f4 = self.pool(f3)
        f4 = self.F4(f4)  # [B, 512, H/8, W/8]

        f5 = self.pool(f4)
        f5 = self.F5(f5)  # [B, 1024, H/16, W/16]

        # 数值稳定性检查
        if torch.isnan(x1).any() or torch.isnan(x2).any() or torch.isnan(x3).any() or torch.isnan(
                x4).any() or torch.isnan(x5).any():
            print('nan detected in CPEN spatial branch')
        if torch.isnan(f1).any() or torch.isnan(f2).any() or torch.isnan(f3).any() or torch.isnan(
                f4).any() or torch.isnan(f5).any():
            print('nan detected in CPEN frequency branch')

        return (x1, x2, x3, x4, x5), (f1, f2, f3, f4, f5)


# Main UNet class with dual-stream CPEN and CSPAM
class UNet(nn.Module):
    """CSPAM U-Net：集成双分支CPEN和CSPAM的扩散模型去噪网络"""

    def __init__(
            self,
            in_channel=6,  # 输入通道数 (条件+噪声图像)
            out_channel=3,  # 输出通道数
            inner_channel=32,  # 基础通道数
            norm_groups=32,  # GroupNorm分组数
            channel_mults=(1, 2, 4, 6, 8),  # 通道倍增系数
            attn_res=(16,),  # 使用自注意力的分辨率
            res_blocks=2,  # 每个分辨率的残差块数量
            dropout=0,  # Dropout率
            with_noise_level_emb=True,  # 是否使用时间嵌入
            image_size=128,  # 输入图像尺寸
            lowres_cond=True,  # 低分辨率条件（未使用）
            condition_ch=3,  # 条件图像通道数
            use_cspam=True,  # 是否使用CSPAM
            cspam_blocks=(3, 4, 5),  # 使用CSPAM的块索引
            cspam_type='efficient'  # CSPAM类型：'efficient'或'standard'
    ):
        super().__init__()

        # CSPAM配置
        self.use_cspam = use_cspam
        self.cspam_blocks = cspam_blocks
        self.cspam_type = cspam_type

        # CPEN输出的多尺度通道数 [c1, c2, c3, c4, c5]
        self.cpen_channels = [64, 128, 256, 512, 1024]

        # 时间嵌入网络
        if with_noise_level_emb:
            noise_level_channel = inner_channel
            self.noise_level_mlp = nn.Sequential(
                PositionalEncoding(inner_channel),  # 位置编码
                nn.Linear(inner_channel, inner_channel * 4),  # 升维
                Swish(),
                nn.Linear(inner_channel * 4, inner_channel)  # 降维
            )
        else:
            noise_level_channel = None
            self.noise_level_mlp = None

        # 网络参数
        self.res_blocks = res_blocks
        num_mults = len(channel_mults)
        self.num_mults = num_mults
        pre_channel = inner_channel
        feat_channels = [pre_channel]  # 存储跳跃连接特征通道数
        now_res = image_size  # 当前分辨率

        # ==================== 下采样路径 ====================
        downs = [nn.Conv2d(in_channel, inner_channel, kernel_size=3, padding=1)]  # 输入卷积

        # 为每个块配置CSPAM
        block_idx = 0  # 块索引计数器
        for ind in range(num_mults):
            is_last = (ind == num_mults - 1)  # 是否最后一个倍数
            use_attn = (now_res in attn_res)  # 是否在当前分辨率使用自注意力
            channel_mult = inner_channel * channel_mults[ind]  # 当前阶段通道数

            # 每个分辨率的残差块
            for _ in range(0, res_blocks):
                # 决定是否在当前块使用CSPAM
                use_block_cspam = use_cspam and (block_idx in cspam_blocks)
                cspam_module = None

                if use_block_cspam:
                    if cspam_type == 'efficient':
                        cspam_module = EfficientCSPAM(
                            query_dim=channel_mult,
                            key_dims=self.cpen_channels,
                            value_dims=self.cpen_channels,
                            split_ratio=0.5
                        )
                    else:
                        cspam_module = CSPAM(
                            query_dim=channel_mult,
                            key_dims=self.cpen_channels,
                            value_dims=self.cpen_channels,
                            split_ratio=0.5
                        )

                # 创建带CSPAM的残差块 - 添加 condition_ch 参数
                downs.append(ResnetBlocWithAttn(
                    pre_channel, channel_mult,
                    noise_level_emb_dim=noise_level_channel,
                    norm_groups=norm_groups,
                    dropout=dropout,
                    with_attn=use_attn,
                    size=now_res,
                    cspam_module=cspam_module,
                    condition_ch=condition_ch  # 新增
                ))
                feat_channels.append(channel_mult)
                pre_channel = channel_mult
                block_idx += 1

            # 下采样（除了最后一个阶段）
            if not is_last:
                downs.append(Downsample(pre_channel))
                feat_channels.append(pre_channel)
                now_res = now_res // 2  # 分辨率减半

        self.downs = nn.ModuleList(downs)

        # ==================== 中间层 ====================
        self.mid = nn.ModuleList([
            ResnetBlocWithAttn(pre_channel, pre_channel,
                               noise_level_emb_dim=noise_level_channel,
                               norm_groups=norm_groups,
                               dropout=dropout, with_attn=True, size=now_res,
                               condition_ch=condition_ch),  # 添加 condition_ch
            ResnetBlocWithAttn(pre_channel, pre_channel,
                               noise_level_emb_dim=noise_level_channel,
                               norm_groups=norm_groups,
                               dropout=dropout, with_attn=False, size=now_res,
                               condition_ch=condition_ch)  # 添加 condition_ch
        ])

        # ==================== 上采样路径 ====================
        ups = []
        for ind in reversed(range(num_mults)):
            is_last = (ind < 1)
            use_attn = (now_res in attn_res)
            channel_mult = inner_channel * channel_mults[ind]

            for _ in range(0, res_blocks + 1):  # 上采样路径多一个残差块
                # 上采样路径也使用CSPAM
                use_block_cspam = use_cspam and (block_idx in cspam_blocks)
                cspam_module = None

                if use_block_cspam:
                    if cspam_type == 'efficient':
                        cspam_module = EfficientCSPAM(
                            query_dim=channel_mult,
                            key_dims=self.cpen_channels,
                            value_dims=self.cpen_channels,
                            split_ratio=0.5
                        )
                    else:
                        cspam_module = CSPAM(
                            query_dim=channel_mult,
                            key_dims=self.cpen_channels,
                            value_dims=self.cpen_channels,
                            split_ratio=0.5
                        )

                # 上采样残差块（包含跳跃连接）- 添加 condition_ch 参数
                ups.append(ResnetBlocWithAttn(
                    pre_channel + feat_channels.pop(),  # 跳跃连接通道拼接
                    channel_mult,
                    noise_level_emb_dim=noise_level_channel,
                    norm_groups=norm_groups,
                    dropout=dropout, with_attn=use_attn, size=now_res,
                    cspam_module=cspam_module,
                    condition_ch=condition_ch  # 新增
                ))
                pre_channel = channel_mult
                block_idx += 1

            # 上采样（除了最后一个阶段）
            if not is_last:
                ups.append(Upsample(pre_channel))
                now_res = now_res * 2  # 分辨率加倍

        self.ups = nn.ModuleList(ups)

        # 最终输出卷积
        self.final_conv = Block(pre_channel, default(out_channel, in_channel), groups=norm_groups)

        # 双分支条件编码网络
        self.condition = CPEN(inchannel=condition_ch)
        self.condition_ch = condition_ch

        # 条件特征投影层：融合空间和频率特征
        cond_channel_list = [64, 128, 256, 512, 1024]
        self.cond_proj = nn.ModuleList([
            nn.Conv2d(ch * 2, ch, kernel_size=1) for ch in cond_channel_list
        ])

    def forward(self, x, time, img_s1=None, class_label=None, return_condition=False, t_ori=0):
        """
        U-Net前向传播完整流程：
        1. 分离条件和噪声图像 → 2. 提取双分支多尺度条件特征 → 3. 融合空间和频率特征
        4. 时间嵌入 → 5. 下采样路径 + CSPAM融合 → 6. 中间层 → 7. 上采样路径 + CSPAM融合 → 8. 最终输出

        输入:
            x: 拼接的输入 [B, condition_ch + 3, H, W] (条件图像 + 噪声图像)
            time: 时间步 [B]
        输出: 预测的噪声 [B, 3, H, W] 或 (预测噪声, 条件特征)
        """
        # 1. 分离条件和噪声图像
        condition = x[:, :self.condition_ch, ...].clone()  # 条件图像 [B, 3, H, W]
        x = x[:, self.condition_ch:, ...]  # 噪声图像 [B, 3, H, W]

        # 2. 获取双分支多尺度条件特征
        (c1, c2, c3, c4, c5), (f1, f2, f3, f4, f5) = self.condition(condition)

        # 3. 融合空间和频率特征
        multi_scale_conditions = []
        for i, (spatial_feat, freq_feat) in enumerate(zip([c1, c2, c3, c4, c5], [f1, f2, f3, f4, f5])):
            # 拼接空间和频率特征
            fused_feat = torch.cat([spatial_feat, freq_feat], dim=1)
            # 投影到原始通道数
            fused_feat = self.cond_proj[i](fused_feat)
            multi_scale_conditions.append(fused_feat)

        # 4. 时间嵌入
        t = self.noise_level_mlp(time) if exists(self.noise_level_mlp) else None

        # 5. 下采样路径
        feats = []
        for layer in self.downs:
            if isinstance(layer, ResnetBlocWithAttn):
                # 残差块：传入多尺度条件特征用于CSPAM
                x = layer(x, t, condition, multi_scale_conditions)
            else:
                x = layer(x)
            feats.append(x)

        # 6. 中间层
        for layer in self.mid:
            if isinstance(layer, ResnetBlocWithAttn):
                x = layer(x, t, condition, multi_scale_conditions)
            else:
                x = layer(x)

        # 7. 上采样路径
        for layer in self.ups:
            if isinstance(layer, ResnetBlocWithAttn):
                # 跳跃连接：拼接下采样特征
                x = layer(torch.cat((x, feats.pop()), dim=1), t, condition, multi_scale_conditions)
            else:
                x = layer(x)

        # 8. 最终输出
        out = self.final_conv(x)

        if return_condition:
            return out, multi_scale_conditions
        else:
            return out