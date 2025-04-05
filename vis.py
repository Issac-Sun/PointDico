# 修改字体为 Times New Roman，调整坐标轴字号
import matplotlib.pyplot as plt
import numpy as np

# 数据
masking_ratio = np.arange(0, 1.0, 0.1)  # 横坐标
cls_mIoU = [80.4, 80.5, 81.8, 82.3, 83.7, 83.8, 84.5, 83.6, 83.4, 83.0]  # 第一个子图左轴
inst_mIoU = [83.7, 83.8, 84.5, 85.0, 85.4, 86.1, 86.5, 85.5, 85.7, 85.2]  # 第一个子图右轴
OA = [90.94, 91.0, 92.5, 93.0, 93.5, 93.8, 94.32, 94.0, 93.5, 93.0]  # 第二个子图左轴
OA_right = [83.07, 85.4, 86.5, 87.0, 88.29, 88.51, 89.10, 88.5, 88.21, 87.5]  # 第二个子图右轴

# 创建画布，调整子图尺寸比例为正方形
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# 增加全局字体大小和线宽设置
font_size = 18
line_width = 4
offset = 0.2  # 数值上方的偏移量


# 第一个子图
ax1 = axes[0]
ax1.plot(masking_ratio, cls_mIoU, color='blue', marker='o', label='Cls.mIoU (%)', linewidth=line_width, markersize=8)
ax1.set_xlabel('Masking Ratio', fontsize=font_size)
ax1.set_ylabel('Cls.mIoU (%)', fontsize=font_size)
ax1.set_ylim(80, 87)
ax1.tick_params(axis='y', labelsize=font_size)
ax1.tick_params(axis='x', labelsize=font_size)

ax1_right = ax1.twinx()
ax1_right.plot(masking_ratio, inst_mIoU, color='red', marker='s', label='Inst.mIoU (%)', linewidth=line_width, markersize=8)
ax1_right.set_ylabel('Inst.mIoU (%)', fontsize=font_size)
ax1_right.set_ylim(80, 87)
ax1_right.tick_params(axis='y', labelsize=font_size)

# 仅显示横轴为0.6的数据点的数值
index = np.where(np.isclose(masking_ratio, 0.6))[0][0]
ax1.text(masking_ratio[index], cls_mIoU[index] + offset, f'{cls_mIoU[index]:.1f}', fontsize=font_size, color='blue', ha='center')
ax1_right.text(masking_ratio[index], inst_mIoU[index] + offset, f'{inst_mIoU[index]:.1f}', fontsize=font_size, color='red', ha='center')

# 添加图例
lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax1_right.get_legend_handles_labels()
ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left', fontsize=font_size)

# 设置边框
for spine in ax1.spines.values():
    spine.set_linewidth(1.5)
for spine in ax1_right.spines.values():
    spine.set_linewidth(1.5)

ax1.grid(axis='y', linestyle='--', alpha=0.7)
ax1.set_title('ShapeNetPart Cls.mIoU(%) and Inst.mIoU(%)', fontsize=font_size)

# 第二个子图
ax2 = axes[1]
ax2.plot(masking_ratio, OA, color='blue', marker='o', label='OA BG (%)', linewidth=line_width, markersize=8)
ax2.set_xlabel('Masking Ratio', fontsize=font_size)
ax2.set_ylabel('OA_BG (%)', fontsize=font_size)
ax2.set_ylim(90, 95)
ax2.tick_params(axis='y', labelsize=font_size)
ax2.tick_params(axis='x', labelsize=font_size)

ax2_right = ax2.twinx()
ax2_right.plot(masking_ratio, OA_right, color='red', marker='s', label='OA Hardest (%)', linewidth=line_width, markersize=8)
ax2_right.set_ylabel('OA_Hardest (%)', fontsize=font_size)
ax2_right.set_ylim(83, 92)
ax2_right.tick_params(axis='y', labelsize=font_size)

# 仅显示横轴为0.6的数据点的数值
ax2.text(masking_ratio[index], OA[index] + offset, f'{OA[index]:.2f}', fontsize=font_size, color='blue', ha='center')
ax2_right.text(masking_ratio[index], OA_right[index] + offset, f'{OA_right[index]:.1f}', fontsize=font_size, color='red', ha='center')

# 添加图例
lines_1, labels_1 = ax2.get_legend_handles_labels()
lines_2, labels_2 = ax2_right.get_legend_handles_labels()
ax2.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left', fontsize=font_size)

# 设置边框
for spine in ax2.spines.values():
    spine.set_linewidth(1.5)
for spine in ax2_right.spines.values():
    spine.set_linewidth(1.5)

ax2.grid(axis='y', linestyle='--', alpha=0.7)
ax2.set_title('ScanObjectNN OBJ_BG(%) and Hardest(%)', fontsize=font_size)

# 调整布局并显示图像
plt.tight_layout()
plt.show()
