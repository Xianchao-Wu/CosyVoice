#########################################################################
# File Name: wav2mp4.sh
# Author: Xianchao Wu
# mail: xianchaow@nvidia.com
# Created Time: Mon Sep  1 23:42:07 2025
#########################################################################
#!/bin/bash


#ffmpeg -f lavfi -i color=c=black:s=1280x720:d=0.1 -i input.wav -c:v libx264 -c:a aac -shortest output.mp4

#!/bin/bash

# 输出文件夹
outdir="mp4_out"
mkdir -p "$outdir"

# 清理旧的文件列表
listfile="file_list.txt"
> "$listfile"

# 遍历所有 .wav 文件
for wav in *.wav; do
    # 去掉扩展名
    base="${wav%.wav}"
    png="$base.png"
    mp4="$outdir/$base.mp4"

    if [[ -f "$png" ]]; then
        echo "🎬 正在处理: $wav + $png -> $mp4"
        ffmpeg -y -loop 1 -i "$png" -i "$wav" -c:v libx264 -c:a aac -b:a 192k -shortest "$mp4"
        echo "file '$mp4'" >> "$listfile"
    else
        echo "⚠️ 没找到 $png，跳过 $wav"
    fi
done

# 合并所有 mp4
final="final_merged.mp4"
if [[ -s "$listfile" ]]; then
    echo "📀 正在合并所有 mp4 -> $final"
    ffmpeg -f concat -safe 0 -i "$listfile" -c copy "$final"
else
    echo "❌ 没有找到任何可以合并的 mp4"
fi

echo "✅ 全部完成！输出在 $outdir 和 $final"


