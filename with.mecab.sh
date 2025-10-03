#########################################################################
# File Name: with.mecab.sh
# Author: Xianchao Wu
# mail: xianchaow@nvidia.com
# Created Time: Thu Oct  2 07:55:11 2025
#########################################################################
#!/bin/bash

echo "$1" \
  | mecab -Ochasen \
  | awk '{print $2}' \
  | grep -v EOS \
  | tr '\n' ' ' \
  | sed 's/ $/\n/' \
  | python3 -c "import sys, jaconv; print(jaconv.kata2hira(sys.stdin.read()))"

