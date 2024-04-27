#!/bin/bash

# 使用sudo新建一个控制台并执行第一个Python脚本
sudo gnome-terminal -- bash -c "python main.py --topo $1 2>&1 | tee main.log" &

sleep 20
# 使用sudo新建另一个控制台并执行第二个Python脚本
sudo gnome-terminal -- bash -c "python route.py 2>&1 | tee route.log" &

sleep 1


# sudo gnome-terminal -- bash -c "sudo wireshark" &
