#!/bin/bash

# 使用方法: ./duplicate_airsim.sh <數量>
# 例如: ./duplicate_airsim.sh 2 會生成 2 個訓練設定檔 (第 1 個用 settings.json, 第 2 個用 settings2.json) + 1 個 eval 設定檔 (用 settings2.json)

START_NUM=41451
NUM_COPIES=${1:-1}   # 訓練 instance 數量

SETTINGS_DIR="/home/jonas/Documents/AirSim"
TARGET_DIR="/home/jonas/Documents/AirSim"
LINUX_SRC="/home/jonas/lai/LinuxNoEditor"
LINUX_DEST="/home/jonas/lai"

CREATED_NUMS=()   # 記錄訓練 instance
EVAL_NUM=0        # 記錄 eval instance

# === 1. 複製訓練 instance ===
for ((i=0; i<NUM_COPIES; i++)); do
    NUM=$((START_NUM + i))
    NEW_SETTINGS="$TARGET_DIR/settings${NUM}.json"
    
    # 判斷是否為最後一個訓練 instance
    if [ $((i + 1)) -eq $NUM_COPIES ]; then
        # 最後一個訓練 instance 使用 settings2.json
        SOURCE_SETTINGS="$SETTINGS_DIR/settings2.json"
        echo "複製並覆蓋 $SOURCE_SETTINGS 到 $NEW_SETTINGS (作為最後一個訓練實例)"
    else
        # 其他訓練 instance 使用 settings.json
        SOURCE_SETTINGS="$SETTINGS_DIR/settings.json"
        echo "複製並覆蓋 $SOURCE_SETTINGS 到 $NEW_SETTINGS"
    fi

    # 複製並修改設定檔 (不再檢查檔案是否存在，直接覆蓋)
    cp "$SOURCE_SETTINGS" "$NEW_SETTINGS"
    sed -i "s/\"ApiServerPort\": *[0-9]\+/\"ApiServerPort\": $NUM/" "$NEW_SETTINGS"
    echo "已建立並修改 $NEW_SETTINGS"

    NEW_LINUX="$LINUX_DEST/LinuxNoEditor${NUM}"
    if [ ! -d "$NEW_LINUX" ]; then
        cp -r "$LINUX_SRC" "$NEW_LINUX"
        echo "已建立 $NEW_LINUX"
    fi

    CREATED_NUMS+=("$NUM")
done

# # === 2. 複製 eval instance ===
# EVAL_NUM=$((START_NUM + NUM_COPIES))
# EVAL_SETTINGS="$TARGET_DIR/settings${EVAL_NUM}.json"
# SOURCE_SETTINGS_EVAL="$SETTINGS_DIR/settings2.json"

# # 複製並修改 eval 設定檔 (不再檢查檔案是否存在，直接覆蓋)
# cp "$SOURCE_SETTINGS_EVAL" "$EVAL_SETTINGS"
# sed -i "s/\"ApiServerPort\": *[0-9]\+/\"ApiServerPort\": $EVAL_NUM/" "$EVAL_SETTINGS"
# echo "已建立並修改 eval 設定檔 $EVAL_SETTINGS"

# EVAL_LINUX="$LINUX_DEST/LinuxNoEditor${EVAL_NUM}"
# if [ ! -d "$EVAL_LINUX" ]; then
#     cp -r "$LINUX_SRC" "$EVAL_LINUX"
#     echo "已建立 eval LinuxNoEditor $EVAL_LINUX"
# fi

# === 3. 啟動訓練 instance ===
echo "===== 啟動訓練 instance ====="
NUM_COUNT=${#CREATED_NUMS[@]}

for idx in "${!CREATED_NUMS[@]}"; do
    NUM=${CREATED_NUMS[$idx]}
    APP_DIR="$LINUX_DEST/LinuxNoEditor${NUM}"
    APP_PATH="$APP_DIR/Blocks.sh"
    NEW_SETTINGS="$TARGET_DIR/settings${NUM}.json"

    if [ -f "$APP_PATH" ]; then
        echo "啟動訓練 $APP_PATH 使用 $NEW_SETTINGS ..."

        # 判斷是不是最後一個 instance
        if [ $idx -lt $((NUM_COUNT - 1)) ]; then
            # 其他 instance 使用 -nullrhi
            gnome-terminal -- bash -c "cd '$APP_DIR' && ./Blocks.sh -nullrhi -windowed -resx=640-resy=360 -settings='${NEW_SETTINGS}'; exec bash"
        else
            # 最後一個正常 window
            gnome-terminal -- bash -c "cd '$APP_DIR' && ./Blocks.sh -windowed -resx=640 -resy=360 -settings='${NEW_SETTINGS}'; exec bash"
        fi

        sleep 2
    fi
done


# # === 4. 啟動 eval instance ===
# echo "===== 啟動 eval instance ====="
# APP_DIR="$LINUX_DEST/LinuxNoEditor${EVAL_NUM}"
# APP_PATH="$APP_DIR/Blocks.sh"
# EVAL_SETTINGS="$TARGET_DIR/settings${EVAL_NUM}.json" # 重新定義以確保變數可用
# if [ -f "$APP_PATH" ]; then
#     echo "啟動 eval $APP_PATH 使用 $EVAL_SETTINGS ..."
#     gnome-terminal -- bash -c "cd '$APP_DIR' && ./Blocks.sh -windowed -resx=1280 -resy=720 -settings='${EVAL_SETTINGS}'; exec bash"
# fi

# echo "===== 所有訓練與 eval 實例已啟動 ====="