export CUDA_VISIBLE_DEVICES=0
CKPT_DIR="/root/autodl-tmp/dataset/result/mosi/mr0.7_seed1234"
LOG_DIR="/root/CoRA-Diff/eval/mosi/main"
for epoch in {32..53}
do
    CKPT_PATH="${CKPT_DIR}/epoch${epoch}.pth"
    LOG_FILE="${LOG_DIR}/mosi_epoch_${epoch}.log"
    python /root/CoRA-Diff/eval/evaluation_main.py \
        --ckpt "${CKPT_PATH}" \
        --dataset mosi \
        > "${LOG_FILE}" 2>&1
    
    echo "Evaluation for epoch ${epoch} completed. Logs saved to ${LOG_FILE}"
done
