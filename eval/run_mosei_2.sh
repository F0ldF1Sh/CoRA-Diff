export CUDA_VISIBLE_DEVICES=1
CKPT_DIR="/root/autodl-tmp/dataset/result/mosei/mr0.7_seed1234"
LOG_DIR="/root/CoRA-Diff/eval/mosei/main"
for epoch in {32..42}
do
    CKPT_PATH="${CKPT_DIR}/epoch${epoch}.pth"
    LOG_FILE="${LOG_DIR}/mosei_epoch_${epoch}.log"
    python /root/CoRA-Diff/eval/evaluation_main.py \
        --ckpt "${CKPT_PATH}" \
        --dataset mosei \
        > "${LOG_FILE}" 2>&1
    
    echo "Evaluation for epoch ${epoch} completed. Logs saved to ${LOG_FILE}"
done
