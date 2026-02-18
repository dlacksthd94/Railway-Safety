#!/bin/bash
cd ~/PROJECT/Railway-Safety

source ~/miniconda3/etc/profile.d/conda.sh
conda activate rw_safety

echo -e "========================================\n========================================" >> cron.log
echo -e "Running cron job at $(date)\n" >> cron.log
echo -e "========================================\n========================================" >> cron.log
python daemon.py >> cron.log 2>&1