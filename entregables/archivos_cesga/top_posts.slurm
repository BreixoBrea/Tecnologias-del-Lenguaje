#!/bin/bash
#SBATCH --job-name=top_posts
#SBATCH --output=/mnt/netapp2/Store_uni/home/usc/cursos/curso1278/teclin/logs/top_posts_%j.out
#SBATCH --error=/mnt/netapp2/Store_uni/home/usc/cursos/curso1278/teclin/logs/top_posts_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --time=6:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=breixo.brea@rai.usc.es

# --- Limpiar y cargar módulos ---
module --force purge
module --ignore-cache load cesga/system miniconda3/22.11.1-1

# --- Activar el entorno de embeddings ---
source $(conda info --base)/etc/profile.d/conda.sh
conda activate /mnt/netapp2/Store_uni/home/usc/cursos/curso1278/teclin/teclin_env

# --- Información de entorno ---
echo "=== INICIO DEL JOB ${SLURM_JOB_ID} ==="
hostname
nvidia-smi
echo "======================================"

# --- Ejecutar script de reducción CSV / top posts ---
mkdir -p logs  # asegura que el directorio logs está creado
python reduccion_csv_por_rasgo.py > logs/top_posts_${SLURM_JOB_ID}.log 2>&1

# --- Estado final de la GPU ---
echo "=== GPU STATUS POST-PROCESSING ==="
nvidia-smi

