#!/bin/bash
#SBATCH --job-name=bigfive_llm
#SBATCH --output=/mnt/netapp2/Store_uni/home/usc/cursos/curso1070/logs/bigfive_llm_%j.log
#SBATCH --error=/mnt/netapp2/Store_uni/home/usc/cursos/curso1070/logs/bigfive_llm_%j.err
#SBATCH --time=1:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=roque.de@rai.usc.es


# 1. Configurar entorno

module purge
module load cesga/system
module load miniconda3/22.11.1-1

STORE=/mnt/netapp2/Store_uni/home/usc/cursos/curso1070

# Activar entorno virtual
source $STORE/myenv/bin/activate

# Redirigir Hugging Face cache dentro de Store_uni
export HF_HOME=$STORE/huggingface_cache
mkdir -p $HF_HOME

# Instalar dependencias si aún no existen
pip install --upgrade --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install --upgrade --no-cache-dir transformers huggingface_hub scikit-learn swifter tqdm pandas


# 2. Ejecutar script Python

python3 $STORE/tec/bigfive_pipeline.py

