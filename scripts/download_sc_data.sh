if [ $# -eq 0 ]; then
  echo "Provide output dir as an argument!"
  exit 1
fi

if [ -d "$1" ]; then
  if [ -f "$1/challenge_pbmc_cellxgene_230223.h5ad" ]; then
    echo "File challenge_pbmc_cellxgene_230223.h5ad already exists, skipping download"
  else
    wget https://covid19.cog.sanger.ac.uk/challenge_pbmc_cellxgene_230223.h5ad -P $1
  fi
  export PBMC_ROOT=$1/challenge_pbmc_cellxgene_230223.h5ad
  echo $PBMC_ROOT
  python3 preprocess_sc_data.py --cfg_file configs/single_cell/pbmc.yaml
else
  echo "Output directory does not exist!"
  exit 1
fi


