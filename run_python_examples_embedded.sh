#!/usr/bin/env bash

BASE_DIR=`pwd`"/"`dirname $0`
EXAMPLES=`echo $1 | sed -e 's/ //g'`

set -euo pipefail

USE_CUDA=$(python -c "import torchvision, torch; print(torch.cuda.is_available())")
case $USE_CUDA in
  "True")
    echo "using cuda"
    CUDA=1
    CUDA_FLAG="--cuda"
    ;;
  "False")
    echo "not using cuda"
    CUDA=0
    CUDA_FLAG=""
    ;;
  "")
    exit 1;
    ;;
esac

ERRORS=""

function error() {
  ERR=$1
  ERRORS="$ERRORS\n$ERR"
  echo $ERR
}

function start() {
  EXAMPLE=${FUNCNAME[1]}
  cd $BASE_DIR/$EXAMPLE
  echo "Running example: $EXAMPLE"
}

function fast_neural_style() {
  start

  echo "running fast neural style model"
  python neural_style/neural_style.py eval --content-image images/content-images/amber.jpg --model saved_models/candy.pth --output-image images/output-images/amber-candy.jpg --cuda $CUDA || error "neural_style.py failed"
}

function imagenet() {
  start
  python main.py --epochs 1 sample/ || error "imagenet example failed"
}

function mnist_hogwild() {
  start
  python main.py --epochs 1 --num-processes 6 $CUDA_FLAG || error "mnist hogwild failed"
}

function regression() {
  start
  python main.py --epochs 1 $CUDA_FLAG || error "regression failed"
}

function super_resolution() {
  start
  python main.py --upscale_factor 3 --batchSize 1 --nEpochs 1 --lr 0.001  || error "super resolution failed"
}

function time_sequence_prediction() {
  start
  python generate_sine_wave.py || { error "generate sine wave failed";  return; }
  python train.py --steps 2 || error "time sequence prediction training failed"
}

function vae() {
  start
  python main.py --epochs 1 || error "vae failed"
}

function word_language_model() {
  start
  python main.py --epochs 1 $CUDA_FLAG || error "word_language_model failed"
}

function run_all() {
  fast_neural_style
  imagenet
  mnist_hogwild
  regression
  super_resolution
  time_sequence_prediction
  vae
  word_language_model
}

function clean() {
  cd $BASE_DIR
  echo "running clean to remove cruft"

  rm -rf imagenet/checkpoint.pth.tar \
    imagenet/model_best.pth.tar \
    super_resolution/model_epoch_1.pth \
    time_sequence_prediction/predict*.pdf \
    time_sequence_prediction/traindata.pt \
    data/MNIST/raw/t10k-images-idx3-ubyte \
    data/MNIST/raw/t10k-labels-idx1-ubyte \
    data/MNIST/raw/train-images-idx3-ubyte \
    data/MNIST/raw/train-labels-idx1-ubyte \
    data/MNIST/processed/

  git checkout fast_neural_style/images/output-images/amber-candy.jpg || error "couldn't clean up fast neural style image"
}

# by default, run all examples
if [ "" == "$EXAMPLES" ]; then
  run_all
else
  for i in $(echo $EXAMPLES | sed "s/,/ /g")
  do
    echo "Starting $i"
    $i
    echo "Finished $i, status $?"
  done
fi

clean

if [ "" == "$ERRORS" ]; then
  echo "Completed successfully with status $?"
else
  echo "Some examples failed:"
  printf "$ERRORS"
  exit 1
fi
