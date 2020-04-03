# Image Inpainting via Generative Multi-column Convolutional Neural Networks

## Prerequisites
- Python3.5 (or higher)
- Tensorflow 1.4 (or later versions, excluding 2.x) with NVIDIA GPU or CPU
- OpenCV
- numpy
- scipy
- easydict
- Pytorch 1.0 with NVIDIA GPU or CPU
- tensorboardX

## Testing(Model not Trained Yet)
`checkpoints/`. To test images in a folder, you can specify the folder address by the opinion `--dataset_path`, and set the pretrained model path by `--load_model_dir` when calling `test.py`.

For example:

python test.py --dataset paris_streetview --data_file ./imgs/paris-streetview_256x256/ --load_model_dir ./checkpoints/paris-streetview_256x256_rect --random_mask 0
or

sh ./script/test.sh

### Training
For a given dataset, the training is formed of two stages. We pretrain the whole network with only confidence-driven reconstruction loss first, and finetune this network using adversarial and ID-MRF loss along with the reconstruction loss after the previous phase converges.

To pretrain the network,

python train.py --dataset [DATASET_NAME] --data_file [DATASET_TRAININGFILE] --gpu_ids [NUM] --pretrain_network 1 --batch_size 16

where `[DATASET_TRAININGFILE]` indicates a file storing the full paths of the training images.

Then finetune the network,

python train.py --dataset [DATASET_NAME] --data_file [DATASET_TRAININGFILE] --gpu_ids [NUM] --pretrain_network 0 --load_model_dir [PRETRAINED_MODEL_PATH] --batch_size 8

We provide both random <b>stroke</b> and <b>rectangle</b> masks in the training and testing phase. The used mask type is indicated by specifying `--mask_type [rect(default)|stroke]` option when calling `train.py` or `test.py`.


