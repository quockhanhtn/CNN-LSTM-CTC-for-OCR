# CNN-LSTM-CTC Model for Optical Character Recognition (OCR)

### Dataset

- Download [here](https://www.robots.ox.ac.uk/~vgg/data/text/)
<br/><br/>


### Train model

- Change global variables before train at file train.py, from line 32 to 36

  ```python
  PATH_TO_DATASET = "C:/Users/quock/Downloads/Compressed/mnt/ramdisk/max/90kDICT32px"
  NO_OF_IMAGES = 150000
  MODEL_CK_POINT_FPATH = "best_model.hdf5"
  BATCH_SIZE = 256
  EPOCHS = 10
  ```

- Start train model by using command

  ```bash
  python train.py
  ```
<br/><br/>


### Predict model

  ```bash
  python predict.py -dir {path_to_image_directory}
  ```
<br/><br/>


### Reference:  
- [A-CRNN-model-for-Text-Recognition-in-Keras, TheAILearner](https://github.com/TheAILearner/A-CRNN-model-for-Text-Recognition-in-Keras)