from flask import Flask, flash, redirect, render_template, \
     request, url_for
#from pynput.keyboard import Key, Controller
import tensorflow as tf
import os
import json
import random
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
import shutil
import numpy as np
from tensorflow.keras.utils import plot_model
from yolov3.dataset import Dataset
from yolov3.yolov4 import Create_Yolo, compute_loss
from yolov3.utils import load_yolo_weights
from yolov3.configs import *
from evaluate_mAP import get_mAP
from toxml import *

app = Flask(__name__)

@app.route('/')
def index():
    ex = ['.jpg', '.png']
    train_len = len([f for f in os.listdir('./prepare_data/with_annotation/train') if f.endswith(tuple(ex))])
    test_len = len([f for f in os.listdir('./prepare_data/with_annotation/test') if f.endswith(tuple(ex))])
    txtArea_len = len([f for f in os.listdir('./txtArea') if f.endswith('.txt')])
    
    total_len = train_len + test_len + txtArea_len - 10
 
    return render_template('index.html'
                        ,Train_Semi=[{'name':'False'}, {'name':'True'}]
                        ,Train_from_checkpoint=[{'name':'False'}, {'name':'True'}]
                        ,total_len = total_len
                        ,init_test_num = test_len if test_len > 0 else int(str(total_len*0.2)[0])*10**(len(str(total_len*0.2).split('.')[0])-1)
                        )

@app.route('/Reset_all_file/')
def Reset_all_file():
    ex = ['.jpg', '.png']
    
    train_img_path_list = ['./prepare_data/with_annotation/train/' + f for f in os.listdir('./prepare_data/with_annotation/train') if f.endswith(tuple(ex))]
    test_img_path_list = ['./prepare_data/with_annotation/test/' + f for f in os.listdir('./prepare_data/with_annotation/test') if f.endswith(tuple(ex))]
    txt_img_path_list = [f for f in os.listdir('./txtTemp')]

    for i in train_img_path_list:
            file_name = i.split('/')[-1]
            print(file_name)
            shutil.move(i, './prepare_data/without_annotation/train/' + file_name)
            print('./prepare_data/with_annotation/' + 'train' + file_name.split('.')[0] + '.xml')
            shutil.move('.' + i.split('.')[-2] + '.xml', './prepare_data/without_annotation/train/'  + file_name.split('.')[0] + '.xml')
        
    for i in test_img_path_list:
            file_name = i.split('/')[-1]
            print(file_name)
            shutil.move(i, './prepare_data/without_annotation/train/' + file_name)
            print('./prepare_data/with_annotation/' + 'test' + file_name.split('.')[0] + '.xml')
            shutil.move('.' + i.split('.')[-2] + '.xml', './prepare_data/without_annotation/train/'  + file_name.split('.')[0] + '.xml')

    for i in txt_img_path_list:
            shutil.move('txtTemp/'+i,'txtArea/'+i)
    return 'Succeed.'


@app.route('/Train_Predict/')
def Train_Predict():
  toxml() # modify label, move to with annotation from without (need to delete original xml in without annotation)
  
  with open(os.path.join(os.getcwd(),'yolov3','custom.json')) as file:
    data = json.load(file)
  file.close()
  
  ex = ['.jpg', '.png']
  #handel test set
  train_img_path_list = ['./prepare_data/with_annotation/train/' + f for f in os.listdir('./prepare_data/with_annotation/train') if f.endswith(tuple(ex))]
  test_img_path_list = ['./prepare_data/with_annotation/test/' + f for f in os.listdir('./prepare_data/with_annotation/test') if f.endswith(tuple(ex))]
  num_move = data['Num_Test'] - len(test_img_path_list)
  if num_move > 0 and len(train_img_path_list) - num_move < 10:
    return 'Failed (Because of test num parameter)'
  
  move_list = train_img_path_list
  dest = 'test/'
  if num_move >= 0:
    choose_index = random.sample(range(len(train_img_path_list)), num_move)
  else:
    choose_index = random.sample(range(len(test_img_path_list)), -num_move)
    move_list = test_img_path_list
    dest = 'train/'
    
  for i in choose_index:
    file_name = move_list[i].split('/')[-1]
    print(file_name)
    shutil.move(move_list[i], './prepare_data/with_annotation/' + dest + file_name)
    print('./prepare_data/with_annotation/' + dest + file_name.split('.')[0] + '.xml')
    shutil.move('.' + move_list[i].split('.')[-2] + '.xml', './prepare_data/with_annotation/' + dest + file_name.split('.')[0] + '.xml')
   
  os.system("python ./tools/XML_to_YOLOv3.py") # update with annotation xml to yolo
  if data['Train_Semi'] == True: 
    os.system("python ./add_aug.py") # increase dataset by augmentation for low AP object
    os.system("python ./tools/Predict_XML_to_YOLOv3.py")# update without annotation xml to yolo
  
  global TRAIN_EPOCHS, TRAIN_FROM_CHECKPOINT, TRAIN_CLASSES, TRAIN_ANNOT_PATH
  TRAIN_EPOCHS = data['Epoch']
  TRAIN_FROM_CHECKPOINT       = "checkpoints/yolov4_custom" if data["Train_from_checkpoint"] == True else False
  TRAIN_CLASSES               = "model_data/class_names.txt" if data['Train_Semi'] == False else "model_data/semi_model_data/class_names.txt"
  TRAIN_ANNOT_PATH            = "model_data/class_train.txt" if data['Train_Semi'] == False else "model_data/semi_model_data/class_train.txt"
  print('Epochs:', TRAIN_EPOCHS)
  train()
  os.system("python detection_image_from_directory.py 1") # produce pseudo label in without annotation
  return 'Succeed.'


@app.route("/set_parameter/", methods=['GET', 'POST'])
def set_parameter():
    semi_select = request.form.get('semi_select')
    checkpoint_select = request.form.get('checkpoint_select')
    epoch = int(request.form.get('epoch'))
    num_test = int(request.form.get('num_test'))
    dictionary = {
        "Train_Semi":semi_select == "True",
        "Train_from_checkpoint":checkpoint_select == "True",
        "Epoch":epoch,
        "Num_Test":num_test
    }
    json_object = json.dumps(dictionary, indent=4)
    
    with open(os.path.join(os.getcwd(),"yolov3","custom.json"), "w") as outfile:
        outfile.write(json_object)
    return "Parameters Set"

Darknet_weights = YOLO_V4_WEIGHTS # yolo v4 weights path
    
def train():
    global TRAIN_FROM_CHECKPOINT
    
    gpus = tf.config.experimental.list_physical_devices('GPU')
    # print(f'GPUs {gpus}')
    if len(gpus) > 0:
        try: tf.config.experimental.set_memory_growth(gpus[0], True)
        except RuntimeError: pass

    if os.path.exists(TRAIN_LOGDIR): shutil.rmtree(TRAIN_LOGDIR) # remove the files in log directory
    writer = tf.summary.create_file_writer(TRAIN_LOGDIR) # write the log

    trainset = Dataset('train')
    testset = Dataset('test')

    steps_per_epoch = len(trainset)
    global_steps = tf.Variable(1, trainable=False, dtype=tf.int64)
    warmup_steps = TRAIN_WARMUP_EPOCHS * steps_per_epoch
    total_steps = TRAIN_EPOCHS * steps_per_epoch

    if TRAIN_TRANSFER:
        Darknet = Create_Yolo(input_size=YOLO_INPUT_SIZE, CLASSES=YOLO_COCO_CLASSES)
        load_yolo_weights(Darknet, Darknet_weights) # use darknet weights

    yolo = Create_Yolo(input_size=YOLO_INPUT_SIZE, training=True, CLASSES=TRAIN_CLASSES)
    if TRAIN_FROM_CHECKPOINT:
        try:
            yolo.load_weights(f"./checkpoints/{TRAIN_MODEL_NAME}")
        except ValueError:
            print("Shapes are incompatible, transfering Darknet weights")
            TRAIN_FROM_CHECKPOINT = False

    if TRAIN_TRANSFER and not TRAIN_FROM_CHECKPOINT:
        for i, l in enumerate(Darknet.layers):
            layer_weights = l.get_weights()
            if layer_weights != []:
                try:
                    yolo.layers[i].set_weights(layer_weights)
                except:
                    print("skipping", yolo.layers[i].name)
    
    optimizer = tf.keras.optimizers.Adam()


    def train_step(image_data, target):
        with tf.GradientTape() as tape:
            pred_result = yolo(image_data, training=True)
            giou_loss=conf_loss=prob_loss=0

            # optimizing process
            grid = 3 if not TRAIN_YOLO_TINY else 2
            for i in range(grid):
                conv, pred = pred_result[i*2], pred_result[i*2+1]
                loss_items = compute_loss(pred, conv, *target[i], i, CLASSES=TRAIN_CLASSES)
                giou_loss += loss_items[0]
                conf_loss += loss_items[1]
                prob_loss += loss_items[2]

            total_loss = giou_loss + conf_loss + prob_loss

            gradients = tape.gradient(total_loss, yolo.trainable_variables)
            optimizer.apply_gradients(zip(gradients, yolo.trainable_variables))

            # update learning rate
            # about warmup: https://arxiv.org/pdf/1812.01187.pdf&usg=ALkJrhglKOPDjNt6SHGbphTHyMcT0cuMJg
            global_steps.assign_add(1)
            if global_steps < warmup_steps:# and not TRAIN_TRANSFER:
                lr = global_steps / warmup_steps * TRAIN_LR_INIT
            else:
                lr = TRAIN_LR_END + 0.5 * (TRAIN_LR_INIT - TRAIN_LR_END)*(
                    (1 + tf.cos((global_steps - warmup_steps) / (total_steps - warmup_steps) * np.pi)))
            optimizer.lr.assign(lr.numpy())

            # writing summary data
            with writer.as_default():
                tf.summary.scalar("lr", optimizer.lr, step=global_steps)
                tf.summary.scalar("loss/total_loss", total_loss, step=global_steps)
                tf.summary.scalar("loss/giou_loss", giou_loss, step=global_steps)
                tf.summary.scalar("loss/conf_loss", conf_loss, step=global_steps)
                tf.summary.scalar("loss/prob_loss", prob_loss, step=global_steps)
            writer.flush()
            
        return global_steps.numpy(), optimizer.lr.numpy(), giou_loss.numpy(), conf_loss.numpy(), prob_loss.numpy(), total_loss.numpy()

    validate_writer = tf.summary.create_file_writer(TRAIN_LOGDIR)
    def validate_step(image_data, target):
        with tf.GradientTape() as tape:
            pred_result = yolo(image_data, training=False)
            giou_loss=conf_loss=prob_loss=0

            # optimizing process
            grid = 3 if not TRAIN_YOLO_TINY else 2
            for i in range(grid):
                conv, pred = pred_result[i*2], pred_result[i*2+1]
                loss_items = compute_loss(pred, conv, *target[i], i, CLASSES=TRAIN_CLASSES)
                giou_loss += loss_items[0]
                conf_loss += loss_items[1]
                prob_loss += loss_items[2]

            total_loss = giou_loss + conf_loss + prob_loss
            
        return giou_loss.numpy(), conf_loss.numpy(), prob_loss.numpy(), total_loss.numpy()

    mAP_model = Create_Yolo(input_size=YOLO_INPUT_SIZE, CLASSES=TRAIN_CLASSES) # create second model to measure mAP

    best_val_loss = 1000 # should be large at start
    for epoch in range(TRAIN_EPOCHS):
        for image_data, target in trainset:
            results = train_step(image_data, target)
            cur_step = results[0]%steps_per_epoch
            print("epoch:{:2.0f} step:{:5.0f}/{}, lr:{:.6f}, giou_loss:{:7.2f}, conf_loss:{:7.2f}, prob_loss:{:7.2f}, total_loss:{:7.2f}"
                  .format(epoch, cur_step, steps_per_epoch, results[1], results[2], results[3], results[4], results[5]))

        if len(testset) == 0:
            print("configure TEST options to validate model")
            yolo.save_weights(os.path.join(TRAIN_CHECKPOINTS_FOLDER, TRAIN_MODEL_NAME))
            continue
        
        count, giou_val, conf_val, prob_val, total_val = 0., 0, 0, 0, 0
        for image_data, target in testset:
            results = validate_step(image_data, target)
            count += 1
            giou_val += results[0]
            conf_val += results[1]
            prob_val += results[2]
            total_val += results[3]
        # writing validate summary data
        with validate_writer.as_default():
            tf.summary.scalar("validate_loss/total_val", total_val/count, step=epoch)
            tf.summary.scalar("validate_loss/giou_val", giou_val/count, step=epoch)
            tf.summary.scalar("validate_loss/conf_val", conf_val/count, step=epoch)
            tf.summary.scalar("validate_loss/prob_val", prob_val/count, step=epoch)
        validate_writer.flush()
            
        print("\n\ngiou_val_loss:{:7.2f}, conf_val_loss:{:7.2f}, prob_val_loss:{:7.2f}, total_val_loss:{:7.2f}\n\n".
              format(giou_val/count, conf_val/count, prob_val/count, total_val/count))

        if TRAIN_SAVE_CHECKPOINT and not TRAIN_SAVE_BEST_ONLY:
            save_directory = os.path.join(TRAIN_CHECKPOINTS_FOLDER, TRAIN_MODEL_NAME+"_val_loss_{:7.2f}".format(total_val/count))
            yolo.save_weights(save_directory)
        if TRAIN_SAVE_BEST_ONLY and best_val_loss>total_val/count:
            save_directory = os.path.join(TRAIN_CHECKPOINTS_FOLDER, TRAIN_MODEL_NAME)
            yolo.save_weights(save_directory)
            best_val_loss = total_val/count
        if not TRAIN_SAVE_BEST_ONLY and not TRAIN_SAVE_CHECKPOINT:
            save_directory = os.path.join(TRAIN_CHECKPOINTS_FOLDER, TRAIN_MODEL_NAME)
            yolo.save_weights(save_directory)

    # measure mAP of trained custom model
    try:
        mAP_model.load_weights(save_directory) # use keras weights
        get_mAP(mAP_model, testset, score_threshold=TEST_SCORE_THRESHOLD, iou_threshold=TEST_IOU_THRESHOLD)
    except UnboundLocalError:
        print("You don't have saved model weights to measure mAP, check TRAIN_SAVE_BEST_ONLY and TRAIN_SAVE_CHECKPOINT lines in configs.py")


if __name__ == '__main__':
  app.run(debug=True)