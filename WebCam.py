# import opencv package
from __future__ import division

# import tkinter for gui
import tkinter as tk
from tkinter import Message ,Text
import tkinter.ttk as ttk
import tkinter.font as font
from PIL import Image, ImageTk
# to mesure time detecting an image
import time
import random
import sys
import pickle
import re
import cv2
import os
# import numpy package to operate matrix
import numpy as np
# read tensorflow packages' classes
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from faster_rcnn import config, data_generators
import faster_rcnn.roi_helpers as roi_helpers
from tensorflow.python.keras.utils import generic_utils
import tensorflow as tf
# read necessary functions
from faster_rcnn import resnet as nn
from faster_rcnn import losses as lossess

# create tkinter gui
window = tk.Tk()
window.title("Face Detect")
dialog_title = 'QUIT'
dialog_text = 'Are you sure?'
window.geometry('500x600')
window.configure(background='#01122c')

# create message to show title of program
message1 = tk.Label(window, text="face detection System", width=30, height=2, bg="#01122c",  fg="white", font=("Courier", 18, "italic bold")) 
message1.place(x=10, y=10)
# set status of program
message = tk.Label(window, text="" ,bg="#b3d1ff", fg="black", width=30, height=3, activebackground = "yellow", font=('times', 15, ' bold ')) 
message.place(x=60, y=450)
# add image to GUI
image = Image.open("face1.jpg")
photo = ImageTk.PhotoImage(image)
img_label = tk.Label(image=photo)
img_label.image = photo
img_label.place(x=90, y=140)


#sys.setrecursionlimit(40000)
# create config class instance
C = config.Config()
model_path_regex = re.match("^(.+)(\.hdf5)$", C.model_path)
C.model_path = './model_frcnn.hdf5'
C.num_rois = 32
# set net type to use
C.network = 'resnet50'
# create empty models
model_rpn = Model()
model_classifier = Model()
model_classifier_only = Model()
# store class name
class_mapping = []

def format_img_size(img, C):
	# change image size based on config
	img_min_side = float(C.im_size)
	(height,width,_) = img.shape
    # resize sub img according to img_min_side
	if width <= height:
		ratio = img_min_side/width
		new_height = int(ratio * height)
		new_width = int(img_min_side)
	else:
		ratio = img_min_side/height
		new_width = int(ratio * width)
		new_height = int(img_min_side)
	img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
	return img, ratio

def format_img_channels(img, C):
	# formats the image channels based on config
	img = img[:, :, (2, 1, 0)]
	img = img.astype(np.float32)
	img[:, :, 0] -= C.img_channel_mean[0]
	img[:, :, 1] -= C.img_channel_mean[1]
	img[:, :, 2] -= C.img_channel_mean[2]
	img /= C.img_scaling_factor
	img = np.transpose(img, (2, 0, 1))
	img = np.expand_dims(img, axis=0)
	return img

def format_img(img, C):
	# formats an image for model prediction based on config
	img, ratio = format_img_size(img, C)
	img = format_img_channels(img, C)
	return img, ratio

# Method to transform the coordinates of the bounding box to its original size
def get_real_coordinates(ratio, x1, y1, x2, y2):
    # change to original size
	real_x1 = int(round(x1 // ratio))
	real_y1 = int(round(y1 // ratio))
	real_x2 = int(round(x2 // ratio))
	real_y2 = int(round(y2 // ratio))

	return (real_x1, real_y1, real_x2 ,real_y2)

# this is used to get all training data
# get data from annotation input path
def get_data(input_path):
	found_bg = False
	all_imgs = {}

	classes_count = {}
    # store the classes' name
	class_mapping = {}

	visualise = True

	with open(input_path,'r') as f:

		print('Parsing annotation files')
        # analyze every line of annotation file
		for line in f:
			line_split = line.strip().split(',')
			(filename,x1,y1,x2,y2,class_name) = line_split
            # add to class_name
			if class_name not in classes_count:
				classes_count[class_name] = 1
			else:
				classes_count[class_name] += 1
            # add to class mapping if is not there
			if class_name not in class_mapping:
                # "bg" class_name
				if class_name == 'bg' and found_bg == False:
					print('Found class name with special name bg. Will be treated as a background region (this is usually for hard negative mining).')
					found_bg = True
				class_mapping[class_name] = len(class_mapping)
            # get image properties if it is new
			if filename not in all_imgs:
				all_imgs[filename] = {}

				img = cv2.imread(filename)
				(rows,cols) = img.shape[:2]
				all_imgs[filename]['filepath'] = filename
                # get image width and height
				all_imgs[filename]['width'] = cols
				all_imgs[filename]['height'] = rows
                # get bounding box
				all_imgs[filename]['bboxes'] = []
				all_imgs[filename]['imageset'] = 'test'
            # add image box bound
			all_imgs[filename]['bboxes'].append({'class': class_name, 'x1': int(x1), 'x2': int(x2), 'y1': int(y1), 'y2': int(y2)})


		all_data = []
        # get all data
		for key in all_imgs:
			all_data.append(all_imgs[key])

		# make sure the bg class is last in the list
		if found_bg:
			if class_mapping['bg'] != len(class_mapping) - 1:
				key_to_switch = [key for key in class_mapping.keys() if class_mapping[key] == len(class_mapping)-1][0]
				val_to_switch = class_mapping['bg']
				class_mapping['bg'] = len(class_mapping) - 1
				class_mapping[key_to_switch] = val_to_switch

		return all_data, classes_count, class_mapping


def TrainImages():
    # show message
    res = "Training Started"
    message.configure(text= res)
    train_path = "./annotation.txt"

	# set the path to initial weights on backend and model
    global C
    C.base_net_weights = 'resnet50_weights_tf_dim_ordering_tf_kernels.h5'

    # get classes mapping
    global class_mapping

    # get data from training dataset
    train_imgs, classes_count, class_mapping = get_data(train_path)
    val_imgs, _, _ = get_data(train_path)

    # add background class
    if 'bg' not in classes_count:
        classes_count['bg'] = 0
        class_mapping['bg'] = len(class_mapping)

    C.class_mapping = class_mapping

    inv_map = {v: k for k, v in class_mapping.items()}

    print('Training images per class:')
    print(f'Num classes (including bg) = {len(classes_count)}')

    config_output_filename = 'config.pickle'
    # store to the config output file
    with open(config_output_filename, 'wb') as config_f:
        pickle.dump(C,config_f)
        print(f'Config has been written to {config_output_filename}, and can be loaded when testing to ensure correct results')

    random.shuffle(train_imgs)
    # get anchor point from image
    data_gen_train = data_generators.get_anchor_gt(train_imgs, classes_count, C, nn.get_img_output_length, 'tf', mode='train')
    data_gen_val = data_generators.get_anchor_gt(val_imgs, classes_count, C, nn.get_img_output_length, 'tf', mode='val')
    # define input imag shape
    input_shape_img = (None, None, 3)

    img_input = Input(shape=input_shape_img)
    roi_input = Input(shape=(None, 4))

    # define shared_layer of the base network(resnet)
    shared_layers = nn.nn_base(img_input, trainable=True)

    # define the RPN, built on the base layers
    num_anchors = len(C.anchor_box_scales) * len(C.anchor_box_ratios)
    rpn = nn.rpn(shared_layers, num_anchors)
    # define classifier using shared_layers
    classifier = nn.classifier(shared_layers, roi_input, C.num_rois, nb_classes=len(classes_count), trainable=True)

    # set rpn model
    global model_rpn
    model_rpn = Model(img_input, rpn[:2])

    # set model_classifier input: [img_input, roi_input], output: classifier
    global model_classifier
    model_classifier = Model([img_input, roi_input], classifier)

    # set model_classifier_only for only classifier not bounding
    global model_classifier_only
    model_classifier_only = Model([img_input, roi_input], classifier)
    # this is a model that holds both the RPN and the classifier, used to load/save weights for the models
    model_all = Model([img_input, roi_input], rpn[:2] + classifier)
    # loading base net weight
    print(f'loading weights from {C.base_net_weights}')
    model_rpn.load_weights(C.base_net_weights, by_name=True)
    model_classifier.load_weights(C.base_net_weights, by_name=True)

    # compile models with adam optimizer
    optimizer = Adam(lr=1e-5)
    optimizer_classifier = Adam(lr=1e-5)
    model_rpn.compile(optimizer=optimizer, loss=[lossess.rpn_loss_cls(num_anchors), lossess.rpn_loss_regr(num_anchors)])
    model_classifier.compile(optimizer=optimizer_classifier, loss=[lossess.class_loss_cls, lossess.class_loss_regr(len(classes_count)-1)], metrics={f'dense_class_{len(classes_count)}': 'accuracy'})
    model_all.compile(optimizer='sgd', loss='mae')

    # set epoch length and number of epoch
    epoch_length = 1000
    num_epochs = 2#200
    iter_num = 0
    # losses and rpn_accuracy
    losses = np.zeros((epoch_length, 5))
    rpn_accuracy_rpn_monitor = []
    rpn_accuracy_for_epoch = []

    # set start_time for training
    start_time = time.time()

    best_loss = np.Inf
    # inverse class_mapping
    class_mapping_inv = {v: k for k, v in class_mapping.items()}
    print('Starting training')

    vis = True
    # training model num_epochs times
    for epoch_num in range(num_epochs):
        # show progbar in training
        progbar = generic_utils.Progbar(epoch_length)
        print(f'Epoch {epoch_num + 1}/{num_epochs}')
        # training until limit condition
        while True:
            try:
                if len(rpn_accuracy_rpn_monitor) == epoch_length and C.verbose:
                    mean_overlapping_bboxes = float(sum(rpn_accuracy_rpn_monitor))/len(rpn_accuracy_rpn_monitor)
                    rpn_accuracy_rpn_monitor = []
                    print(f'Average number of overlapping bounding boxes from RPN = {mean_overlapping_bboxes} for {epoch_length} previous iterations')
                    if mean_overlapping_bboxes == 0:
                        print('RPN is not producing bounding boxes that overlap the ground truth boxes. Check RPN settings or keep training.')
                # get data
                X, Y, img_data = next(data_gen_train)

                # get loss of rpn in running a single gradient update on a single batch of data.
                loss_rpn = model_rpn.train_on_batch(X, Y)

                # get prediction about batch of data
                P_rpn = model_rpn.predict_on_batch(X)

                R = roi_helpers.rpn_to_roi(P_rpn[0], P_rpn[1], C, use_regr=True, overlap_thresh=0.7, max_boxes=300)

                # calc_iou converts from (x1,y1,x2,y2) to (x,y,w,h) format
                X2, Y1, Y2, IouS = roi_helpers.calc_iou(R, img_data, C, class_mapping)

                if X2 is None:
                    rpn_accuracy_rpn_monitor.append(0)
                    rpn_accuracy_for_epoch.append(0)
                    continue
                # get positive and negative sample
                neg_samples = np.where(Y1[0, :, -1] == 1)
                pos_samples = np.where(Y1[0, :, -1] == 0)

                if len(neg_samples) > 0:
                    neg_samples = neg_samples[0]
                else:
                    neg_samples = []

                if len(pos_samples) > 0:
                    pos_samples = pos_samples[0]
                else:
                    pos_samples = []

                rpn_accuracy_rpn_monitor.append(len(pos_samples))
                rpn_accuracy_for_epoch.append((len(pos_samples)))

                if C.num_rois > 1:
                    if len(pos_samples) < C.num_rois//2:
                        selected_pos_samples = pos_samples.tolist()
                    else:
                        selected_pos_samples = np.random.choice(pos_samples, C.num_rois//2, replace=False).tolist()
                    try:
                        selected_neg_samples = np.random.choice(neg_samples, C.num_rois - len(selected_pos_samples), replace=False).tolist()
                    except:
                        selected_neg_samples = np.random.choice(neg_samples, C.num_rois - len(selected_pos_samples), replace=True).tolist()

                    sel_samples = selected_pos_samples + selected_neg_samples
                else:
                    # in the extreme case where num_rois = 1, we pick a random pos or neg sample
                    selected_pos_samples = pos_samples.tolist()
                    selected_neg_samples = neg_samples.tolist()
                    if np.random.randint(0, 2):
                        sel_samples = random.choice(neg_samples)
                    else:
                        sel_samples = random.choice(pos_samples)
                # get loss class from classifer model
                loss_class = model_classifier.train_on_batch([X, X2[:, sel_samples, :]], [Y1[:, sel_samples, :], Y2[:, sel_samples, :]])

                losses[iter_num, 0] = loss_rpn[1]
                losses[iter_num, 1] = loss_rpn[2]

                losses[iter_num, 2] = loss_class[1]
                losses[iter_num, 3] = loss_class[2]
                losses[iter_num, 4] = loss_class[3]
                # progress bar update
                progbar.update(iter_num+1, [('rpn_cls', losses[iter_num, 0]), ('rpn_regr', losses[iter_num, 1]),
                                        ('detector_cls', losses[iter_num, 2]), ('detector_regr', losses[iter_num, 3])])

                iter_num += 1

                if iter_num == epoch_length:
                    # get loss of rpn and class
                    loss_rpn_cls = np.mean(losses[:, 0])
                    loss_rpn_regr = np.mean(losses[:, 1])
                    loss_class_cls = np.mean(losses[:, 2])
                    loss_class_regr = np.mean(losses[:, 3])
                    class_acc = np.mean(losses[:, 4])

                    mean_overlapping_boxes = float(sum(rpn_accuracy_for_epoch)) / len(rpn_accuracy_for_epoch)
                    rpn_accuracy_for_epoch = []
                    # print training states
                    if C.verbose:
                        print(f'Mean number of bounding boxes from RPN overlapping ground truth boxes: {mean_overlapping_boxes}')
                        print(f'Classifier accuracy for bounding boxes from RPN: {class_acc}')
                        print(f'Loss RPN classifier: {loss_rpn_cls}')
                        print(f'Loss RPN regression: {loss_rpn_regr}')
                        print(f'Loss Detector classifier: {loss_class_cls}')
                        print(f'Loss Detector regression: {loss_class_regr}')
                        print(f'Elapsed time: {time.time() - start_time}')
                    # get current loss
                    curr_loss = loss_rpn_cls + loss_rpn_regr + loss_class_cls + loss_class_regr
                    iter_num = 0
                    start_time = time.time()

                    # limit condition
                    if curr_loss < best_loss:
                        if C.verbose:
                            print(f'Total loss decreased from {best_loss} to {curr_loss}, saving weights')
                        best_loss = curr_loss
                    model_all.save_weights(model_path_regex.group(1) + "_" + '{:04d}'.format(epoch_num) + model_path_regex.group(2))
                    break

            except Exception as e:
                print(f'Exception: {e}')
                continue

    print('Training complete, exiting.')
    res = "Image Training ended"
    message.configure(text= res)


def TrackImages():
    # set number of features to 1024
    num_features = 1024
    # read config file
    with open('config.pickle', 'rb') as f_in:
        C = pickle.load(f_in)
    # read class_mapping from config
    class_mapping = C.class_mapping

    if 'bg' not in class_mapping:
        class_mapping['bg'] = len(class_mapping)
    # inverse mapping from class_mapping
    class_mapping = {v: k for k, v in class_mapping.items()}
    print(class_mapping)
    # set classes mapping
    class_to_color = {class_mapping[v]: np.random.randint(0, 255, 3) for v in class_mapping}
    C.num_rois = 32
    input_shape_img = (None, None, 3)
    input_shape_features = (None, None, num_features)


    img_input = Input(shape=input_shape_img)
    roi_input = Input(shape=(C.num_rois, 4))
    feature_map_input = Input(shape=input_shape_features)

    # define the base network (resnet here, can be VGG, Inception, etc)
    shared_layers = nn.nn_base(img_input, trainable=True)

    # define the RPN, built on the base layers
    num_anchors = len(C.anchor_box_scales) * len(C.anchor_box_ratios)
    rpn_layers = nn.rpn(shared_layers, num_anchors)
    # set classifier
    classifier = nn.classifier(feature_map_input, roi_input, C.num_rois, nb_classes=len(class_mapping), trainable=True)

    # set rpn model
    model_rpn = Model(img_input, rpn_layers)
    model_classifier_only = Model([feature_map_input, roi_input], classifier)

    # set classifier model
    model_classifier = Model([feature_map_input, roi_input], classifier)

    # loading weight from trained model(.h5 file)
    print(f'Loading weights from {C.model_path}')
    model_rpn.load_weights(C.model_path, by_name=True)
    model_classifier.load_weights(C.model_path, by_name=True)

    # compile model
    model_rpn.compile(optimizer='sgd', loss='mse')
    model_classifier.compile(optimizer='sgd', loss='mse')
    bbox_threshold = 0.8

    cam = cv2.VideoCapture(0)
    # continue to get image from camera
    while(True):
        ret, img =cam.read()

        st = time.time()
        # format image from camera image
        X, ratio = format_img(img, C)
        X = np.transpose(X, (0, 2, 3, 1))

        # get the feature maps and output from the RPN
        [Y1, Y2, F] = model_rpn.predict(X)

        # rpn to roi
        R = roi_helpers.rpn_to_roi(Y1, Y2, C, overlap_thresh=0.7)

        # convert from (x1,y1,x2,y2) to (x,y,w,h)
        R[:, 2] -= R[:, 0]
        R[:, 3] -= R[:, 1]

        # apply the spatial pyramid pooling to the proposed regions
        bboxes = {}
        probs = {}

        for jk in range(R.shape[0]//C.num_rois + 1):
            # expand dimension to tensor
            ROIs = np.expand_dims(R[C.num_rois*jk:C.num_rois*(jk+1), :], axis=0)
            if ROIs.shape[1] == 0:
                break

            if jk == R.shape[0]//C.num_rois:
                # pad R
                curr_shape = ROIs.shape
                target_shape = (curr_shape[0],C.num_rois,curr_shape[2])
                ROIs_padded = np.zeros(target_shape).astype(ROIs.dtype)
                ROIs_padded[:, :curr_shape[1], :] = ROIs
                ROIs_padded[0, curr_shape[1]:, :] = ROIs[0, 0, :]
                ROIs = ROIs_padded
            # get class probablity
            [P_cls, P_regr] = model_classifier_only.predict([F, ROIs])

            for ii in range(P_cls.shape[1]):
                # apply threshold
                if np.max(P_cls[0, ii, :]) < bbox_threshold or np.argmax(P_cls[0, ii, :]) == (P_cls.shape[2] - 1):
                    continue
                # get max probablity class name
                cls_name = class_mapping[np.argmax(P_cls[0, ii, :])]
                # if cls_name is new
                if cls_name not in bboxes:
                    bboxes[cls_name] = []
                    probs[cls_name] = []

                (x, y, w, h) = ROIs[0, ii, :]
                cls_num = np.argmax(P_cls[0, ii, :])

                try:
                    (tx, ty, tw, th) = P_regr[0, ii, 4*cls_num:4*(cls_num+1)]
                    tx /= C.classifier_regr_std[0]
                    ty /= C.classifier_regr_std[1]
                    tw /= C.classifier_regr_std[2]
                    th /= C.classifier_regr_std[3]
                    x, y, w, h = roi_helpers.apply_regr(x, y, w, h, tx, ty, tw, th)
                except:
                    pass
                # get box bound and their probablity
                bboxes[cls_name].append([C.rpn_stride*x, C.rpn_stride*y, C.rpn_stride*(x+w), C.rpn_stride*(y+h)])
                probs[cls_name].append(np.max(P_cls[0, ii, :]))

        all_dets = []

        for key in bboxes:
            bbox = np.array(bboxes[key])
            # get new boxes and their probablity
            new_boxes, new_probs = roi_helpers.non_max_suppression_fast(bbox, np.array(probs[key]), overlap_thresh=0.5)
            for jk in range(new_boxes.shape[0]):
                (x1, y1, x2, y2) = new_boxes[jk,:]
                # get coordination
                (real_x1, real_y1, real_x2, real_y2) = get_real_coordinates(ratio, x1, y1, x2, y2)

                cv2.rectangle(img,(real_x1, real_y1), (real_x2, real_y2), (int(class_to_color[key][0]), int(class_to_color[key][1]), int(class_to_color[key][2])),2)

                textLabel = f'{key}: {int(100*new_probs[jk])}'
                all_dets.append((key,100*new_probs[jk]))

                (retval,baseLine) = cv2.getTextSize(textLabel,cv2.FONT_HERSHEY_SIMPLEX,1,1)
                textOrg = (real_x1, real_y1-0)
                # show box bound and text label
                cv2.rectangle(img, (textOrg[0] - 5, textOrg[1]+baseLine - 5), (textOrg[0]+retval[0] + 5, textOrg[1]-retval[1] - 5), (0, 0, 0), 2)
                cv2.rectangle(img, (textOrg[0] - 5,textOrg[1]+baseLine - 5), (textOrg[0]+retval[0] + 5, textOrg[1]-retval[1] - 5), (255, 255, 255), -1)
                cv2.putText(img, textLabel, textOrg, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        print(f'Elapsed time = {time.time() - st}')
        print(all_dets)
        # show box bounded image and label
        cv2.imshow("Detection", img)
        if (cv2.waitKey(1)==ord('q')):
            break
    cam.release()
    cv2.destroyAllWindows()


# create train button
trainImg = tk.Button(window, text="Train Images", command=TrainImages, bg="#1a75ff", width=10, height=1, activebackground="#0066ff", font=('times', 15, ' bold '))
trainImg.place(x=90, y=380)
# create Track image button
trackImg = tk.Button(window, text="Track Images", command=TrackImages, bg="#1a75ff", width=10  ,height=1, activebackground="#0066ff", font=('times', 15, ' bold '))
trackImg.place(x=260, y=380)
# create quit button
quitWindow = tk.Button(window, text="Quit", command=window.destroy, bg="red", width=5, activebackground = "red", font=('times', 15, ' bold '))
quitWindow.place(x=420, y=10)
# windows settings
window.resizable(False, False)
window.mainloop()
