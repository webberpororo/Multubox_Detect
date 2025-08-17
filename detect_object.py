import numpy as np
import math
import cv2
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, AveragePooling2D, Input, Activation,\
    BatchNormalization, GlobalAveragePooling2D, Add, multiply, Concatenate, GlobalMaxPooling2D, \
    DepthwiseConv2D, Flatten, Dropout, ReLU, Layer
from tensorflow.keras import layers
import itertools as it
import json
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import time
from download import download
import cv2
import json

# dataset_url = "https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/notebook/datasets/ssd_datasets.zip"
# path = "./"
# path = download(dataset_url, path, kind="zip", replace=True)

coco_root = "./datasets/"
anno_json = "./datasets/annotations/instances_val2017.json"

train_cls = ['background', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
             'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
             'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
             'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
             'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
             'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
             'kite', 'baseball bat', 'baseball glove', 'skateboard',
             'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
             'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
             'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
             'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
             'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
             'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
             'refrigerator', 'book', 'clock', 'vase', 'scissors',
             'teddy bear', 'hair drier', 'toothbrush']

train_cls_dict = {}
for i, cls in enumerate(train_cls):
    train_cls_dict[cls] = i


class generatedefaultbox():  # generate ssd default box
    def __init__(self):
        fk = 300 / np.array([8, 16, 32, 64, 100, 300], dtype = np.float32) # 不同縮放比例
        scale_rate = (0.95 - 0.1) / (len([4, 6, 6, 6, 4, 4]) - 1)
        scales = [0.1 + scale_rate * i for i in range(len([4, 6, 6, 6, 4, 4]))] + [1.0]
        self.default_boxes = []
        for idex, feature_size in enumerate([38, 19, 10, 5, 3, 1]):
            sk1 = scales[idex] # 當前尺寸
            sk2 = scales[idex + 1] # 下一次的尺寸
            sk3 = math.sqrt(sk1 * sk2)
            all_sizes = [(sk1, sk1)]
            for aspect_ratio in [[2], [2, 3], [2, 3], [2, 3], [2], [2]][idex]:
                w, h = sk1 * math.sqrt(aspect_ratio), sk1 / math.sqrt(aspect_ratio)
                all_sizes.append((w, h))
                all_sizes.append((h, w))
            all_sizes.append((sk3, sk3))
            for i, j in it.product(range(feature_size), repeat = 2): # 雙重迴圈，遍歷圖中的每個位置，計算center
                for w, h in all_sizes:
                    cx, cy = (j + 0.5) / fk[idex], (i + 0.5) / fk[idex]
                    self.default_boxes.append([cy, cx, h, w])
        def to_tlbr(cy, cx, h, w):
            return cy - h / 2, cx - w / 2, cy + h / 2, cx + w / 2
        self.default_boxes_tlbr = np.array([to_tlbr(*i) for i in self.default_boxes], dtype = 'float32')
        self.default_boxes = np.array(self.default_boxes, dtype = 'float32')

default_boxes_tlbr = generatedefaultbox().default_boxes_tlbr
default_boxes = generatedefaultbox().default_boxes

y1, x1, y2, x2 = tf.split(default_boxes_tlbr[:, :4], 4, axis = -1)
vol_anchors = (y2 - y1) * (x2 - x1)
matching_threshold = 0.5

#SSD Shot MultiBox Detector 目標資料預處理
def rand(a = 0., b = 1.):
    return np.random.rand() * (b - a) + a

def intersect(box_a, box_b): # 計算相交面積 box_a是一組框集合(shape = (n(框的數量), 4(上下左右))) box_b是單個框
    # 其範圍[ymin, xmin, ymax, xmax]
    settle = box_a
    settle = np.array(settle)
    boxas = settle
    if boxas.ndim == 1:
        boxas = np.expand_dims(boxas, axis=0)
    box_b = np.array(box_b)
    print(box_a)
    print(box_b)
    max_yx = np.minimum(boxas[:, 2:4], box_b[2:4]) # 右下角
    min_yx = np.maximum(boxas[:, :2], box_b[:2]) # 左上角
    inter = np.clip((max_yx - min_yx), a_min = 0, a_max = np.inf)
    return inter[:, 0] * inter[:, 1] #面積

def jaccard_numpy(box_a, box_b): # 計算相似度 IOU ***
    inter = intersect(box_a, box_b)
    settle = box_a
    settle = np.array(settle)
    boxas = settle
    if boxas.ndim == 1:
        boxas = np.expand_dims(boxas, axis=0)
    box_b = np.array(box_b)
    # 計算每個框的面積
    area_a = ((boxas[:, 2] - boxas[:, 0]) * (boxas[:, 3] - boxas[:, 1]))
    area_b = ((box_b[2] - box_b[0]) * (box_b[3] - box_b[1]))
    union = area_a + area_b - inter
    return inter / union

def random_sample_crop(image, boxes): # 隨機剪裁 ***
    height, width, _ = image.shape
    min_iou = np.random.choice([None, 0.1, 0.3, 0.5, 0.7, 0.9])

    if(min_iou is None):
        # boxes = boxes["bbox"]
        settel = []
        settel.append(boxes)
        return image, np.array(settel)

    for _ in range(50):
        image_t = image
        w = rand(0.3, 0.1) * width
        h = rand(0.3, 0.1) * height
        if h / w < 0.5 or h / w > 2:
            continue # 放棄這次剪裁

        left = rand() * (width - w)
        top = rand() * (height - h)
        rect = []
        rect.append(int(top))
        rect.append(int(left))
        rect.append(int(top + h))
        rect.append(int(left + w))
        print("rect", rect)
        print("boxes", boxes)
        overlap = jaccard_numpy(boxes, rect)
        drop_mask = overlap > 0 # boolean表示哪些邊界框有重疊
        if not drop_mask.any():
            continue
        if overlap[drop_mask].min() < min_iou and overlap[drop_mask].max() > (min_iou + 0.2):
            continue
        image_t = image_t[rect[0]:rect[2], rect[1]:rect[3], :]

        settle = boxes
        settle = np.array(settle)
        boxas = settle
        if boxas.ndim == 1:
            boxas = np.expand_dims(boxas, axis=0)
        print(boxas)
        # 更新座標
        centers = (boxas[:, :2] + boxas[:, 2:4]) / 2.0
        m1 = (rect[0] < centers[:, 0]) * (rect[1] < centers[:, 1]) # 前後兩者boolean相乘(=and)
        m2 = (rect[2] < centers[:, 0]) * (rect[3] > centers[:, 1])
        mask = m1 * m2 * drop_mask
        if not mask.any():
            continue
        boxes_t = boxas[mask, :].copy()
        boxes_t[:, :2] = np.maximum(boxes_t[:, :2], rect[:2])
        boxes_t[:, :2] -= rect[:2]
        boxes_t[:, 2:4] = np.minimum(boxes_t[:, 2:4], rect[2:4])
        boxes_t[:, 2:4] -= rect[:2]
        return image_t, boxes_t
    # boxes = boxes["bbox"]
    settel = []
    settel.append(boxes)
    return image, np.array(settel)

def ssd_bboxes_encode(boxes): # 對預設anchors編碼
    def jaccard_with_anchors(bbox): # 計算真實標記框與anchors之iou
        ymin = np.maximum(y1, bbox[0])
        xmin = np.maximum(x1, bbox[1])
        ymax = np.minimum(y2, bbox[2])
        xmax = np.minimum(x2, bbox[3])
        w = np.maximum(xmax - xmin, 0.)
        h = np.maximum(ymax - ymin, 0.)

        inter_vol = h * w
        union_vol = vol_anchors + (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) - inter_vol
        jaccard = inter_vol / union_vol
        return np.squeeze(jaccard)

    pre_scores = np.zeros((8732), dtype = np.float32) # 每隔錨點最大iou
    t_boxes = np.zeros((8732, 4), dtype = np.float32) # 座標
    t_label = np.zeros((8732), dtype = np.int64) # 標籤
    print(boxes)
    boxes = np.squeeze(boxes)
    print("boxes", boxes)
    for bbox in boxes:
        label = int(bbox[4])
        print("label: ", label)
        scores = jaccard_with_anchors(bbox)
        idx = np.argmax(scores)
        scores[idx] = 2.0 # 使最大iou之分數定為2確保被分配
        mask = (scores > matching_threshold) # boolean
        mask = mask & (scores > pre_scores) # & = and
        pre_scores = np.maximum(pre_scores, scores * mask) # 由於這邊是相乘，mask中true = 1 false = 0
        t_label = mask * label + (1 - mask) * label
        for i in range(4): # 更新四個邊界座標
            t_boxes[:, i] = mask * bbox[i] + (1 - mask) * t_boxes[:, i]

    index = np.nonzero(t_label) # 傳遞非零索引

    bboxes = np.zeroes(( 8732, 4), dtype = np.float32) # 中心點座標
    bboxes[:, [0, 1]] = (t_boxes[:, [0, 1]] + t_boxes[:, [2, 3]]) / 2 # 左上右下平均值
    bboxes[:, [2, 3]] = t_boxes[:, [2, 3]] - t_boxes[:, [0, 1]] # 高、寬

    bboxes_t = bboxes[index] # 儲存被批配到的座標並進行轉換
    default_boxes_t = default_boxes[index]
    bboxes_t[:, :2] = (bboxes_t[:, :2] - default_boxes_t[:, :2]) / (default_boxes_t[:, :2] * 0.1) # 縮放，增加學習效率
    tmp = np.maximum(bboxes_t[:, 2:4] / default_boxes_t[:, 2:4], 0.000001)
    bboxes_t[:, 2:4] = np.log(tmp) / 0.2
    bboxes[index] = bboxes_t

    num_match = np.array([len(np.nonzero(t_label)[0])], dtype = np.int32)
    print("output")
    print(bboxes)
    print(t_label)
    print(num_match)
    return bboxes, t_label.astype(np.int32), num_match

def preprocess_fn(img_id, image, box, is_training): # 對影像、邊界框預處理
    cv2.setNumThreads(2)

    def infer_data(image, input_shape): # 推論階段
        img_h, img_w = image.shape
        input_h, input_w = input_shape
        image = cv2.resize(image, (input_w, input_h))
        if len(image.shape) == 2: # 避免黑白影像
            image = np.expand_dims(image, axis = -1) # 根據axis值增加維度
            image = np.concatenate([image, image, image], axis = -1)

        return img_id, image, np.array((img_h, img_w))

    def data_aug(image, box, is_training, image_size = (640, 640)): # 訓練階段
        print("image2:", image)
        print(box)
        image = image.numpy()
        box = box.numpy().decode('utf-8')
        print("box", box)
        box = json.loads(box)
        ih, iw, _ = image.shape
        h, w = image_size
        if not is_training:
            return infer_data(image, image_size)
        # box = tf.cast(box, tf.float32)
        boxes = box["bbox"]
        imagess, settel = random_sample_crop(image, boxes)
        # boxes settle[] 為單獨座標
        print("final", settel)
        # settel = np.squeeze(settel)
        # box["bbox"] = settel
        # print(box)
        boxes = settel

        ih, iw, _ = imagess.shape
        imagess = cv2.resize(imagess, (w, h))
        flip = rand() < .5
        if flip:
            imagess = cv2.flip(imagess, 1, dst = None) # 水平翻轉
        if len(imagess.shape) == 2:
            imagess = np.expand_dims(imagess, axis = -1)
            imagess = np.concatenate([imagess, imagess, imagess], axis = -1)
        print("aaa")
        print(boxes)
        print(ih)
        print(iw)

        boxes[:, [0, 2]] = boxes[:, [0, 2]] / ih # 計算比例
        boxes[:, [1, 3]] = boxes[:, [1, 3]] / iw
        print("bbb")
        if flip:
            boxes[:, [1, 3]] = 1 - boxes[:, [3, 1]]

        boxess, label, num_match = ssd_bboxes_encode(boxes)
        print("ssd", boxess)
        print(label)
        print(num_match)
        return imagess, boxess, label, num_match

    print("image1:", image)
    print(box)
    images, boxs, labels, num_matchs = tf.py_function(
        func = lambda img, bx: data_aug(img, bx, is_training),
                                        inp = [image, box],
                                        Tout = [tf.float32, tf.float32, tf.int32, tf.int32]
    )
    # image, box = tf.py_function(func = data_aug, inp = [image, box], Tout = [tf.float32, tf.float32])
    # return data_aug(image, box, is_training, image_size = (300, 300))
    images.set_shape([640, 640, 3])
    boxs.set_shape([None, 4])
    labels.set_shape([None])
    num_matchs.set_shape([])
    print("image3: ", images)
    print(boxs)
    print("labels", labels)
    print("num", num_matchs)
    return image, boxs, labels, num_matchs

# def preprocess_fn(image, annotation, is_training):
#     if is_training:
#         image = tf.image.random_brightness(image, max_delta=0.4)
#         image = tf.image.random_contrast(image, lower=0.6, upper=1.4)
#         image = tf.image.random_saturation(image, lower=0.6, upper=1.4)
#     image = tf.image.per_image_standardization(image)
#     image = tf.transpose(image, [2, 0, 1])  # HWC to CHW
#     return image, annotation

def class_loss(logits, labels): # 用來計算AP
    labels = tf.one_hot(labels, tf.shape(logits)[-1], dtype = tf.float32)
    sigmoid_cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels = labels, logits = logits) # ***
    # 計算預測機率
    sigmoid = tf.sigmoid(logits)
    p_t = labels * sigmoid + (1 - labels) * (1 - 0.75) # 0.75為調節因子
    # focal loss 調節因子
    modulating_factor = tf.pow(1.0 - p_t, 2.0) # 減低易區分樣本損失貢獻，增加難區分樣本之比例
    alpha_weight_factor = labels * 0.75 + (1 - labels) * (1 - 0.75)

    focal_loss = modulating_factor * alpha_weight_factor * sigmoid_cross_entropy
    return focal_loss

def apply_eval(eval_parm_dict): # 對網路進行評估
    net = eval_parm_dict["net"] # 提取網路模型
    ds = eval_parm_dict["dataset"] # 提取數據集
    anno_json = eval_parm_dict["anno_json"] # 評估COCO之標註文件
    coco_metrics = cocometrics(anno_json = anno_json,
                               classes = train_cls,
                               num_classes = 10,
                               max_boxes = 100,
                               nms_threshold = 0.6,
                               min_score = 0.1)
    for data in ds:
        img_id = data['img_id']
        img_np = data['image']
        image_shape = data['image_shape']

        output = net(img_np, training = False)

        for batch_idx in range(img_np.shape[0]):
            pred_batch = {
                'boxes' : output[0].numpy()[batch_idx],
                'box_scores' : output[1].numpy()[batch_idx],
                'img_id' : int(img_id[batch_idx]),
                'image_shape' : image_shape[batch_idx]
            }
            coco_metrics.update(pred_batch)
        eval_metrics = coco_metrics.get_metrics() # results
        return eval_metrics

def apply_nms(all_boxes, all_scores, thres, max_boxes): # 非極大值抑制NMS
    y1 = all_boxes[:, 0] # 取上邊界
    x1 = all_boxes[:, 1] # 左
    y2 = all_boxes[:, 2]
    x2 = all_boxes[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = tf.argsort(all_scores, direction = 'DESCENDING') # 排序並返回索引
    keep = []
    while len(order) > 0:
        i = order[0]
        keep.append(i)

        if len(keep) >= max_boxes:
            break

        xx1 = tf.maximum(x1[i], tf.gather(x1, order[1:])) # 計算重疊左邊界
        yy1 = tf.maximum(y1[i], tf.gather(y1, order[1:]))
        xx2 = tf.minimum(x2[i], tf.gather(x2, order[1:]))
        yy2 = tf.minimum(y2[i], tf.gather(y2, order[1:]))

        w = tf.maximum(0.0, xx2 - xx1 + 1)
        h = tf.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h

        ovr = inter / (areas[i] + tf.gather(areas, order[1:]) - inter) # iou
        inds = tf.where(ovr <= thres)[:, 0] # 找出索引並提取結果的第一列
        order = tf.gather(order, inds + 1) # 輸出索引為inds +１的數值
    return tf.convert_to_tensor(keep)


class cocometrics:
    def __init__(self, anno_json, classes, num_classes, min_score, nms_threshold, max_boxes):
        self.num_classes = num_classes
        self.classes = classes
        self.min_score = min_score
        self.nms_threshold = nms_threshold
        self.max_boxes = max_boxes

        self.val_cls_dict = {i: cls for i, cls in enumerate(classes)}
        self.coco_gt = COCO(anno_json)
        cat_ids = self.coco_gt.loadCats(self.coco_gt.getCatIds())
        self.class_dict = {cat['name']: cat['id'] for cat in cat_ids}

        self.predictions = []
        self.img_ids = []

    def update(self, batch):
        pred_boxes = batch['boxes']
        box_scores = batch['box_scores']
        img_id = batch['img_id']
        h, w = batch['image_shape']

        final_boxes = []
        final_label = []
        final_score = []
        self.img_ids.append(img_id)

        for c in range(1, self.num_classes):
            class_box_scores = box_scores[:, c]
            score_mask = class_box_scores > self.min_score
            class_box_scores = tf.boolean_mask(class_box_scores, score_mask) # 返回boolean為true的數值
            class_boxes = tf.boolean_mask(pred_boxes, score_mask) * [h, w, h, w]

            if tf.reduce_any(score_mask): # 是否有任何true
                nms_index = apply_nms(class_boxes, class_box_scores, self.nms_threshold, self.max_boxes)
                class_boxes = tf.gather(class_boxes, nms_index)
                class_box_scores = tf.gather(class_box_scores, nms_index)
                final_boxes += class_boxes.numpy().tolist() # 添加列表
                final_score += class_box_scores.numpy().tolist()
                final_label += [self.class_dict[self.val_cls_dict[c]]] * len(class_box_scores) # print
        for loc, label, score in zip(final_boxes, final_label, final_score):
            res = {
                "image_id" : img_id,
                "bbox" : [loc[1], loc[0], loc[3] - loc[1], loc[2] - loc[0]],
                "score" : score,
                "category_id" : label
            }
            self.predictions.append(res)

    def get_metrics(self):
        with open('predictions.json', 'w') as f:
            json.dump(self.predictions, f)

        coco_dt = self.coco_gt.loadRes('predictions.json')
        E = COCOeval(self.coco_gt, coco_dt, iouType = 'bbox')
        E.params.imgIds = self.img_ids
        E.evaluate()
        E.accumulate()
        E.summarize()
        return E.stats[0]

class ssdinferwithdecoder(tf.Module):
    def __init__(self, network, default_boxes, ckpt_path):
        super(ssdinferwithdecoder, self).__init__()
        ckpt = tf.train.Checkpoint(model = network) # 保存模型權重
        ckpt.restore(ckpt_path) # 恢復模型權重
        self.network = network
        self.default_boxes = default_boxes
        self.prior_scaling_xy = 0.1
        self.prior_scaling_wh = 0.2

    @tf.function(input_signature = [tf.TensorSpec(shape = [None, None, None, 3], dtype = tf.float32)])
    # 裝飾器用來減少時間成本，簡單來說轉換成計算圖(裡面的w等都用第一次計算的數值)，input_signature為輸入的shape
    def __call__(self, x): # 計算最終座標 __call__可以被當作函數來用
        pred_loc, pred_label = self.network(x, training = False)
        default_boxes_xy = self.default_boxes[..., :2] # ...代表任意維度
        default_boxes_wh = self.default_boxes[..., 2:]
        pred_xy = pred_loc[..., :2] * self.prior_scaling_xy * default_boxes_wh + default_boxes_xy
        pred_wh = tf.exp(pred_loc[..., 2:] * self.prior_scaling_wh) * default_boxes_wh

        pred_xy_0 = pred_xy - pred_wh / 2.0
        pred_xy_1 = pred_xy + pred_wh / 2.0
        pred_xy = tf.concat([pred_xy_0, pred_xy_1], axis = -1)
        pred_xy = tf.clip_by_value(pred_xy, 0.0, 1.0)
        return pred_xy, pred_label

def init_net_param(network, initialize_mode = 'TruncatedNormal'):
    # 對神經網路的權重初始化，其目的為阻止activation爆炸消失等，使其更有效率
    for layer in network.layers:
        for var in layer.trainable_variables:
            if 'beta' not in var.name and 'gamma' not in var.name and 'bias' not in var.name:
                if initialize_mode == 'TruncatedNormal':
                    initializer = tf.keras.initializers.TruncatedNormal(stddev = 0.2)
                else:
                    initializer = initialize_mode
                var.assign(initializer(var.shape, dtype = var.dtype))

def get_lr(global_step, lr_init, lr_end, lr_max, warmup_epochs, total_epochs, steps_per_epoch):
    lr_each_step =[]
    total_steps = steps_per_epoch * total_epochs
    warmup_steps = steps_per_epoch * warmup_epochs
    for i in range(total_steps):
        if i < warmup_steps: #預熱階段
            lr = lr_init + (lr_max - lr_init) * i / warmup_steps # lr_init 逐漸增加到 lr_max
        else:
            lr = lr_end + (lr_max - lr_end) * (1. + np.cos(np.pi * (i - warmup_steps) / (total_steps - warmup_steps))) / 2.
            # lr_max逐漸減少至lr_end
        if lr < 0.0:
            lr = 0.0
        lr_each_step.append(lr)
    lr_each_step = np.array(lr_each_step).astype(np.float32)
    learning_rate = lr_each_step[global_step]

    return learning_rate

# (x_train, y_train), (x_test, y_test) = mnist.load_data()
# x_train = x_train / 255.0
# x_test = x_test / 255.0
# x_train = x_train.reshape(-1, 28, 28, 1) #這邊-1是代表不確定這邊的通道數是多少
# x_test = x_test.reshape(-1, 28, 28, 1)
# y_train = to_categorical(y_train, num_classes = 10)
# y_test = to_categorical(y_test, num_classes = 10)

def last_conv(filters, kernel = 3, stride = 1, padding = 'same'):
    depthwise = DepthwiseConv2D(kernel_size = kernel, strides = stride, padding = padding)
    batch = BatchNormalization(epsilon = 1e-3, momentum = 0.97)
    relu6 = ReLU(max_value = 6)
    conv = Conv2D(filters=filters, kernel_size=1, strides=1, padding=padding)
    return Sequential([depthwise, batch, relu6, conv])

class flattenconcat(tf.keras.layers.Layer):
    def __init__(self, num_ssd = 8732):
        super(flattenconcat, self).__init__()
        self.num_ssd = num_ssd

    def call(self, inputs):
        output = []
        batch_size = tf.shape(inputs[0])[0]
        for i in inputs:
            i = tf.transpose(i, perm = [0, 2, 3, 1])
            output.append(tf.reshape(i, (batch_size, -1)))
        res = tf.concat(output, axis = 1)
        return tf.reshape(res, (batch_size, self.num_ssd, -1))

class multibox(tf.keras.layers.Layer):
    def __init__(self, num_classes = 10):
        super(multibox, self).__init__()

        channels = [512, 1024, 512, 256, 256, 256]
        default = [4, 6, 6, 6, 4, 4]
        self.loc_layers = []
        self.cls_layers = []

        for k, channel in enumerate(channels):
            self.loc_layers.append(last_conv(4 * default[k], 3, 1, 'same'))
            self.cls_layers.append(last_conv(num_classes * default[k], 3, 1, 'same'))
            self.flatten = flattenconcat()

    def call(self, inputs):
        print(self.loc_layers)
        loc_output = []
        cls_output = []
        for i in range(len(self.loc_layers)):
            loc_output.append(self.loc_layers[i](inputs[i]))
            cls_output.append(self.cls_layers[i](inputs[i]))
        return self.flatten(loc_output), self.flatten(cls_output)

class ssd300vgg16(tf.keras.Model):
    def __init__(self):
        super(ssd300vgg16, self).__init__()

    def call(self, images):
        # x = Input(shape=(28, 28, 1), dtype="float32")
        # x = Input(shape=image, dtype="float32")

        x = layers.experimental.preprocessing.Resizing(224,
                224, interpolation = 'bilinear')(images)

        x = Conv2D(64, 3, strides = 1, padding = 'same')(x)
        x = Activation('relu')(x)
        x = Conv2D(64, 3, strides = 1, padding = 'same')(x)
        x = Activation('relu')(x)
        # x = MaxPooling2D(2, strides = 2)(x)

        x = Conv2D(128, 3, strides=1, padding = 'same')(x)
        x = Activation('relu')(x)
        x = Conv2D(128, 3, strides=1, padding = 'same')(x)
        x = Activation('relu')(x)
        # x = MaxPooling2D(2, strides = 2)(x)

        x = Conv2D(256, 3, strides=1, padding = 'same')(x)
        x = Activation('relu')(x)
        x = Conv2D(256, 3, strides=1, padding = 'same')(x)
        x = Activation('relu')(x)
        x = Conv2D(256, 1, strides=1, padding = 'same')(x)
        x = Activation('relu')(x)
        # x = MaxPooling2D(2, strides = 2)(x)

        x = Conv2D(512, 3, strides=1, padding = 'same')(x)
        x = Activation('relu')(x)
        x = Conv2D(512, 3, strides=1, padding = 'same')(x)
        x = Activation('relu')(x)
        x = Conv2D(512, 1, strides=1, padding = 'same')(x)
        x = Activation('relu')(x)
        block4 = x
        # x = MaxPooling2D(2, strides = 2)(x)

        x = Conv2D(512, 3, strides=1, padding = 'same')(x)
        x = Activation('relu')(x)
        x = Conv2D(512, 3, strides=1, padding = 'same')(x)
        x = Activation('relu')(x)
        x = Conv2D(512, 1, strides=1, padding = 'same')(x)
        x = Activation('relu')(x)
        # x = MaxPooling2D(3, strides = 1)(x)
        #
        # x = Flatten()(x)

        x = Conv2D(1024, 3, strides = 1, padding = 'same', dilation_rate = 6)(x)
        x = Activation('relu')(x)

        x = Conv2D(1024, 1, strides = 1, padding = 'same')(x)
        x = Activation('relu')(x)
        block7 = x

        x = Conv2D(256, 1, strides = 1, padding = 'same')(x)
        x = Activation('relu')(x)
        x = Conv2D(512, 3, strides = 2, padding = 'valid')(x)
        x = Activation('relu')(x)
        block8 = x

        x = Conv2D(128, 1, strides = 1, padding = 'same')(x)
        x = Activation('relu')(x)
        x = Conv2D(256, 3, strides = 2, padding = 'valid')(x)
        x = Activation('relu')(x)
        block9 = x

        x = Conv2D(128, 1, strides = 1, padding = 'same')(x)
        x = Activation('relu')(x)
        x = Conv2D(256, 3, strides = 1, padding = 'same')(x)
        x = Activation('relu')(x)
        block10 = x

        x = Conv2D(128, 1, strides = 1, padding = 'same')(x)
        x = Activation('relu')(x)
        x = Conv2D(256, 3, strides = 1, padding = 'same')(x)
        x = Activation('relu')(x)
        block11 = x

        multi_feature = (block4, block7, block8, block9, block10, block11)
        multiboxs = multibox(num_classes = 10)
        pred_loc, pred_label = multiboxs(multi_feature)
        training = False
        if not training:
            print(training)
            pred_label = tf.sigmoid(pred_label)
        pred_loc = tf.cast(pred_loc, dtype = 'float32')
        pred_label = tf.cast(pred_label, dtype = 'float32')

        return pred_loc, pred_label

def create_ssd_dataset(tfrecord_file, batch_size=32, is_training=True, num_parallel_calls=1):
    def parse_fn(example_proto):
        features = {
            'image': tf.io.FixedLenFeature([], tf.string), # 解析固定長度輸入要素
            'annotation': tf.io.FixedLenFeature([], tf.string),
            'img_id': tf.io.FixedLenFeature([], tf.int64)
        }
        parsed_features = tf.io.parse_single_example(example_proto, features) # TFRecord當中的單一個示例
        image = tf.image.decode_jpeg(parsed_features['image']) # 解碼jpeg to image
        # 將標註數據從 JSON 字符串解析為 Python 字典
        annotation_json = parsed_features['annotation']
        # annotation_dict = json.loads(annotation_json)

        # 將標註數據轉換為 TensorFlow 張量
        # bbox = tf.convert_to_tensor(annotation_json['bbox'], dtype=tf.float32)
        img_id = parsed_features['img_id']
        return img_id, image, annotation_json

    dataset = tf.data.TFRecordDataset(tfrecord_file)
    print("", dataset)
    dataset = dataset.map(parse_fn, num_parallel_calls=num_parallel_calls) # map會將每一個示例當作example_proto
    print("dataset:", dataset)

    if is_training:
        dataset = dataset.shuffle(buffer_size=10000)

    dataset = dataset.map(lambda img_id, images, annotation: preprocess_fn(img_id, images, annotation, is_training),
                          num_parallel_calls=num_parallel_calls)
    print(type(dataset))
    print(dataset)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset

def create_tfrecord(tfrecord_file, image_files, annotations):
    with tf.io.TFRecordWriter(tfrecord_file) as writer:
        for img_id, (image_file, annotation) in enumerate(zip(image_files, annotations)):
            # 讀取圖像並編碼為 JPEG 格式
            image = cv2.imread(image_file)
            _, image_encoded = cv2.imencode('.jpg', image)
            image_bytes = image_encoded.tobytes()

            # 將標註數據編碼為 JSON 格式
            annotation_json = json.dumps(annotation).encode('utf-8')

            # 創建 TFRecord 示例
            feature = {
                'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_bytes])),
                'annotation': tf.train.Feature(bytes_list=tf.train.BytesList(value=[annotation_json])),
                'img_id': tf.train.Feature(int64_list=tf.train.Int64List(value=[img_id]))
            }
            example = tf.train.Example(features=tf.train.Features(feature=feature))
            # print(example)

            # 寫入 TFRecord 文件
            writer.write(example.SerializeToString())

# 示例圖像文件和標註數據
image_files = ["D:/pythonProject/train/0008_jpg.rf.2e886b7fe6aebca28675b5bedb2d4b4c.jpg",
               "D:/pythonProject/train/0006_38a924bb-2d35-463f-920e-eb990606c9b9_jpg.rf.39b1c6ad144a1f859f044299d75597c8.jpg",
               "D:/pythonProject/train/0005_d817c3dd-6bd1-4ab2-bc21-26ed78d7052f_jpg.rf.0224ae26f3e1f7e150f4e4b7dc90aabc.jpg"]
annotations = [{'bbox': [252, 22, 377, 221], 'label': 'car'},
               {'bbox': [474, 23, 595, 117], 'label': 'car'},
               {'bbox': [261, 281, 411, 483], 'label': 'car'}]

# 創建 TFRecord 文件
create_tfrecord('dataset.tfrecord', image_files, annotations)

# load data
tfrecord_file = "D:\pythonProject\dataset.tfrecord"

dataset = create_ssd_dataset(tfrecord_file, batch_size=5, is_training=True)
# import tensorflow_datasets as tfds
#
# dataset, info = tfds.load('coco', with_info=True, as_supervised=True)
print(dataset)
dataset_size = len(list(dataset))

# Network definition and initialization
network = ssd300vgg16()
network(Input(shape=(640, 640, 3), dtype='float32'))  # Assuming input shape is (300, 300, 3)

# Define the learning rate
lr_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
    boundaries=[2 * dataset_size, 60 * dataset_size],
    values=[0.001, 0.05, 0.001 * 0.05]
)
optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule, momentum=0.9, nesterov=True)

# Define the forward procedure
@tf.function
def forward_fn(x, gt_loc, gt_label, num_matched_boxes):
    with tf.GradientTape() as tape:
        pred_loc, pred_label = network(x, training=True)
        mask = tf.cast(gt_label > 0, tf.float32)
        num_matched_boxes = tf.reduce_sum(tf.cast(num_matched_boxes, tf.float32))

        # Positioning loss
        mask_loc = tf.tile(tf.expand_dims(mask, -1), [1, 1, 4])
        smooth_l1 = tf.keras.losses.Huber()(gt_loc, pred_loc) * mask_loc
        loss_loc = tf.reduce_sum(tf.reduce_sum(smooth_l1, axis=-1), axis=-1)

        # Category loss
        loss_cls = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=gt_label, logits=pred_label)
        loss_cls = tf.reduce_sum(loss_cls, axis=[1, 2])

        loss = tf.reduce_sum((loss_cls + loss_loc) / num_matched_boxes)
        print("loss: ", loss)
    grads = tape.gradient(loss, network.trainable_variables)
    optimizer.apply_gradients(zip(grads, network.trainable_variables))
    print("loss: ", loss)
    return loss

# Training loop
print("=================== Starting Training =====================")
for epoch in range(60):
    start_time = time.time()
    for step, (image, gt_loc, gt_label, num_matched_boxes) in enumerate(dataset):
        loss = forward_fn(image, gt_loc, gt_label, num_matched_boxes)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Epoch:[{epoch + 1}/{60}], loss:{loss.numpy()} , time:{elapsed_time}s ")
network.save_weights("ssd-60_9.h5")
print("=================== Training Success =====================")

# 修改方向為資料集(dataset)的部分+cocometrics+aply_eval+create_ssd_datasets