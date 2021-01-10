import cv2
import numpy as np

def init(cfg_file, weights_file, namesfile):
    with open(namesfile) as f:
        classes = [line.strip() for line in f.readlines()]

    # Load the network architecture
    yolo_model = cv2.dnn.readNet(weights_file,cfg_file)

    # generate different colors for different classes 
    colors = np.random.uniform(0, 255, size=(len(classes), 3))

    return yolo_model,classes, colors

# function to get the output layer names 
# in the architecture
def get_output_layers(net):
    
    layer_names = net.getLayerNames()
    
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    return output_layers

# function to draw bounding box on the detected object with class name
def draw_bounding_box(img, class_id, confidence, x, y, x_plus_w, y_plus_h, COLORS,classes):

    label = str(classes[class_id])

    color = COLORS[class_id]

    cv2.rectangle(img, (x,y), (x_plus_w,y_plus_h), color, 2)

    cv2.putText(img, label, (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return

# run inference through the network
# and gather predictions from output layers
def inference(net):

    outs = net.forward(get_output_layers(net))

    return outs

# for each detetion from each output layer 
# get the confidence, class id, bounding box params
# and ignore weak detections (confidence < 0.5)
def get_boxes(outs, conf_threshold, im_width, im_height):
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            if class_id == 39: # detect only person
                confidence = scores[class_id]
                if confidence > conf_threshold:
                    center_x = int(detection[0] * im_width)
                    center_y = int(detection[1] * im_height)
                    w = int(detection[2] * im_width)
                    h = int(detection[3] * im_height)
                    x = center_x - w / 2
                    y = center_y - h / 2
                    class_ids.append(class_id)
                    confidences.append(float(confidence))
                    boxes.append([x, y, w, h])

    return class_ids, confidences,boxes

# apply non-max suppression
def nms(boxes, confidences, conf_threshold, nms_threshold):
    
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
    indices = range(len(boxes))
    return indices

# draw bounding boxes 
def draw_bb(indices, image, class_ids, confidences, boxes, COLORS, classes):
    # go through the detections remaining
    # after nms and draw bounding box
    for i in range(len(boxes)):
        box = boxes[i]
        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3]
        draw_bounding_box(image, class_ids[i], confidences[i], round(x), round(y), round(x+w), round(y+h),COLORS,classes)

    return
