import os, json, const

bbox_class = {'sjb_rect': 0}
keypoint_class = ['angle_30', 'angle_60', 'angle_90']

choices = ['train', 'val']
for choice in choices:
    Labels_Path = os.path.join(os.path.join(const.OUTPUT_PATH, const.LABELS), choice)
    List_Path = os.listdir(Labels_Path)
    for label in List_Path:
        labelme_path = os.path.join(Labels_Path, label)
        with open(labelme_path, 'r', encoding='utf-8') as f:
            labelme = json.load(f)
            # labelme.keys() = ['version', 'flags', 'shapes', 'imagePath', 'imageData', 'imageHeight', 'imageWidth']

        img_height = labelme['imageHeight']
        img_width  = labelme['imageWidth']

        # Generate a txt file in YOLO format for this 'labelme' json file.
        suffix = label.split('.')[0]
        suffix = os.path.join(suffix + '.txt')
        yolo_txt_path = os.path.join(os.path.join(os.path.join(const.OUTPUT_PATH, const.LABELS), choice), suffix)

        # Iterate each annotation, if a box is encountered,
        # find all the key points in this box and write them to the txt file in order.
        with open(yolo_txt_path, 'w', encoding='utf-8') as f:
            for each_ann in labelme['shapes']: # Iterate each annotation.
                if each_ann['shape_type'] == 'rectangle': # Filter out all the bounding boxes.
                    yolo_str = ''

                    # Write the ID of the bounding box in the ".txt" file.
                    bbox_class_id = bbox_class[each_ann['label']]
                    yolo_str += '{} '.format(bbox_class_id)
                # -----------------------------------------------------------------------------------------
                    bbox_top_left_x = int(min(each_ann['points'][0][0], each_ann['points'][1][0]))
                    bbox_bottom_right_x = int(max(each_ann['points'][0][0], each_ann['points'][1][0]))
                    bbox_top_left_y = int(min(each_ann['points'][0][1], each_ann['points'][1][1]))
                    bbox_bottom_right_y = int(max(each_ann['points'][0][1], each_ann['points'][1][1]))

                    bbox_center_x = int((bbox_top_left_x + bbox_bottom_right_x) / 2)
                    bbox_center_y = int((bbox_top_left_y + bbox_bottom_right_y) / 2)

                    bbox_width = bbox_bottom_right_x - bbox_top_left_x
                    bbox_height = bbox_bottom_right_y - bbox_top_left_y

                    # Normalized coordinates of box center point.
                    bbox_center_x_norm = bbox_center_x / img_width
                    bbox_center_y_norm = bbox_center_y / img_height

                    # Normalized width and height of the bounding box.
                    bbox_width_norm = bbox_width / img_width
                    bbox_height_norm = bbox_height / img_height

                    yolo_str += '{:.5f} {:.5f} {:.5f} {:.5f} '.format(bbox_center_x_norm, bbox_center_y_norm, bbox_width_norm, bbox_height_norm)
                # -----------------------------------------------------------------------------------------
                    # Find all the key points in the box and store them in the dictionary 'bbox_keypoints_dict'.
                    bbox_keypoints_dict = {}
                    for each_ann in labelme['shapes']: # Iterate each annotation.
                        # Filter out all the key points, which may not reside in the above mentioned bounding box.
                        if each_ann['shape_type'] == 'point':

                            x = int(each_ann['points'][0][0])
                            y = int(each_ann['points'][0][1])
                            label = each_ann['label']

                            # Filter out all the key points, which reside in the above mentioned bounding box.
                            if (x>bbox_top_left_x) & (x<bbox_bottom_right_x) & (y<bbox_bottom_right_y) & (y>bbox_top_left_y):
                                bbox_keypoints_dict[label] = [x, y]

                    # Write all the found key points into the txt file in order: 'angle_30', 'angle_60', 'angle_90'.
                    for each_class in keypoint_class:
                        if each_class in bbox_keypoints_dict:
                            keypoint_x_norm = bbox_keypoints_dict[each_class][0] / img_width
                            keypoint_y_norm = bbox_keypoints_dict[each_class][1] / img_height
                            yolo_str += '{:.5f} {:.5f} {} '.format(keypoint_x_norm, keypoint_y_norm, 2) # '2' for visible, not occluded; '1' for occluded; '0' for no point.
                        else:
                            yolo_str += '0 0 0 '
                    f.write(yolo_str + '\n')
                # -----------------------------------------------------------------------------------------
            print('{} --> {} completed!'.format(label, suffix))
        os.remove(labelme_path)
