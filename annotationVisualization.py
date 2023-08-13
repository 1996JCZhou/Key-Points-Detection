import cv2, json
from matplotlib import pyplot as plt

"""Load and visualize an unannotated image."""
img_path = "D:\Python Spyder\Key Point Detection\Dataset\Triangle_215_Keypoint_Labelme\images\DSC_0331.jpg"
img_bgr = cv2.imread(img_path)
plt.subplot(211)
plt.imshow(img_bgr[:,:,::-1])

"""Load the json annotation file in 'labelme' format for the image."""
labelme_path = "D:\Python Spyder\Key Point Detection\Dataset\Triangle_215_Keypoint_Labelme\labelme_jsons\DSC_0331.json"
with open(labelme_path, 'r', encoding='utf-8') as f:
    labelme = json.load(f)

"""Display the annotation information."""
print(labelme.keys())
print()
print(labelme['shapes'])
print()
"""
[{'label': 'sjb_rect',
  'points': [[866.1290322580645, 769.3548387096774], [3162.9032258064517, 2730.645161290323]],
  'group_id': None,
  'shape_type': 'rectangle',
  'flags': {}},
 {'label': 'angle_30',
  'points': [[3079.032258064516, 2654.838709677419]],
  'group_id': None,
  'shape_type': 'point',
  'flags': {}},
 {'label': 'angle_60',
  'points': [[927.4193548387096, 1167.741935483871]],
  'group_id': None,
  'shape_type': 'point',
  'flags': {}},
 {'label': 'angle_90',
  'points': [[2027.4193548387098, 832.258064516129]],
  'group_id': None,
  'shape_type': 'point',
  'flags': {}}]
"""
# -----------------------------------------------------------------------------------------------
"""Visualize the bounding box."""
# Visual configuration of the bounding box.
bbox_color = (255, 0, 0)
bbox_thickness = 5
bbox_labelstr = {
    'font_size': 6,
    'font_thickness': 14,
    'offset_x': 0,
    'offset_y': -80
}

for each_ann in labelme['shapes']:
    if each_ann['shape_type'] == 'rectangle': # Filter out the bounding box.
        bbox_label = each_ann['label'] 

        bbox_keypoint_A_xy = each_ann['points'][0]
        bbox_keypoint_B_xy = each_ann['points'][1]
        bbox_top_left_x = int(min(bbox_keypoint_A_xy[0], bbox_keypoint_B_xy[0]))
        bbox_top_left_y = int(min(bbox_keypoint_A_xy[1], bbox_keypoint_B_xy[1]))
        bbox_bottom_right_x = int(max(bbox_keypoint_A_xy[0], bbox_keypoint_B_xy[0]))
        bbox_bottom_right_y = int(max(bbox_keypoint_A_xy[1], bbox_keypoint_B_xy[1]))

        img_bgr = cv2.rectangle(img_bgr,
                                (bbox_top_left_x, bbox_top_left_y),
                                (bbox_bottom_right_x, bbox_bottom_right_y),
                                bbox_color,
                                bbox_thickness)

        img_bgr = cv2.putText(img_bgr,
                              bbox_label,
                              (bbox_top_left_x + bbox_labelstr['offset_x'], bbox_top_left_y + bbox_labelstr['offset_y']),
                              cv2.FONT_HERSHEY_SIMPLEX,
                              bbox_labelstr['font_size'],
                              bbox_color,
                              bbox_labelstr['font_thickness'])
# -----------------------------------------------------------------------------------------------
"""Visualize the annotated key points."""
# Visual configuration of the annotated key points.
kpt_colors = {
    'Angle_30': {'id': 0, 'color': [0,0,0], 'radius': 30, 'thickness': -1},
    'Angle_60': {'id': 1, 'color': [0,0,0], 'radius': 30, 'thickness': -1},
    'Angle_90': {'id': 2, 'color': [0,0,0], 'radius': 30, 'thickness': -1}
}

kpt_labelstr = {
    'font_size': 5,
    'font_thickness': 15,
    'offset_x': -250,
    'offset_y': 200
}

for each_ann in labelme['shapes']:
    if each_ann['shape_type'] == 'point': # Filter out the annotated key points.
        kpt_label = each_ann['label']

        kpt_x, kpt_y = int(each_ann['points'][0][0]), int(each_ann['points'][0][1])

        kpt_radius = kpt_colors[kpt_label]['radius']
        kpt_color = kpt_colors[kpt_label]['color']
        kpt_thickness = kpt_colors[kpt_label]['thickness'] # Line width (-1 means padding).

        img_bgr = cv2.circle(img_bgr,
                             (kpt_x, kpt_y),
                             kpt_radius,
                             kpt_color,
                             kpt_thickness)

        img_bgr = cv2.putText(img_bgr,
                              kpt_label,
                              (kpt_x+kpt_labelstr['offset_x'], kpt_y+kpt_labelstr['offset_y']),
                              cv2.FONT_HERSHEY_SIMPLEX,
                              kpt_labelstr['font_size'],
                              kpt_color,
                              kpt_labelstr['font_thickness'])

plt.subplot(212)
plt.imshow(img_bgr[:,:,::-1])
plt.show()
