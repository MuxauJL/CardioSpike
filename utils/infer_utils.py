import numpy as np



def expand_bbox(box, scale=1.0):
    x1, y1, x2, y2 = box
    center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
    w, h = x2 - x1, y2 - y1
    new_w, new_h = round(w * scale), round(h * scale)
    new_x, new_y = center_x - new_w // 2, center_y - new_h // 2
    new_x2, new_y2 = center_x + new_w // 2, center_y + new_h // 2
    return (max(0,new_x), max(0, new_y), new_x2, new_y2)

def square_box(box):
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    if w > h:
        y1 -= (w - h) / 2
        h = w
        y2 = y1 + h
    else:
        x1 -= (h - w) / 2
        w = h
        x2 = x1 + w

    return max(0, int(x1)),max(0, int(y1)), int(x2), int(y2)

def crop_faces(img, detection, scales=(1,)):
    faces = detection[0]
    if faces is None or len(faces) == 0:
        return []
    faces = faces.astype(np.int)
    crops = []

    for i, face in enumerate(faces):
        face_crops = []
        for scale in scales:
            bbox = face
            bbox = expand_bbox(bbox, scale)
            bbox = square_box(bbox)
            x1, y1, x2, y2 = max(0, bbox[0]), max(0, bbox[1]), bbox[2], bbox[3]


            face_crop = img[y1:y2, x1:x2, :]
            face_crops.append(face_crop)
        crops.append(face_crops)

    return crops
