import os
import cv2
import torch
import xml.etree.ElementTree as ET
from torch.utils.data import Dataset


class HollowKnightActionDataset(Dataset):
    def __init__(
        self,
        root_dir,
        annotation_file='annotations/annotations.xml',
        transform=None,
        target_size=(64, 64),
        key_frame=None
    ):
        self.root_dir = root_dir
        self.image_dir = os.path.join(root_dir, 'images', 'default')
        self.transform = transform
        self.target_size = target_size

        anno_path = os.path.join(root_dir, annotation_file)
        tree = ET.parse(anno_path)
        root = tree.getroot()

        self.images = []
        self.annotations = {}
        class_names = set()

        STUPID_COUNTER = 1
        STUPID_COUNTER_for_skips = 1
        

        
        for img in root.findall('image'):
            boxes = img.findall('box')

            if not boxes:
                STUPID_COUNTER_for_skips += 1
                continue
            
            
            STUPID_COUNTER += 1
            
            img_id = int(img.attrib['id'])
            img_name = img.attrib['name']

            tags = img.findall('tag')

            if tags:
                final_label = tags[0].attrib['label']
            else:
                final_label = boxes[0].attrib['label']

            class_names.add(final_label)

            b = boxes[0]
            bbox = (
                int(float(b.attrib['xtl'])),
                int(float(b.attrib['ytl'])),
                int(float(b.attrib['xbr'])),
                int(float(b.attrib['ybr']))
            )

            self.images.append({
                'id': img_id,
                'file_name': f'{img_name}.png'
            })

            self.annotations[img_id] = {
                'label': final_label,
                'bbox': bbox
            }

        if key_frame is not None:
            self.images = self.images[:key_frame]


        print(STUPID_COUNTER, STUPID_COUNTER_for_skips)
        self.classes = sorted(class_names)
        self.class_to_id = {c: i for i, c in enumerate(self.classes)}
        self.id_to_class = {i: c for c, i in self.class_to_id.items()}

        print('[Dataset] Classes:', self.classes)
        print('[Dataset] Total samples:', len(self.images))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_info = self.images[idx]
        img_id = img_info['id']
        img_path = os.path.join(self.image_dir, img_info['file_name'])

        image = cv2.imread(img_path)
        if image is None:
            return torch.zeros((3, *self.target_size)), -1

        h, w, _ = image.shape
        ann = self.annotations[img_id]

        label = self.class_to_id[ann['label']]

        x1, y1, x2, y2 = ann['bbox']
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            crop = image

        crop = cv2.resize(crop, self.target_size)
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

        if self.transform:
            crop = self.transform(image=crop)['image']
        else:
            crop = torch.from_numpy(crop).permute(2, 0, 1).float() / 255.0

        return crop, label
