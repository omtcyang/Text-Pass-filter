from PIL import Image
import json
import os
from shapely.geometry import Polygon
from tqdm import tqdm


class COCOConverter(object):
    def __init__(self, datasets, coco_label_file, info='msra', licenses='none', with_angle=False):
        self.datasets = datasets
        self.coco_label_file = coco_label_file

        self.data = {
            "info": info,
            "images": [],
            "annotations": [],
            "categories": [
                {
                    "id": 1,
                    "name": "text",
                    "supercategory": "foreground"
                }
            ],
            "licenses": licenses
        }

        self.with_angle = with_angle

    def add_image(self, id, width, height, filename):
        image = {
            "id": id,
            "width": width,
            "height": height,
            "file_name": filename
        }
        self.data['images'].append(image)

    def add_annotation(self, id, image_id, category_id, segmentation, area, bbox, iscrowd):
        annotation = {
            "id": id,
            "image_id": image_id,
            "category_id": category_id,
            "segmentation": segmentation,
            "area": area,
            "bbox": bbox,
            "iscrowd": iscrowd
        }
        self.data['annotations'].append(annotation)

    def convert(self):
        print('start converting...')
        ann_id = 0
        image_id = 0
        for dataset in self.datasets:
            print(f'converting {dataset.name}')

            if dataset.version == '2017':
                icdar_label_dir = dataset.label
                icdar_images_dir = dataset.images_dir
                label_files = os.listdir(icdar_label_dir)
                for label_file in tqdm(label_files):
                    label_file_path = os.path.join(icdar_label_dir, label_file)
                    skip = True
                    with open(label_file_path, 'r', encoding='utf8') as f:
                        for line in f:
                            line = line.strip()
                            iterms = line.split(',')
                            points = iterms[:8]
                            script = iterms[8]
                            if script != '###':
                                skip = False
                                segmentation = []
                                for point in points:
                                    segmentation.append(int(point))

                                xs = [int(points[i]) for i in range(0, len(points), 2)]
                                ys = [int(points[i]) for i in range(1, len(points), 2)]
                                xmin = min(xs)
                                xmax = max(xs)
                                ymin = min(ys)
                                ymax = max(ys)

                                width = xmax - xmin
                                height = ymax - ymin
                                bbox = [xmin, ymin, width, height]

                                points = [[xs[i], ys[i]] for i in range(len(xs))]
                                poly = Polygon(points)
                                area = round(poly.area, 2)

                                self.add_annotation(ann_id,
                                                    image_id,
                                                    category_id=1,  # text regions are foregrounds
                                                    segmentation=[segmentation],
                                                    area=area,
                                                    bbox=bbox,
                                                    iscrowd=0)
                                ann_id += 1

                    if not skip:
                        img_idx = label_file.replace('txt','jpg')
                        image_path = os.path.join(icdar_images_dir, img_idx)
                        im = Image.open(image_path + '.jpg')
                        width, height = im.size
                        self.add_image(image_id, width, height, img_idx)
                        image_id += 1

        with open(self.coco_label_file, 'w')as f:
            json.dump(self.data, f)
        # print(self.data)
        print(f'Done. {image_id + 1} images totally.')
