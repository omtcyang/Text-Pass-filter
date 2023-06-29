from Dataset import Dataset
from COCOConverter import COCOConverter

if __name__ == "__main__":
    msra = Dataset('MSRA-TD500', '2017', '/home/xiekaiyu/ocr/dataset/ICDAR2017MLT/train_gt', '/home/xiekaiyu/ocr/dataset/ICDAR2017MLT/train')
    datasets = [msra]
    coco_label_file = '/home/xiekaiyu/ocr/dataset/test/coco_labels.json'
    converter = COCOConverter(datasets, coco_label_file)
    converter.convert()
