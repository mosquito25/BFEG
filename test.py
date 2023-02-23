import os
import torch
import transform
from my_dataset import VOC2012DataSet
from src.ssd_model import SSD300, Backbone
import train_utils.train_eval_utils as utils
from train_utils.coco_utils import get_coco_api_from_dataset
import json
def create_model(num_classes):
    backbone = Backbone()
    model = SSD300(backbone=backbone, num_classes=num_classes)
    return model

def main(parser_data):
    device = torch.device(parser_data.device if torch.cuda.is_available() else "cpu")
    print("Using {} device training.".format(device.type))
    if not os.path.exists("save_weights"):
        os.mkdir("save_weights")
    data_transform = {"test": transform.Compose([transform.Resize(),transform.ToTensor()])}
    VOC_root = parser_data.data_path
    if os.path.exists(os.path.join(VOC_root, "VOCdevkit")) is False:
        raise FileNotFoundError("VOCdevkit dose not in path:'{}'.".format(VOC_root))
    test_dataset = VOC2012DataSet(VOC_root, data_transform['test'], train_set='test.txt')
    batch_size = parser_data.batch_size
    assert batch_size > 1, "batch size must be greater than 1"
    drop_last = True if len(test_dataset) % batch_size == 1 else False
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using %g dataloader workers' % nw)
    test_data_loader = torch.utils.data.DataLoader(test_dataset,
                                                  batch_size=batch_size,
                                                  shuffle=False,
                                                  num_workers=nw,
                                                  collate_fn=test_dataset.collate_fn)

    model = create_model(num_classes=3)

    train_weights = ""
    train_weights_dict = torch.load(train_weights, map_location=device)['model']
    model.load_state_dict(train_weights_dict, strict=False)
    model.to(device)



    val_data = get_coco_api_from_dataset(test_data_loader.dataset)
    model.eval()
    coco_info,ssims,psnrs=utils.evaluate(model=model, data_loader=test_data_loader,
                       device=device, data_set=val_data)
    print(ssims,psnrs)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description=__doc__)

    parser.add_argument('--device', default='', help='device')
    parser.add_argument('--data-path', default='', help='dataset')
    parser.add_argument('--batch_size', default=8, type=int, metavar='N',
                        help='batch size when training.')
    args = parser.parse_args()
    main(args)


































