import os
import datetime
import torch
import transform
from my_dataset import VOC2012DataSet
from src.ssd_model import SSD300, Backbone
import train_utils.train_eval_utils as utils
from train_utils.coco_utils import get_coco_api_from_dataset
from tqdm import tqdm
import os

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import torchvision.models as modelss
from torchsummary import summary


def create_model(num_classes=3, device=torch.device('cpu')):
    backbone = Backbone()
    model = SSD300(backbone=backbone, num_classes=num_classes)
    pre_ssd_path = "./src/ssd.pt"
    if os.path.exists(pre_ssd_path) is False:
        raise FileNotFoundError("ssd.pt not find in {}".format(pre_ssd_path))
    pre_model_dict = torch.load(pre_ssd_path, map_location=device)
    pre_weights_dict = pre_model_dict["model"]

    del_conf_loc_dict = {}
    i=0
    for k, v in pre_weights_dict.items():
        i = i + 1
        split_key = k.split(".")
        if i <=6:
            split_key[1]="conv1"
        if i <=66 and i>=7:
            split_key[1]="conv2"
            split_key[2]="1"
        if i <=144 and i>=67:
            split_key[1]="conv3"
            split_key[2] = "0"
        if i <=258 and i>=145:
            split_key[1]="conv3"
            split_key[2] = "1"

        if "conf" in split_key:
            continue
        new_k=""
        for kk in split_key:
            new_k=new_k+"."+kk
        k=new_k[1:]
        del_conf_loc_dict.update({k: v})


    missing_keys, unexpected_keys = model.load_state_dict(del_conf_loc_dict, strict=False)
    if len(missing_keys) != 0 or len(unexpected_keys) != 0:
        print("missing_keys: ", missing_keys)
        print("unexpected_keys: ", unexpected_keys)

    return model.to(device)


def main(parser_data):

    device = torch.device(parser_data.device if torch.cuda.is_available() else "cpu")
    print("Using {} device training.".format(device.type))

    if not os.path.exists("save_weights"):
        os.mkdir("save_weights")

    results_file="results{}.txt".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    data_transform = {
        "train": transform.Compose([
                                    transform.Resize(),
                                    transform.ToTensor(),
                                    transform.RandomHorizontalFlip(),
                                    transform.Normalization(),
                                    transform.AssignGTtoDefaultBox()]),
        "val": transform.Compose([transform.Resize(),
                                  transform.ToTensor(),
                                  transform.Normalization()
                                  ])
    }

    VOC_root = parser_data.data_path
    if os.path.exists(os.path.join(VOC_root, "VOCdevkit")) is False:
        raise FileNotFoundError("VOCdevkit dose not in path:'{}'.".format(VOC_root))

    train_dataset = VOC2012DataSet(VOC_root, data_transform['train'], train_set='train.txt')
    batch_size = parser_data.batch_size
    assert batch_size > 1, "batch size must be greater than 1"
    drop_last = True if len(train_dataset) % batch_size == 1 else False
    nw=0
    print('Using %g dataloader workers' % nw)
    train_data_loader = torch.utils.data.DataLoader(train_dataset,
                                                    batch_size=batch_size,
                                                    shuffle=True,
                                                    num_workers=nw,
                                                    collate_fn=train_dataset.collate_fn,
                                                    drop_last=drop_last,
                                                    pin_memory=True)

    val_dataset = VOC2012DataSet(VOC_root, data_transform['val'], train_set='val.txt')
    val_data_loader = torch.utils.data.DataLoader(val_dataset,
                                                  batch_size=batch_size,
                                                  shuffle=False,
                                                  num_workers=nw,
                                                  collate_fn=train_dataset.collate_fn,
                                                  pin_memory=True
                                                  )

    model = create_model(num_classes=args.num_classes+1, device=device)
    model.to(device)

    # define optimizer

    params = [p for p in model.parameters() if p.requires_grad]


    optimizer = torch.optim.SGD(params, lr=parser_data.lr,
                                momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                    step_size=parser_data.step_size,
                                                    gamma=0.3)

    if parser_data.resume != "":
        checkpoint = torch.load(parser_data.resume)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        parser_data.start_epoch = checkpoint['epoch'] + 1
        print("the training process from epoch{}...".format(parser_data.start_epoch))

    train_loss = []
    learning_rate = []
    val_map = []
    val_data = get_coco_api_from_dataset(val_data_loader.dataset)
    for epoch in tqdm(range(parser_data.start_epoch, parser_data.epochs)):
        mean_loss,lr=utils.train_one_epoch(model=model, optimizer=optimizer,
                              data_loader=train_data_loader,
                              device=device, epoch=epoch,
                              print_freq=50)
        train_loss.append(mean_loss.item())
        learning_rate.append(lr)

        lr_scheduler.step()
        coco_info,ssims,psnrs=utils.evaluate(model=model, data_loader=val_data_loader,
                           device=device, data_set=val_data)
        recinfo="ssim:  "+str(ssims)+"  psnr: "+str(psnrs)


        with open(results_file,"a") as f:
            result_info = [str(round(i, 4)) for i in coco_info + [mean_loss.item()]] + [str(round(lr, 6))]
            txt="epoch:{} {} {}".format(epoch,'  '.join(result_info),recinfo)
            f.write(txt+"\n")

        val_map.append(coco_info[1])  #pascal mAP
        if epoch>=10:
            save_files = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch}
            torch.save(save_files, "./save_weights/ssd300-{}.pth".format(epoch))

    # plot loss and lr curve
    if len(train_loss) != 0 and len(learning_rate) != 0:
        from plot_curve import plot_loss_and_lr
        plot_loss_and_lr(train_loss, learning_rate)

    # plot mAP curve
    if len(val_map) != 0:
        from plot_curve import plot_map
        plot_map(val_map)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description=__doc__)

    parser.add_argument('--device', default='cuda:0', help='device')
    parser.add_argument('--num_classes', default=2, type=int,help='num_classes')
    parser.add_argument('--data-path', default=r'./', help='dataset')
    parser.add_argument('--output-dir', default='./save_weights', help='path where to save')
    parser.add_argument('--resume', default='', type=str, help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, help='start epoch')
    parser.add_argument('--epochs', default=50, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--batch_size', default=8, type=int, metavar='N',
                        help='batch size when training.')
    parser.add_argument('--lr', default=0.0005, type=int, metavar='N',
                        help='lr.')
    parser.add_argument('--step_size', default=15, type=int, metavar='N',
                        help='step_size')


    args = parser.parse_args()
    print(args)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    main(args)
