from torch.utils.data import DataLoader
import torch
import numpy as np
from model import Model
from dataset import Dataset
from test import test_single_video
import option
import time
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def test():
    print('perform testing...')
    args = option.parser.parse_args()
    args.device = 'cuda:' + str(args.cuda) if torch.cuda.is_available() else 'cpu'
    # device = torch.device("cuda")

    test_loader = DataLoader(Dataset(args, test_mode=True),
                              batch_size=5, shuffle=False,
                              num_workers=args.workers, pin_memory=True)
    model = Model(args)
    model = model.to(args.device)
    model_dict = model.load_state_dict(
        {k.replace('module.', ''): v for k, v in torch.load('./ckpt/pretrained.pkl', map_location=torch.device('cpu')).items()})
    st = time.time()

    message, message_second, message_frames  = test_single_video(test_loader, model, args)
    time_elapsed = time.time() - st
    print(message + message_frames + "\n"+ message + message_second)
    print('Test complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    return message + message_frames
if __name__ == '__main__':
    test()
