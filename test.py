from sklearn.metrics import auc, precision_recall_curve
import numpy as np
import torch
import csv
import os

def test(dataloader, model, gt, args):
    with torch.no_grad():
        model.eval()
        pred = torch.zeros(0).to(args.device)
        vanilla_input = torch.zeros(0).to(args.device)
        trained_output = torch.zeros(0).to(args.device)
        for i, inputs in enumerate(dataloader):
            inputs = inputs.to(args.device)
            _, logits, = model(inputs, None)  # (bs, len)
            sig = logits
            sig = torch.sigmoid(sig)
            sig = torch.mean(sig, 0)
            pred = torch.cat((pred, sig))

        pred = list(pred.cpu().detach().numpy())
        precision, recall, th = precision_recall_curve(list(gt), np.repeat(pred, 16))
        pr_auc = auc(recall, precision)
        # precision, recall, th = precision_recall_curve(list(gt), np.repeat(pred2, 16))
        # pr_auc2 = auc(recall, precision)

        return pr_auc, 0.0

def test_single_video(dataloader, model, args):
    with torch.no_grad():
        model.eval()
        pred = torch.zeros(0).to(args.device)

        for i, inputs in enumerate(dataloader):
            inputs = inputs.to(args.device)
            _, logits, = model(inputs, None)
            sig = torch.sigmoid(logits)
            sig = torch.mean(sig, 0)
            pred = torch.cat((pred, sig))

        pred = list(pred.cpu().detach().numpy())
        pred_binary = [1 if pred_value[0] > 0.45 else 0 for pred_value in pred]

        video_duration = int(np.ceil(len(pred_binary) * 0.96)) # len(pred_binary) = video_duration / 0.96

        if any(pred == 1 for pred in pred_binary):
            message= "El video contiene violencia. "
            message_second = "Los intervalos con violencia son: "
            message_frames = "En un rango de [0-"+ str(len(pred_binary) - 1) +"] los frames con violencia son: "

            start_idx = None
            for i, pred in enumerate(pred_binary):
                if pred == 1:
                    if start_idx is None:
                        start_idx = i
                elif start_idx is not None:
                    message_frames += ("[" + str(start_idx) + " - " + str(i - 1) + "]" + ", ") if i-1 != start_idx else ("[" + str(start_idx) + "], ")
                    message_second += ("[" + parse_time(int(np.floor((start_idx + 1)* 0.96))) + " - " + parse_time(int(np.ceil(i * 0.96))) + "], ")
                    start_idx = None

            if start_idx is not None:
                message_frames += ("[" + str(start_idx) + " - " + str(len(pred_binary) - 1) + "]") if len(pred_binary) - 1 != start_idx else ("[" + str(start_idx) + "]")
                message_second += ("[" + parse_time(int(np.floor((start_idx + 1) * 0.96))) + " - " + parse_time(video_duration) + "]")
            else:
                message_frames = message_frames[:-2]              
                message_second = message_second[:-2]              

        else:
            message= "El video no contiene violencia."
            message_frames = ""            
            message_second = ""            

        if args.evaluate == 'true':
            # Create a list of dictionaries to store the data
            data = []
            data.append({
                'video_id': "IDVIDEO",
                'frame_number': pred_binary,
                "violence_label": "1" if any(pred == 1 for pred in pred_binary) else "0",
            })

            # Write the data to a CSV file
            csv_file = 'inference.csv'

            fieldnames = ['video_id', 'frame_number', 'violence_label']
            file_exists = os.path.isfile(csv_file)

            with open(csv_file, 'a', newline='') as file:
                writer = csv.DictWriter(file, fieldnames=fieldnames)
                if not file_exists:
                    writer.writeheader()
                writer.writerows(data)
            
        return message, message_second, message_frames

def parse_time(seconds):
    seconds = max(0, seconds)
    sec = seconds % 60
    if sec < 10:
        sec = "0" + str(sec)
    else:
        sec = str(sec)
    return str(seconds // 60) + ":" + sec