import os, time
from operator import add
import numpy as np
from glob import glob
import cv2
from tqdm import tqdm
import imageio
import torch
from sklearn.metrics import accuracy_score, f1_score, jaccard_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

from model3 import USNet
from utils import create_dir, seeding

def calculate_sensitivity_specificity(y_true, y_pred):
    # Calculate confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true.flatten(), y_pred.flatten()).ravel()

    # Sensitivity (Recall) TPR = TP / (TP + FN)
    sensitivity = tp / (tp + fn)
    
    # Specificity SPC = TN / (FP + TN)
    specificity = tn / (fp + tn)

    return sensitivity, specificity
def calculate_metrics(y_true, y_pred):
    """ Ground truth """
    y_true = y_true.cpu().numpy()
    y_true = y_true > 0.5
    y_true = y_true.astype(np.uint8)
    y_true = y_true.reshape(-1)

    """ Prediction """
    y_pred = y_pred.cpu().numpy()
    y_pred = y_pred > 0.5
    y_pred = y_pred.astype(np.uint8)
    y_pred = y_pred.reshape(-1)

    score_jaccard = jaccard_score(y_true, y_pred)
    score_f1 = f1_score(y_true, y_pred)
    score_recall = recall_score(y_true, y_pred)
    score_precision = precision_score(y_true, y_pred)
    score_acc = accuracy_score(y_true, y_pred)
    #score_sensitivity=sensitive_score(y_true,y_pred)
    #score_specificity=specificity_score(y_true,y_pred)
    score_sensitivity,score_specificity=calculate_sensitivity_specificity(y_true,y_pred)
    return [score_jaccard, score_f1, score_recall, score_precision, score_acc,score_sensitivity,score_specificity]

def mask_parse(mask):
    mask = np.expand_dims(mask, axis=-1)    ## (512, 512, 1)
    mask = np.concatenate([mask, mask, mask], axis=-1)  ## (512, 512, 3)
    return mask

if __name__ == "__main__":
    """ Seeding """
    seeding(42)

    """ Folders """
    create_dir("results3")

    """ Load dataset """
    test_x = sorted(glob("D:/clgproj/Progs/new_data/test/image/*"))
    test_y = sorted(glob("D:/clgproj/Progs/new_data/test/mask/*"))

    """ Hyperparameters """
    H = 512
    W = 512
    size = (W, H)
    checkpoint_path = "files/checkpoint3.pth"

    """ Load the checkpoint """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = USNet()
    model = model.to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    metrics_score = [0.0, 0.0, 0.0, 0.0, 0.0,0.0,0.0]
    time_taken = []
    A=[]
    R=[]
    P=[]
    S=[]

    for i, (x, y) in tqdm(enumerate(zip(test_x, test_y)), total=len(test_x)):
        """ Extract the name """
        name=list(os.path.split(x))[-1].split('.')[0]
        print(name)
        """ Reading image """
        image = cv2.imread(x, cv2.IMREAD_COLOR) ## (512, 512, 3)
        # image = cv2.resize(image, size)
        x = np.transpose(image, (2, 0, 1))      ## (3, 512, 512)
        x = x/255.0
        x = np.expand_dims(x, axis=0)           ## (1, 3, 512, 512)
        x = x.astype(np.float32)
        x = torch.from_numpy(x)
        x = x.to(device)

        """ Reading mask """
        mask = cv2.imread(y, cv2.IMREAD_GRAYSCALE)  ## (512, 512)
        ## mask = cv2.resize(mask, size)
        y = np.expand_dims(mask, axis=0)            ## (1, 512, 512)
        y = y/255.0
        y = np.expand_dims(y, axis=0)               ## (1, 1, 512, 512)  
        y = y.astype(np.float32)
        y = torch.from_numpy(y)
        y = y.to(device)

        with torch.no_grad():
            """ Prediction and Calculating FPS """
            start_time = time.time()
            pred_y = model(x)
            #pred_y = torch.sigmoid(pred_y)
            total_time = time.time() - start_time
            time_taken.append(total_time)

            score = calculate_metrics(y, pred_y)
            metrics_score = list(map(add, metrics_score, score))
            pred_y = pred_y[0].cpu().numpy()        ## (1, 512, 512)
            pred_y = np.squeeze(pred_y, axis=0)     ## (512, 512)
            pred_y = pred_y > 0.5
            pred_y = np.array(pred_y, dtype=np.uint8)
            A.append(score[4])
            R.append(score[2])
            P.append(score[3])
            S.append(score[6])

        """ Saving masks """
        ori_mask = mask_parse(mask)
        pred_y = mask_parse(pred_y)
        line = np.ones((size[1], 10, 3)) * 128

        cat_images = np.concatenate(
            [image, line, ori_mask, line, pred_y * 255], axis=1
        )
        cv2.imwrite(f"results3/{name}.png", cat_images)

    if len(test_x)>0:
        jaccard = metrics_score[0]/len(test_x)
        f1 = metrics_score[1]/len(test_x)
        recall = metrics_score[2]/len(test_x)
        precision = metrics_score[3]/len(test_x)
        acc = metrics_score[4]/len(test_x)
        sensitivity=metrics_score[5]/len(test_x)
        specificity=metrics_score[6]/len(test_x)
    else:
        jaccard = metrics_score[0]
        f1 = metrics_score[1]
        recall = metrics_score[2]
        precision = metrics_score[3]
        acc = metrics_score[4]
        sensitivity=metrics_score[5]
        specificity=metrics_score[6]
    print(f"F1: {f1:1.4f} - Sensitivity: {sensitivity:1.4f} - Specificity: {specificity:1.4f} - Precision: {precision:1.4f} - Acc: {acc:1.4f}")
plt.figure(figsize=(10, 5))

# Plot Accuracy vs Recall
plt.subplot(2, 2, 1)
plt.plot(A,marker='.', linestyle='--')
plt.xlabel('Images')
plt.ylabel('%')
plt.title('Accuracy')

# Plot Precision vs Recall
plt.subplot(2, 2, 2)
plt.plot(P, marker='.', linestyle='--')
plt.xlabel('Images')
plt.ylabel('%')
plt.title('Precision')
plt.subplot(2, 2, 3)
plt.plot(R, marker='.', linestyle='--')
plt.xlabel('Images')
plt.ylabel('%')
plt.title('Recall')
plt.subplot(2, 2, 4)
plt.plot(S, marker='.', linestyle='--')
plt.xlabel('Images')
plt.ylabel('%')
plt.title('Specificity')
plt.tight_layout()
plt.show()
'''fps=0
    if len(time_taken) > 0 and not np.isnan(time_taken).any():
        fps = 1/np.mean(time_taken)
    print("FPS: ", fps)'''