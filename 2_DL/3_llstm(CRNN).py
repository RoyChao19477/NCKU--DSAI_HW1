import sys
from tkinter import E
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error

from read_df import df_lstm
from models import LLSTM_05

if torch.cuda.is_available():
    print("GPU: ", torch.cuda.get_device_name())
    device = 'cuda'
else:
    device = 'cpu'
    #sys.exit("I want a cup of milk tea. Thanks!")

train_dataloader = DataLoader(
        df_lstm(train=True),
        batch_size = 1,
        # batch_size = 8,
        shuffle = False,
        num_workers=0
        )

test_dataloader = DataLoader(
        df_lstm(train=False),
        batch_size = 1,
        shuffle = False,
        num_workers=0
        )

# model = AE_01()
model = LLSTM_05(4)
model.to(device)
model.double()

# optimizer = optim.Adam(model.parameters(), lr=0.00001, weight_decay=0)
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0005)
#judge = torch.nn.MSELoss()
judge = torch.nn.L1Loss()

min_loss = 1000000000
try:
    for i in range(100):
        model.train()
        total_loss = 0.0
        for x, y in train_dataloader:
            if device=='cpu':
                pred = model(x)
                loss = judge(pred, y)
            else:
                pred = model(x.cuda())
                loss = judge(pred, y.cuda())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        print("Training: ", i, " traing loss: ", total_loss)
        
        
        model.eval()
        pred_list = []
        test_loss = 0
        for x, y in test_dataloader:
            if device=='cpu':
                pred = model(x)
            else:
                pred = model(x.cuda())
            pred_list.append(pred.detach().cpu().numpy()[0])

            if device=='cpu':
                loss = judge(pred, y)
            else:
                loss = judge(pred, y.cuda())
            
            test_loss += loss.item()
    
        # print(np.asarray( pred_list ))
        # print("pred_list: ", pred_list)
        mse = mean_squared_error( pred_list, [3032,2301,3167,2963,2549,3464,4700,3346,3087,3111]) / 10
        mae = mean_absolute_error( pred_list, [3032,2301,3167,2963,2549,3464,4700,3346,3087,3111]) / 10
        test_loss = mae

        if min_loss > test_loss:
            torch.save({ 
                'model_state_dict': model.state_dict(), 
                'optimizer_state_dict': optimizer.state_dict()}, 'src/results/9_CRNN_05_epoch100_7days.pt')
            #np.savetxt("src/results/9_CRNN_05_pred_epoch100_7days.csv", np.asarray( pred_list ), delimiter=',')
            #print(f"Predict: {pred_list}")

            print(f"    MSE: {mse}, MAE: {mae}")
            min_loss = test_loss
            
except Exception as e:
    exception = e
    raise

"""
model.eval()
pred_list = []
for x, y in test_dataloader:
    # pred = model(x.cuda())[0][0]
    pred, hid = model(x.cuda())[0][0][0]
    pred_list.append(pred.detach().cpu())
print(np.asarray(pred_list))
np.savetxt("result_AE_01_1000epoch.csv", np.asarray( pred_list ), delimiter=',')
"""
