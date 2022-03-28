import sys
from tabnanny import verbose
from tkinter import E
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error

from read_df import df_lstm_v2
from models import LSTM_05
from models import L_05
from models import LM
from models import Transformer_03
from models import DNN_01

if torch.cuda.is_available():
    print("GPU: ", torch.cuda.get_device_name())
    device = 'cuda'
else:
    device = 'cpu'
    #sys.exit("I want a cup of milk tea. Thanks!")

train_dataloader = DataLoader(
        df_lstm_v2(train=True),
        batch_size = 1,
        # batch_size = 8,
        shuffle = False,
        num_workers=0
        )

test_dataloader = DataLoader(
        df_lstm_v2(train=False),
        batch_size = 1,
        shuffle = False,
        num_workers=0
        )

show_dataloader = DataLoader(
        df_lstm_v2(train=True),
        batch_size = 1,
        # batch_size = 8,
        shuffle = False,
        num_workers=0
        )

# model = AE_01()
model = LSTM_05(4)
#model = L_05(4)
#model = Transformer_03()
#model = DNN_01(4)
model = LM() 
model.to(device)
model.double()

# optimizer = optim.Adam(model.parameters(), lr=0.00001, weight_decay=0)
optimizer = optim.Adam(model.parameters())
#scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=10, gamma=0.5, verbose=True)
#judge = torch.nn.MSELoss()
judge = torch.nn.L1Loss()

min_loss = 1000000000
t_best = 100000000
try:
    for i in range(1000):
        model.train()
        total_loss = 0.0
        """
        if i > 100:
            for g in optimizer.param_groups:
                g['lr'] = 0.0001
        elif i > 15:
            for g in optimizer.param_groups:
                g['lr'] = 0.001
        elif i > 5:
            for g in optimizer.param_groups:
                g['lr'] = 0.01
        """

        for x, y in train_dataloader:
            y = y.double()
            #model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size).double(),
            #           torch.zeros(1, 1, model.hidden_layer_size).double())

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
            #model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size).double(),
            #            torch.zeros(1, 1, model.hidden_layer_size).double())

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

        if min_loss > test_loss or i > 90:
            torch.save({ 
                'model_state_dict': model.state_dict(), 
                'optimizer_state_dict': optimizer.state_dict()}, 'src/results/7_LSTM_05_epoch50_7days.pt')
            # np.savetxt("src/results/7_LSTM_05_pred_epoch50_7days.csv", np.asarray( pred_list ), delimiter=',')
            # print(f"Predict: {pred_list}")
            
            train_list = []
            model.eval()
            for x, y in show_dataloader:
                #model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size).double(),
                #        torch.zeros(1, 1, model.hidden_layer_size).double())

                if device=='cpu':
                    pred = model(x)
                else:
                    pred = model(x.cuda())
                train_list.append(pred.detach().cpu().numpy()[0])
                print(pred)
            np.savetxt("src/training_best/7_LSTM_05_train_epoch50_7days.csv", np.asarray( train_list ), delimiter=',')

            np.savetxt("src/training_best/7_LSTM_05_train_epoch50_7days_pred.csv", np.asarray( pred_list ), delimiter=',')
            

            print(f"    MSE: {mse}, MAE: {mae}")
            min_loss = test_loss
        #scheduler.step()
            
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
