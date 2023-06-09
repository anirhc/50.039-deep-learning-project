import torch
import datetime
import data_preprocessing as dp
import model as m

def main():
    """
    This function trains, tests, and saves the parameters of deep neural network model.
    """
    torch.manual_seed(42)
    model = m.DeepNeuralNet(n_x = 164, n_h = [64, 32, 8], n_y = 1)
    train_loss_list, val_loss_list, train_acc, val_acc = model.trainer(dp.train_inputs_pt, dp.train_outputs_pt, dp.val_inputs_pt, dp.val_outputs_pt, N_max = 150, lr = 0.01, batch_size = 32)
    test_loss, test_acc = model.tester(dp.test_inputs_pt, dp.test_outputs_pt)
    torch.save(model.state_dict(), "weights/pytorch_model_{}.pt".format(datetime.datetime.now().strftime("%Y%m%d_%H%M%S")))

if __name__ == '__main__':
    main()