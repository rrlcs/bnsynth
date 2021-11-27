import torch
import torch.nn as nn

# weight init
def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

def train_classifier(train_loader, loss_fn, learning_rate, max_epochs, input_size, num_of_ouputs, K, device, P, torch, GCLN):
    lossess = []
    lambda1 = 1e-5
    lambda2 = 1e-5
    gcln = GCLN(input_size, num_of_ouputs, K, device, P).to(device)
    gcln.apply(init_weights)
    optimizer = torch.optim.Adam(list(gcln.parameters()), lr=learning_rate)
    criterion = loss_fn
    gcln.train()
    optimizer.zero_grad()
    for epoch in range(max_epochs):
        total_epoch_loss = 0
        for batch_idx, (inps, tgts) in enumerate(train_loader):
            tgts = tgts.reshape((-1)).to(device)
            out = gcln(inps).squeeze(-1)
            loss = criterion(out, tgts)
            loss = loss + lambda1*torch.linalg.norm(gcln.G1, 1) + lambda2*torch.linalg.norm(gcln.G2, 1)
            total_epoch_loss += loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # average_loss = (total_epoch_loss/training_size) * 1000
        # lossess.append(average_loss.item())
        lossess.append(total_epoch_loss.item() / len(train_loader.dataset))
        print("total epoch loss: ", total_epoch_loss)
        if epoch % 5 == 0:
            print('epoch {}, loss {}'.format(epoch, loss.item()))
            # print('cln or weights grad:', cln.G1.grad.data.cpu().numpy().flatten().round(2))
    return gcln, lossess
