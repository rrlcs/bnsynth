import torch
import torch.nn as nn

# weight init
def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

def train_classifier(train_loader, loss_fn, learning_rate, max_epochs, input_size, K, device, name, P, torch, CLN, util, spec):
    lossess = []
    lambda1 = 1e-4
    lambda2 = 1e-6
    cln = CLN(input_size, K, device, name, P, p=0).to(device)
    cln.apply(init_weights)
    optimizer = torch.optim.Adam(list(cln.parameters()), lr=learning_rate)
    criterion = loss_fn
    cln.train()
    optimizer.zero_grad()
    for epoch in range(max_epochs):
        total_epoch_loss = 0
        for batch_idx, (inps, tgts) in enumerate(train_loader):
            tgts = tgts.reshape((-1)).to(device)
            out = cln(inps)
            inpOut = torch.cat((inps, out), dim=1)
            if spec == 1:
                fOut = util.spec1(inpOut.T, name).to(device)
            elif spec == 2:
                fOut = util.spec2(inpOut.T, name).to(device)
            elif spec == 3:
                fOut = util.spec3(inpOut.T, name).to(device)
            elif spec == 4:
                fOut = util.spec4(inpOut.T, name).to(device)
            elif spec == 5:
                fOut = util.spec5(inpOut.T, name).to(device)
            loss = criterion(fOut, tgts)
            loss = loss + lambda1*torch.linalg.norm(cln.G1, 1) + lambda2*torch.linalg.norm(cln.G2, 1)
            total_epoch_loss += loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # average_loss = (total_epoch_loss/training_size) * 1000
        # lossess.append(average_loss.item())
        lossess.append(total_epoch_loss.item())
        print("total epoch loss: ", total_epoch_loss)
        if epoch % 5 == 0:
            print('epoch {}, loss {}'.format(epoch, loss.item()))
            # print('cln or weights grad:', cln.G1.grad.data.cpu().numpy().flatten().round(2))
    return cln, lossess

# loss_fn = nn.BCEWithLogitsLoss()
# cln, lossess = train_classifier(train_loader, loss_fn)
# torch.save(cln.state_dict(), "classifier")

# f = open("lossess", "w")
# lossess = np.array(lossess)
# lossess.tofile(f, sep=",", format="%s")