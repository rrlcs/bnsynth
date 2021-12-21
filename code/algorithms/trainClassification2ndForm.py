import torch
import torch.nn as nn

# import func_spec

# weight init
def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

def train_classifier(
    train_loader,
    validation_loader, 
    loss_fn, 
    learning_rate, 
    max_epochs, 
    input_size,
    num_of_outputs, 
    K, 
    device, 
    P, 
    torch, 
    GCLN, 
    util, 
    func_spec
    ):

    train_loss = []
    valid_loss = []
    lambda1 = 1e-2
    lambda2 = 1e-2

    gcln = GCLN(input_size, num_of_outputs, K, device, P).to(device)
    gcln.apply(init_weights)
    
    optimizer = torch.optim.Adam(list(gcln.parameters()), lr=learning_rate)
    criterion = loss_fn
    
    gcln.train()
    optimizer.zero_grad()
    for epoch in range(max_epochs):
        train_epoch_loss = 0
        for batch_idx, (inps, tgts) in enumerate(train_loader):
            tgts = tgts.reshape((tgts.size(0), -1)).to(device)
            out = gcln(inps)
            inpOut = torch.cat((inps, out), dim=1)
            fOut = func_spec.F(inpOut.T, util)
            fOut = fOut.reshape((tgts.size(0), -1)).to(device)
            t_loss = criterion(fOut, tgts)
            t_loss = t_loss + lambda1*torch.linalg.norm(gcln.G1, 1) + \
                lambda2*torch.linalg.norm(gcln.G2, 1)
            t_loss = t_loss + lambda1*torch.linalg.norm(gcln.G1, 2) + \
                lambda2*torch.linalg.norm(gcln.G2, 2)
            
            optimizer.zero_grad()
            t_loss.backward()
            optimizer.step()
            train_epoch_loss += t_loss.item()*inps.size(0)
        train_loss.append(train_epoch_loss / len(train_loader.sampler))

        gcln.eval()
        valid_epoch_loss = 0
        for batch_idx, (inps, tgts) in enumerate(validation_loader):
            tgts = tgts.reshape((tgts.size(0), -1)).to(device)
            out = gcln(inps).to(device)
            inpOut = torch.cat((inps, out), dim=1)
            fOut = func_spec.F(inpOut.T, util)
            fOut = fOut.reshape((tgts.size(0), -1)).to(device)
            v_loss = criterion(fOut, tgts)
            v_loss = v_loss + lambda1*torch.linalg.norm(gcln.G1, 1) + \
                lambda2*torch.linalg.norm(gcln.G2, 1)
            v_loss = v_loss + lambda1*torch.linalg.norm(gcln.G1, 2) + \
                lambda2*torch.linalg.norm(gcln.G2, 2)

            valid_epoch_loss += v_loss.item()*inps.size(0)
        valid_loss.append(valid_epoch_loss / len(validation_loader.sampler))

        if epoch % 5 == 0:
            print('epoch {}, train loss {}, valid loss {}'.format(
                epoch, t_loss.item(), v_loss.item()))
            # print('cln or weights grad:', cln.G1.grad.data.cpu().numpy().flatten().round(2))
    return gcln, train_loss, valid_loss
