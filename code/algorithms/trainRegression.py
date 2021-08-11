import numpy as np
def train_regressor(train_loader, loss_fn, learning_rate, max_epochs, input_size, K, device, P, torch, CLN):
    lossess = []
    lambda1 = 1e-5
    lambda2 = 5e-4
    cln = CLN(input_size, K, device, P, p=0).to(device)
    optimizer = torch.optim.Adam(list(cln.parameters()), lr=learning_rate)
    criterion = loss_fn
    cln.train()
    optimizer.zero_grad()
    emp_loss = []
    for epoch in range(max_epochs):
        total_epoch_loss = 0
        for batch_idx, (inps, tgts) in enumerate(train_loader):
            tgts = tgts.reshape((-1, 1)).to(device)
            out = cln(inps)
            loss = criterion(out, tgts)
            loss = loss + lambda1*torch.linalg.norm(cln.G1, 1) + lambda2*torch.linalg.norm(cln.G2, 1)
            if epoch >= max_epochs // 2:
                emp_loss.append(loss)
            total_epoch_loss += loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # lossess.append(total_epoch_loss/training_size)
        lossess.append(total_epoch_loss.item() / len(train_loader.dataset))
        print("total epoch loss: ", total_epoch_loss)
        if epoch % 5 == 0:
            print('epoch {}, loss {}'.format(epoch, loss.item()))
            # print('cln or weights grad:', cln.G1.grad.data.cpu().numpy().flatten().round(2))
    emp_loss = np.array([e.detach().numpy() for e in emp_loss])
    if emp_loss.all() < 1e-5:
        print("Zero Error")
    print(len(emp_loss))
    print(len(emp_loss[emp_loss < 0.1]))
    print("P(loss < 0.1) = ", len(emp_loss[emp_loss < 0.1]) / len(emp_loss))
    return cln, lossess
