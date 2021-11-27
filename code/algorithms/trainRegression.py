from code.model.gcln import GCLN
import numpy as np
import torch


def train_regressor(train_loader, validation_loader, loss_fn, learning_rate, max_epochs, input_size, num_of_outputs, output_var_idx, K, device, P, flag, checkpoint=None):
    train_loss = []
    valid_loss = []
    lambda1 = 1e-3
    lambda2 = 1e-3
    from code.model.gcln import GCLN
    device='cpu'
    gcln = GCLN(input_size, num_of_outputs, K, device, P).to(device)
    # print("init: ", list(gcln.G1))
    optimizer = torch.optim.Adam(list(gcln.parameters()), lr=learning_rate)

    # if not flag:
        #  checkpoint = torch.load("saved_model")
         # gcln.load_state_dict(checkpoint['model_state_dict'])
         # gcln.cpu()
        #  optimizer = torch.optim.Adam(list(gcln.parameters()), lr=learning_rate)
        # optimizer.load_state_dict(torch.load('optimizer'))
        # for state in optimizer.state.values():
        #     for k, v in state.items():
        #         if torch.is_tensor(v):
        #             state[k] = v.cpu()
    criterion = loss_fn
    emp_loss = []
    for epoch in range(max_epochs):
        gcln.train()
        optimizer.zero_grad()
        train_epoch_loss = 0
        for batch_idx, (inps, tgts) in enumerate(train_loader):
            tgts = tgts.reshape((tgts.size(0), -1))#.to(device)
            out = gcln(inps)#.reshape((-1, num_of_outputs)).to(device)
            l = []
            for i in range(num_of_outputs):
                l.append(criterion(out[:,i], tgts[:,i]))
            t_loss = sum(l)
            
            t_loss = criterion(out, tgts)
            t_loss = t_loss + lambda1*torch.linalg.norm(gcln.G1, 1) + lambda2*torch.linalg.norm(gcln.G2, 1)
            # t_loss = t_loss + lambda1*torch.linalg.norm(gcln.G1, 2) + lambda2*torch.linalg.norm(gcln.G2, 2)

            # if not flag:
                #  loss = checkpoint['loss']
                #  flag = 1

            if epoch >= max_epochs // 2:
                emp_loss.append(t_loss)
            train_epoch_loss += t_loss
            optimizer.zero_grad()
            t_loss.backward()
            optimizer.step()
        train_loss.append(train_epoch_loss.item() / len(train_loader.dataset))
        print("total training epoch loss: ", train_epoch_loss)

        valid_epoch_loss = 0
        gcln.eval()
        for batch_idx, (inps, tgts) in enumerate(validation_loader):
            tgts = tgts.reshape((tgts.size(0), -1))#.to(device)
            out = gcln(inps)#.reshape((-1, num_of_outputs)).to(device)
            l = []
            for i in range(num_of_outputs):
                l.append(criterion(out[:,i], tgts[:,i]))
            v_loss = sum(l)
            
            v_loss = criterion(out, tgts)
            v_loss = v_loss + lambda1*torch.linalg.norm(gcln.G1, 1) + lambda2*torch.linalg.norm(gcln.G2, 1)
            v_loss = v_loss + lambda1*torch.linalg.norm(gcln.G1, 2) + lambda2*torch.linalg.norm(gcln.G2, 2)

            # if not flag:
                #  loss = checkpoint['loss']
                #  flag = 1

            # if epoch >= max_epochs // 2:
            #     emp_loss.append(v_loss)
            valid_epoch_loss += v_loss
        valid_loss.append(valid_epoch_loss.item() / len(validation_loader.dataset))
        print("total validation epoch loss: ", valid_epoch_loss)
        if epoch % 5 == 0:
            print('epoch {}, train loss {}, valid loss {}'.format(epoch, t_loss.item(), v_loss.item()))
        torch.save({
            'epoch': epoch,
            'model_state_dict': gcln.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': t_loss,
            }, "saved_model")
            # print('cln or weights grad:', cln.G1.grad.data.cpu().numpy().flatten().round(2))
    emp_loss = np.array([e.detach().numpy() for e in emp_loss])
    if emp_loss.all() < 1e-5:
        print("Zero Error")
    print(len(emp_loss))
    print(len(emp_loss[emp_loss < 0.1]))
    print("P(loss < 0.1) = ", len(emp_loss[emp_loss < 0.1]) / len(emp_loss))

    
    # torch.save(optimizer.state_dict(), "optimizer")
    return gcln, train_loss, valid_loss
