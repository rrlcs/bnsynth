from code.model.gcln import GCLN
import torch


def train_regressor(
    train_loader, validation_loader, loss_fn, learning_rate, 
    max_epochs, input_size, num_of_outputs, output_var_idx, K, 
    device, P, flag, checkpoint=None
    ):
    
    train_loss = []
    valid_loss = []
    lambda1 = 1e-3
    lambda2 = 1e-3
    gcln = GCLN(input_size, num_of_outputs, K, device, P).to(device)
    optimizer = torch.optim.Adam(list(gcln.parameters()), lr=learning_rate)
    criterion = loss_fn
    for epoch in range(max_epochs):
        gcln.train()
        optimizer.zero_grad()
        train_epoch_loss = 0
        for batch_idx, (inps, tgts) in enumerate(train_loader):
            tgts = tgts.reshape((tgts.size(0), -1)).to(device)
            out = gcln(inps).to(device)
            l = []
            for i in range(num_of_outputs):
                l.append(criterion(out[:,i], tgts[:,i]))
            t_loss = sum(l)
            
            t_loss = torch.sqrt(criterion(out, tgts))
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
            l = []
            for i in range(num_of_outputs):
                l.append(criterion(out[:,i], tgts[:,i]))
            v_loss = sum(l)
            
            v_loss = criterion(out, tgts)
            v_loss = v_loss + lambda1*torch.linalg.norm(gcln.G1, 1) + \
                lambda2*torch.linalg.norm(gcln.G2, 1)
            v_loss = v_loss + lambda1*torch.linalg.norm(gcln.G1, 2) + \
                lambda2*torch.linalg.norm(gcln.G2, 2)
            
            valid_epoch_loss += v_loss.item()*inps.size(0)
        valid_loss.append(valid_epoch_loss / len(validation_loader.sampler))
        
        if epoch % 5 == 0:
            print('epoch {}, train loss {}, valid loss {}'.format(
                epoch, round(t_loss.item(), 4), round(v_loss.item(), 4))
                )
        torch.save({
            'epoch': epoch,
            'model_state_dict': gcln.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': t_loss,
            }, "saved_model")

    return gcln, train_loss, valid_loss
