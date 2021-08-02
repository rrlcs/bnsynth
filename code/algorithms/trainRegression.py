import numpy as np
import wandb
import random
wandb.login()

def train_regressor(train_loader, loss_fn, learning_rate, max_epochs, input_size, K, device, name, P, torch, CLN):

    # Set up your default hyperparameters
    sweep_config = {
    'method': 'random'
    }
    metric = {
    'name': 'loss',
    'goal': 'minimize'   
    }

    sweep_config['metric'] = metric
    parameters_dict = {
        'optimizer': {
            'values': ['adam', 'sgd']
            },
        'learning_rate': {
            'values': [0.01, 0.001, 0.0001]
            },
        'epochs': {
            'values': [20, 30, 50]
            },
        'l1': {
            'values': [1e-3, 1e-4, 5e-4, 1e-5, 1e-6, 1e-7]
            },
        'l2': {
            'values': [1e-3, 5e-3, 1e-4, 1e-5, 1e-6]
            },
        'K': {
            'values': [5, 10, 15, 20]
            },
        }
    # hyperparameter = dict(
    #     learning_rate=[0.01, 0.001, 0.0001],
    #     optimizer=["adam", "sgd"],
    #     epochs=[30, 50, 75, 100, 150, 200],
    #     l1 = [1e-3, 1e-4, 5e-4, 1e-5, 1e-6, 1e-7],
    #     l2 = [1e-3, 5e-3, 1e-4, 1e-5, 1e-6],
    #     K = [5, 10, 15, 20, 30, 50]
    #     )
    sweep_config['parameters'] = parameters_dict
    sweep_id = wandb.sweep(sweep_config, project="bfs1")
    
    # wandb.agent(sweep_id, train, count=5)
    def build_optimizer(network, optimizer, learning_rate):
        if optimizer == "sgd":
            optimizer = torch.optim.SGD(network.parameters(),
                                lr=learning_rate, momentum=0.9)
        elif optimizer == "adam":
            optimizer = torch.optim.Adam(network.parameters(),
                                lr=learning_rate)
        return optimizer

    # Initialize a new wandb run
    with wandb.init(config=sweep_config):
        config = wandb.config
        print(config.parameters['K']['values'])
        lossess = []
        K = random.choice(config.parameters['K']['values'])
        lambda1 = random.choice(config.parameters['l1']['values'])
        lambda2 = random.choice(config.parameters['l2']['values'])
        cln = CLN(input_size, K, device, name, P, p=0).to(device)
        # optimizer = torch.optim.Adam(list(cln.parameters()), lr=learning_rate)
        optimizer = build_optimizer(cln, random.choice(config.parameters['optimizer']['values']), random.choice(config.parameters['learning_rate']['values']))
        criterion = loss_fn
        cln.train()
        optimizer.zero_grad()
        emp_loss = []
        # print(config['epochs']['values'])
        epochs = random.choice(config.parameters['epochs']['values'])
        for epoch in range(epochs):
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
                wandb.log({"batch loss": loss.item()})
            # lossess.append(total_epoch_loss/training_size)
            avg_loss = total_epoch_loss / len(train_loader)
            lossess.append(total_epoch_loss.item() / len(train_loader.dataset))
            wandb.log({"loss": avg_loss, "epoch": epoch, "K":K, "l1":lambda1, "l2":lambda2})
            print("total epoch loss: ", total_epoch_loss)
            if epoch % 5 == 0:
                print('epoch {}, loss {}'.format(epoch, loss.item()))
                # print('cln or weights grad:', cln.G1.grad.data.cpu().numpy().flatten().round(2))
        wandb.agent(sweep_id, train_regressor(train_loader, loss_fn, learning_rate, max_epochs, input_size, K, device, name, P, torch, CLN), count=5)
    emp_loss = np.array([e.detach().numpy() for e in emp_loss])
    if emp_loss.all() < 1e-5:
        print("Zero Error")
    print(len(emp_loss))
    print(len(emp_loss[emp_loss < 0.1]))
    # print("P(loss < 0.1) = ", len(emp_loss[emp_loss < 0.1]) / len(emp_loss))
    return cln, lossess, K, sweep_id
    