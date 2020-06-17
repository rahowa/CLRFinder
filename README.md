# Cyclic learning rate finder
> This repository contains Pytorch implementations of learning rate finder and cyclic scheduler.

# Install 
> Clone repo
```shell
$ git clone https://github.com/rahowa/cyclic_learning_rate_finder.git
```
> Install poetry for package building
```
$ pip install poetry
```
> Build and install python package with poetry
```shell
$ poetry install
```

#Usage 
```python
model = TestModel()
model.to('cpu')
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), 1.0)

transforms = Compose([ToTensor()])
train_dataset = MNIST('./', train=True, download=True, transform=transforms)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, drop_last=True)

finder = LRFinderPytorch(model=model, min_lr=1e-3, max_lr=0.9, loader=train_loader, n_epochs=1)
finder.run(criterion, optimizer)
best_lr = finder.best_lr()
scheduler = PytorchScheduler(best_lr[0], best_lr[1], Scaler.triangle1, len(train_loader), 1.)

optimizer = optim.Adam(model.parameters(), best_lr[1])
for ep in range(10):
    model, acc, loss = train(model, criterion, optimizer, train_loader, 'cpu')
    print(f"Epooch: {ep}, loss: {loss}, acc: {acc}")
```

#License
