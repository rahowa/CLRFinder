import keras
import keras.backend as K
import keras.layers as L
from keras.datasets import mnist
from scalers import Scaler, triangular

from keras_pipeline import KerasScheduler, LRFinderKeras

def keras_model():
    inp = L.Input(shape=(32, 32, 1))
    x = L.Conv2D(16, 3, padding="SAME", use_bias=False)(inp)
    x = L.MaxPool2D()(x)
    x = L.BatchNormalization()(x)
    x = L.Activation('relu')(x)

    x = L.Conv2D(32, 3, padding="SAME", use_bias=False)(inp)
    x = L.MaxPool2D()(x)
    x = L.BatchNormalization()(x)
    x = L.Activation('relu')(x)

    x = L.Conv2D(64 3, padding="SAME", use_bias=False)(inp)
    x = L.MaxPool2D()(x)
    x = L.BatchNormalization()(x)
    x = L.Activation('relu')(x)
    x = L.GlobalAvgPool2D()(x)
    x = L.Dense(10)
    output = L.Activation("softmax")(x)

    model = keras.models.Model(inp, output, name="Base model")
    return model


# def train(model, criterion, optimizer, loader, device, scheduler):
#     epoch_loss = 0
#     epoch_acc = 0
#     model.train()
#     for data, target in loader:
#         data = data.to(device)
#         target = target.to(device)
#         optimizer.zero_grad()
#         output = model(data)
#         loss = criterion(output, target)
#         epoch_loss += loss.item()
#         epoch_acc += output.max(1)[1].eq(target).sum().item()
#         loss.backward()
#         optimizer.step()
#         scheduler.step()
#     return model, epoch_acc / len(loader.dataset), epoch_loss / len(loader)


if __name__ == '__main__':
    batch_size = 64
    model = keras_model()
    criterion = K.categorical_crossentropy
    optimizer = keras.optimizers.Adam(lr=1.0)

    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    finder = LRFinderKeras(model=model, min_lr=1e-3, max_lr=0.9)
    finder.run(criterion, optimizer)
    # finder.plot_result()
    best_lr = finder.best_lr()
    print(best_lr)

    scaler = Scaler("cycle", triangular)
    scheduler = PytorchScheduler(best_lr[0], best_lr[1],
                                 scaler,
                                 len(train_loader), 1., optimizer)

    optimizer = optim.Adam(model.parameters(), best_lr[1])
    for ep in range(10):
        model, acc, loss = train(model, criterion, optimizer,
                                 train_loader, device, scheduler)
        print(f"Epooch: {ep}, loss: {loss}, acc: {acc}")
