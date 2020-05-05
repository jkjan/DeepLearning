from CNN import *
from loader_downloader import *

device = "cuda" if torch.cuda.is_available() else 'cpu'

torch.manual_seed(777)
if device == 'cuda':
    torch.cuda.manual_seed_all(777)

# parameters
learning_rate = 0.001
training_epochs = 15
batch_size = 100

#MNIST dataset
mnist_train, mnist_test = download_train_and_test()
loaded_data = data_loader(mnist_train, mnist_test, batch_size)

model = CNN().to(device)

try:
    model.load_state_dict(torch.load("./mnist_model.pth"))
    print("Trained file exists.")
except FileNotFoundError:
    print("Trained file doesn't exist.")
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # training
    total_batch = len(loaded_data)

    print("learning starts!")

    for epoch in range(training_epochs):
        avg_cost = 0

        for X, Y in loaded_data:
            X = X.to(device)
            Y = Y.to(device)
            optimizer.zero_grad()
            # 중요

            hypothesis = model(X)
            cost = criterion(hypothesis, Y)
            cost.backward()
            optimizer.step()
            print(cost.item())
            avg_cost += cost / total_batch

        print("[Epoch:{}] cost = {}".format(epoch+1, avg_cost))

    print("Learning finished!")

    torch.save(model.state_dict(), "./mnist_model.pth")

with torch.no_grad():
    X_test = mnist_test.test_data.view(len(mnist_test), 1, 28,28).float().to(device)
    Y_test = mnist_test.test_labels.to(device)

    prediction = model(X_test)
    correct = torch.argmax(prediction, 1) == Y_test
    accuracy = correct.float().mean()
    print('Accuracy:', accuracy.item())