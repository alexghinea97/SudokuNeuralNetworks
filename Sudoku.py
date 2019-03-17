import numpy as np
import torch
import pandas as pd
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.optim as optim

quizzes = np.zeros((1000000, 81), np.int32)
solutions = np.zeros((1000000, 81), np.int32)

for i, line in enumerate(open('datasets/sudoku.csv', 'r').read().splitlines()[1:]):
    quiz, solution = line.split(",")
    for j, q_s in enumerate(zip(quiz, solution)):
        q, s = q_s
        quizzes[i, j] = q
        solutions[i, j] = s

quizzes = quizzes.reshape((-1, 81))
solutions = solutions.reshape((-1, 81))

from sklearn.model_selection import train_test_split

X_train, x_test, Y_train, y_test = train_test_split(quizzes,
                                                    solutions,
                                                    test_size=0.2,
                                                    random_state=0 )

dtype = torch.float
X_train_tensor = torch.tensor(X_train, dtype = dtype)
x_test_tensor = torch.tensor(x_test, dtype = dtype)

Y_train_tensor = torch.tensor(Y_train, dtype = dtype)
y_test_tensor = torch.tensor(y_test, dtype = dtype)


inp = 81
out = 810
hid = 120

model = torch.nn.Sequential(torch.nn.Linear(inp, hid),
                            torch.nn.Linear(hid, hid),
                            torch.nn.Linear(hid, out),
)

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())
learning_rate = 0.0001



for iter in range(100):

    optimizer.zero_grad()
    y_pred = model(X_train_tensor)
    
    targetAsLong = Y_train_tensor.long()
    loss = loss_fn(y_pred.view(81 * len(X_train_tensor), 10), \
                                targetAsLong.view(81 * len(X_train_tensor),))
    
    if iter % 10 == 0:
        print(iter, loss.item())

    model.zero_grad()
    loss.backward()
    optimizer.step()

    with torch.no_grad():
        for param in model.parameters():
            param -= learning_rate * param.grad

y_pred_tensor = model(x_test_tensor)
y_pred = y_pred_tensor.detach().numpy()


y_pred_tensor = model(x_test_tensor)
y_pred = y_pred_tensor.detach().numpy()

print(y_pred)
print(y_test)