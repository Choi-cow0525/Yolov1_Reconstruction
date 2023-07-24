import torch
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter("./runs/hellobello/")
# 머신러닝에서는 loss같은 주요 측정 항목과 학습 중 그것이 어떻게 변하는지 이해하는 것이
# 중요하다. 스칼라는 각 학습 단계(step)에서의 손실 값이나 각 에폭 이후의 정확도를 저장하는 데 도움을 준다

# scalar 값을 기록하려면 add_scalar(tag, scalar_value, global_step = None, walltime = None)을 사용해야 함.

x = torch.arange(-5, 5, 0.1).view(-1, 1)
print(x.shape)
print(x.dtype)
print(x.dim())

y = -5 * x + 0.1 * torch.randn(x.size()) # random with normal distribution

model = torch.nn.Linear(1, 1)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr = 0.1)

def train_model(epoch):
    y1 = model(x)
    loss = criterion(y1, y)
    writer.add_scalar("Loss/train", loss, epoch)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

for epoch in range(10):
    train_model(epoch)

x1 = torch.arange(0, 100, 1, dtype=torch.float).view(-1, 1)
print(x1.shape)
print(x1.dtype)
print(x1.dim())
y1 = model(x1)
print(f"y1 shape : {y1.shape}")
ii = torch.arange(0, 100, 1)
print(x1)
print(y1)
for i in ii:
    writer.add_scalar('y=-5x', y1[i], i)
    #print(y1[i])

writer.flush()
writer.close()