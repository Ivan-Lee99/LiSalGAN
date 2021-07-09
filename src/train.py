import time
from discriminator import *
from generator import *
from data_loader import *
from tqdm import tqdm

use_gpu = torch.cuda.is_available()

batch_size = 34
lr = 0.0003
epochs = 180
alpha = 0.05

# Load model
discriminator = Discriminator()
generator = Generator()
loss_function = nn.BCELoss()
if use_gpu:
    discriminator.cuda()
    generator.cuda()
    loss_function.cuda()

discriminator_optim = torch.optim.Adagrad(discriminator.parameters(), lr=lr)
generator_optim = torch.optim.Adagrad(discriminator.parameters(), lr=lr)

train_data = DataLoader(pathToResizedImagesTrain, batch_size)
num_batch = train_data.num_batch
print("%d batches totally" % num_batch)

start_time = time.time()

for epoch in tqdm(range(1, epochs + 1)):
    stage = 1

    g_cost_avg = 0
    d_cost_avg = 0

    for index in range(num_batch):
        img, salmap = train_data.get_batch()
        real = torch.FloatTensor(np.ones(batch_size, dtype=float))
        fake = torch.FloatTensor(np.zeros(batch_size, dtype=float))
        if use_gpu:
            img.cuda()
            salmap.cuda()
            real.cuda()
            fake.cuda()

        if stage % 2 == 1:
            print('Training discriminator...............')

            discriminator_optim.zero_grad()
            input_d = torch.cat((img, salmap), 1)
            outputs = discriminator(input_d).squeeze()
            d_real_loss = loss_function(outputs, real)
            real_score = outputs.data.mean()

            fake_map = generator(img)
            input_d = torch.cat((img, fake_map), 1)
            outputs = discriminator(input_d).squeeze()
            d_fake_loss = loss_function(outputs, fake)

            d_loss = d_real_loss + d_fake_loss
            d_cost_avg += d_loss
            d_loss.backward()
            discriminator_optim.step()

        else:
            print('Training generator...................')

            generator_optim.zero_grad()
            fake_map = generator(img)
            g_loss = loss_function(fake_map, salmap)
            input_d = torch.cat((img, fake_map), 1)
            outputs = discriminator(input_d)
            d_loss = loss_function(outputs, real)
            fake_score = outputs.data.mean()

            g_loss = alpha * g_loss + d_loss
            g_cost_avg += g_loss
            g_loss.backward()
            generator_optim.step()

        stage += 1

        if (index + 1) % 100 == 0:
            print("Epoch [%d/%d], Step[%d/%d], d_loss: %.4f, g_loss: %.4f, D(x): %2.f, D(G(x)): %.2f, time: %4.4f"
                  % (epoch, epochs, index + 1, num_batch, d_loss.data[0], g_loss.data[0],
                         real_score, fake_score, time.time() - start_time))
    if (epoch + 1) % 3 == 0:
        print('Epoch:', epoch, ' train_loss->', (d_cost_avg, g_cost_avg))

torch.save(generator.state_dict(), './generator.pkl')
torch.save(discriminator.state_dict(), './discriminator.pkl')
print('Done')
