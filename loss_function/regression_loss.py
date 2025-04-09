import torch


def bernoulli_kl_divergence(values, targets):
    # targets acts as p for P (P is the actual distribution)
    # values acts as q for Q (Q is the approximative distribution)

    p = targets
    q = values

    numerator = p - p * q
    denominator = q - p * q

    loss = p * torch.log(numerator / denominator)
    loss += torch.log((1 - p) / (1 - q))

    avg_loss = torch.mean(loss)

    return avg_loss





if __name__ == '__main__':
    values = torch.ones(1, 1) * 0.6
    targets = torch.ones(1, 1) * 0.5

    loss = bernoulli_kl_divergence(values, targets)
    print(loss)
