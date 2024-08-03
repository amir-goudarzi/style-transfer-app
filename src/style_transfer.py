import torch
from torch import nn
from gram_matrix import gramMatrix

class FeatureExtractor(nn.Module):
    def __init__(self, model, num_layers):
        super(FeatureExtractor, self).__init__()
        self.features = nn.Sequential()
        i = 0
        for name, layer in model.features.named_children():
            self.features.add_module(name, layer)
            if i == num_layers:
                break
            i += 1

    def forward(self, x):
        return self.features(x)
    
def calculate_gram_matrices(input_image, style_layers, model):
    input_gram_matrices= list()
    M = list()

    for layer_idx in style_layers:
        style_feature_extractor = FeatureExtractor(model, layer_idx)
        input_style_features = style_feature_extractor(input_image)

        gram_matrix = gramMatrix(input_style_features)
        input_gram_matrices.append(gram_matrix)
        m = input_style_features[0].shape[1] * input_style_features[0].shape[2]
        M.append(m)

    return input_gram_matrices, M


def calculate_style_loss(loss_style, input_gram_matrices, style_gram_matrices, W: list, M: list):
    assert len(input_gram_matrices) == len(style_gram_matrices) == len(W) == len(M)
    L = len(style_gram_matrices)

    style_losses = list()

    for layer in range(L):
        N = input_gram_matrices[layer].shape[0]
        style_loss = W[layer] * (loss_style(input_gram_matrices[layer], style_gram_matrices[layer]) / (4 * N**2 * M[layer]**2))
        style_losses.append(style_loss)

    total_style_loss = sum(style_losses)

    return total_style_loss

def generate(optimizer, closure, steps, W: list, a, b):
    for step in steps:
        optimizer.step(closure)
        print(f'Step number {step}')


def apply_style(content_img, style_img, model, ratio):

    content_layer = 19
    style_layers = [0, 5, 10, 17, 24]

    content_feature_extractor = FeatureExtractor(model, content_layer)
    model.eval()


    with torch.no_grad():
        content_features = content_feature_extractor(content_img)

    style_gram_matrices = list()
    for style_layer in style_layers:
        style_feature_extractor = FeatureExtractor(model, style_layer)
        with torch.no_grad():
            style_features = style_feature_extractor(style_img)
            gram_matrix = gramMatrix(style_features)
            style_gram_matrices.append(gram_matrix)
    input_image = torch.clone(content_img).requires_grad_()
    loss_content = nn.MSELoss(reduction= 'sum')
    loss_style = nn.MSELoss(reduction= 'sum')
    optimizer = torch.optim.LBFGS(params= [input_image], lr=1)
    a = 1
    b = a / ratio

    def closure():
        optimizer.zero_grad()

        # Calculate content features and Gram matrices within the closure
        input_content_features = content_feature_extractor(input_image)
        input_gram_matrices, M = calculate_gram_matrices(input_image, style_layers, model)

        # Loss computation
        content_loss = loss_content(input_content_features, content_features) / 2
        style_loss = calculate_style_loss(loss_style, input_gram_matrices, style_gram_matrices, W, M)

        total_loss = (a * content_loss) + (b * style_loss)
        total_loss.backward()

        return total_loss
    
    steps = range(1, 16)
    W = [(1 / len(style_layers)) for x in range(len(style_layers))]
    generate(optimizer, closure, steps, W, a, b)

    return input_image