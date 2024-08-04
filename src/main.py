import torch
from torchvision.models import vgg16, VGG16_Weights
from data import load_images, save_images
from style_transfer import apply_style
import options



def main(configs):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = vgg16(VGG16_Weights).to(device)
    style_img_path, content_img_path, output_img_path = configs.style_path, configs.content_path, configs. output_path
    style_img, content_img = load_images(style_img_path, content_img_path, device)
    output_img = apply_style(content_img, style_img, model, configs.alpha_beta_ratio, configs.num_steps)
    print("Saving the result...")
    if configs.save_concatenated_images:
        save_images(output_img, output_img_path, concated=True, content_img= content_img, style_img= style_img)
    else:
        save_images(output_img, output_img_path, concated= False)
    
    
if __name__ == "__main__":
    configs = options.read_command_line()
    main(configs)