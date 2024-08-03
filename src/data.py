import torch
import cv2
import torch.nn.functional as F
import os

def load_images(style_path, content_path, device, new_size=(512, 512)):
    style_pic = torch.tensor(cv2.cvtColor(cv2.imread(style_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)).permute(2, 0, 1).unsqueeze(0).to(device) / 255.0
    content_pic = torch.tensor(cv2.cvtColor(cv2.imread(content_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)).permute(2, 0, 1).unsqueeze(0).to(device) / 255.0

    style_pic = F.interpolate(style_pic, size=new_size, mode='bilinear', align_corners=False)
    content_pic = F.interpolate(content_pic, size=new_size, mode='bilinear', align_corners=False)

    return style_pic, content_pic

def save_images(output_img, output_img_path= './', concated= False, content_img= None, style_img= None): 

    image_to_save = cv2.cvtColor((output_img.squeeze().permute(1, 2, 0).detach().cpu().numpy() * 255.0), cv2.COLOR_RGB2BGR)
    out = os.path.join(output_img_path, 'output.jpg')
    cv2.imwrite(out, image_to_save)
    
    if concated:
        images_concated = torch.cat([content_img, style_img, output_img], dim= 3)  
        out = os.path.join(output_img_path, '/all_images.jpg')   
        images_to_save = cv2.cvtColor((images_concated.squeeze().permute(1, 2, 0).detach().cpu().numpy() * 255.0), cv2.COLOR_RGB2BGR)        
        cv2.imwrite(out, images_to_save)
