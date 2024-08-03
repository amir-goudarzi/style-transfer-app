import argparse

def read_command_line():
    parser = argparse.ArgumentParser(description='Style Transfer')

    parser.add_argument('--style_path', type=str, required=True, help='Path to the style image')
    parser.add_argument('--content_path', type=str, required=True, help='Path to the content image')
    parser.add_argument('--alpha_beta_ratio', type=float, required=False, default=1e4, help='The ratio of alpha (content weight) to beta (style weight)')
    parser.add_argument('--output_path', type=str, required=False, default= './', help='Path to save the output image')
    parser.add_argument('--save_concatenated_images', type= bool, required= False, default=False, action='store_true', help='Save the images concatenated (content, style, output)')

    args = parser.parse_args()
    return args
