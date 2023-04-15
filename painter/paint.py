import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from actor import *
from stroke_gen import *
from render import *


def paint(img, max_step=40, divide=4, device="cuda"):

    width = 128
    device = torch.device(device)
    canvas_cnt = divide * divide
    T = torch.ones([1, 1, width, width], dtype=torch.float32).to(device)
    origin_shape = (img.shape[1], img.shape[0])
    coord_x, coord_y = torch.meshgrid(torch.arange(0, width) / (width - 1), torch.arange(0, width) / (width - 1))
    coord = torch.stack([coord_x, coord_y], dim=0).unsqueeze(0).to(device)  # Coordconv
    Decoder = FCN()
    Decoder.load_state_dict(torch.load('painter/renderer.pkl'))
    actor = ResNet(9, 18, 65)  # action_bundle = 5, 65 = 5 * 13
    actor.load_state_dict(torch.load('painter/actor.pkl'))
    actor = actor.to(device).eval()
    decoder = Decoder.to(device).eval()
    canvas = torch.zeros([1, 3, width, width]).to(device)

    def decode(x, this_canvas):  # b * (10 + 3)
        x = x.view(-1, 10 + 3)
        stroke = 1 - decoder(x[:, :10])
        stroke = stroke.view(-1, width, width, 1)
        color_stroke = stroke * x[:, -3:].view(-1, 1, 1, 3)
        stroke = stroke.permute(0, 3, 1, 2)
        color_stroke = color_stroke.permute(0, 3, 1, 2)
        stroke = stroke.view(-1, 5, 1, width, width)
        color_stroke = color_stroke.view(-1, 5, 3, width, width)
        result = []
        for idx in range(5):
            this_canvas = this_canvas * (1 - stroke[:, idx]) + color_stroke[:, idx]
            result.append(canvas)
        return this_canvas, result

    def smooth(this_img):
        def smooth_pix(cur_img, tx, ty):
            if tx == divide * width - 1 or ty == divide * width - 1 or tx == 0 or ty == 0:
                return cur_img
            cur_img[tx, ty] = (cur_img[tx, ty] + cur_img[tx + 1, ty] + cur_img[tx, ty + 1] +
                               cur_img[tx - 1, ty] + cur_img[tx, ty - 1] + cur_img[tx + 1, ty - 1] +
                               cur_img[tx - 1, ty + 1] + cur_img[tx - 1, ty - 1] + cur_img[tx + 1, ty + 1]) / 9
            return cur_img

        for p in range(divide):
            for q in range(divide):
                x = p * width
                y = q * width
                for k in range(width):
                    this_img = smooth_pix(this_img, x + k, y + width - 1)
                    if q != divide - 1:
                        this_img = smooth_pix(this_img, x + k, y + width)
                for k in range(width):
                    this_img = smooth_pix(this_img, x + width - 1, y + k)
                    if p != divide - 1:
                        this_img = smooth_pix(this_img, x + width, y + k)
        return this_img

    def small2large(x):
        # (d * d, width, width) -> (d * width, d * width)
        x = x.reshape(divide, divide, width, width, -1)
        x = np.transpose(x, (0, 2, 1, 3, 4))
        x = x.reshape(divide * width, divide * width, -1)
        return x

    def large2small(x):
        # (d * width, d * width) -> (d * d, width, width)
        x = x.reshape(divide, width, divide, width, 3)
        x = np.transpose(x, (0, 2, 1, 3, 4))
        x = x.reshape(canvas_cnt, width, width, 3)
        return x

    def get_img(result, use_divide=False):
        output = result.detach().cpu().numpy()  # d * d, 3, width, width
        output = np.transpose(output, (0, 2, 3, 1))
        if use_divide:
            output = small2large(output)
            output = smooth(output)
        else:
            output = output[0]
        output = (output * 255).astype('uint8')
        output = cv2.resize(output, origin_shape)
        return output

    patch_img = cv2.resize(img, (width * divide, width * divide))
    patch_img = large2small(patch_img)
    patch_img = np.transpose(patch_img, (0, 3, 1, 2))
    patch_img = torch.tensor(patch_img).to(device).float() / 255.

    img = cv2.resize(img, (width, width))
    img = img.reshape(1, width, width, 3)
    img = np.transpose(img, (0, 3, 1, 2))
    img = torch.tensor(img).to(device).float() / 255.

    all_canvas = []
    with torch.no_grad():
        if divide != 1:
            max_step = max_step // 2
        for i in range(max_step):
            step_num = T * i / max_step
            actions = actor(torch.cat([canvas, img, step_num, coord], 1))
            canvas, res = decode(actions, canvas)
            for j in range(5):
                all_canvas.append(get_img(res[j]))
        if divide != 1:
            canvas = canvas[0].detach().cpu().numpy()
            canvas = np.transpose(canvas, (1, 2, 0))
            canvas = cv2.resize(canvas, (width * divide, width * divide))
            canvas = large2small(canvas)
            canvas = np.transpose(canvas, (0, 3, 1, 2))
            canvas = torch.tensor(canvas).to(device).float()
            coord = coord.expand(canvas_cnt, 2, width, width)
            T = T.expand(canvas_cnt, 1, width, width)
            for i in range(max_step):
                step_num = T * i / max_step
                actions = actor(torch.cat([canvas, patch_img, step_num, coord], 1))
                canvas, res = decode(actions, canvas)
                for j in range(5):
                    all_canvas.append(get_img(res[j], True))
    return all_canvas
