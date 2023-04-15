import torch
import numpy as np
import os
import torch.nn.functional
import cv2
import sys
import argparse

from segment_anything import sam_model_registry, SamPredictor
from painter.paint import paint
from lama.lama_inpaint import inpaint_img_with_lama


def dilate_mask(mask, dilate_factor=15):
    mask = mask.astype(np.uint8)
    mask = cv2.dilate(
        mask,
        np.ones((dilate_factor, dilate_factor), np.uint8),
        iterations=1
    )
    return mask


def setup_args(parser):
    parser.add_argument(
        "--img_path", type=str, required=True,
        help="Path to a single image",
    )
    parser.add_argument(
        "--output_dir", type=str, default='output/',
        help="Output path",
    )


def show_animation(all_canvas, window, pos=(0, 0), mask=None, bg=None, sleep=30):
    if bg is None:
        bg = np.zeros((all_canvas[0].shape[0], all_canvas[0].shape[1], 3), dtype=all_canvas[0].dtype)
    if mask is None:
        mask = np.ones((bg.shape[0], bg.shape[1], 1), dtype=bg.dtype)
    else:
        mask = mask.reshape((mask.shape[0], mask.shape[1], 1))
    for canvas in all_canvas:
        fg = np.zeros_like(bg)
        fg[pos[1]:pos[1] + canvas.shape[0], pos[0]:pos[0] + canvas.shape[1]] = canvas
        bg = fg * mask + bg * (1 - mask)
        cv2.imshow(window, cv2.cvtColor(bg, cv2.COLOR_RGB2BGR))
        cv2.waitKey(sleep)
    return bg


def get_mask_pos(mask):
    mask_pad = np.zeros((mask.shape[0] + 2, mask.shape[1] + 2), dtype=mask.dtype)
    mask_pad[1:-1, 1:-1] = mask
    argmax_1 = mask_pad.argmax(1)
    left = argmax_1[argmax_1 > 0].min() - 1
    argmax_0 = mask_pad.argmax(0)
    up = argmax_0[argmax_0 > 0].min() - 1
    argmax_inv_1 = mask_pad[:, ::-1].argmax(1)
    right = mask.shape[1] - argmax_inv_1[argmax_inv_1 > 0].min()
    argmax_inv_0 = mask_pad[::-1].argmax(0)
    bottom = mask.shape[0] - argmax_inv_0[argmax_inv_0 > 0].min()
    return left, up, right, bottom


def main(args):
    """ Argument """
    parser = argparse.ArgumentParser()
    setup_args(parser)
    args = parser.parse_args(args)

    """ Setting Input and Output Images """
    os.makedirs(args.output_dir, exist_ok=True)
    img = cv2.imread(args.img_path)
    h, w = img.shape[:2]

    """ Interaction """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    sam_checkpoint = "segment-anything/sam_vit_h_4b8939.pth"
    model_type = "vit_h"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    predictor = SamPredictor(sam)
    predictor.set_image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    all_vis_input = []
    vis_input = img.copy()
    all_vis_input.append(vis_input)

    cv2.namedWindow('Input', cv2.WINDOW_AUTOSIZE)
    cv2.imshow('Input', img)
    cv2.waitKey(1000)
    cv2.namedWindow('Output', cv2.WINDOW_AUTOSIZE)
    all_canvas = paint(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), max_step=40, divide=4, device=device)
    all_vis_output = []
    #vis_output = show_animation(all_canvas, "Output")
    #all_vis_output.append(vis_output)
    vis_output = np.zeros_like(img)
    all_vis_output.append(vis_output)
    cv2.imshow('Output', vis_output)
    cv2.waitKey(100)

    all_mask = []
    all_pos = []

    def mouse_handle(event, x, y, _1, _2):
        nonlocal predictor, vis_input, vis_input_, is_working, is_dragging, cur_points_labels, \
            cur_box, mask, color, is_valid

        if (not is_working) and (event == cv2.EVENT_LBUTTONDOWN or event == cv2.EVENT_RBUTTONDOWN or (
                is_dragging and event == cv2.EVENT_MBUTTONUP)):
            is_working = True
            if event == cv2.EVENT_LBUTTONDOWN:
                cur_points_labels[0].append([x, y])
                cur_points_labels[1].append(1)
                cv2.circle(vis_input, (x, y), radius=5, color=(0, 255, 0), thickness=-1)
                cv2.imshow('Input', vis_input)
            elif event == cv2.EVENT_RBUTTONDOWN:
                cur_points_labels[0].append([x, y])
                cur_points_labels[1].append(0)
                cv2.circle(vis_input, (x, y), radius=5, color=(0, 0, 255), thickness=-1)
                cv2.imshow('Input', vis_input)
            elif event == cv2.EVENT_MBUTTONUP:
                cur_box += [x, y]
                if cur_box[0] > cur_box[2]:
                    cur_box[2] = cur_box[0]
                    cur_box[0] = x
                if cur_box[1] > cur_box[3]:
                    cur_box[3] = cur_box[1]
                    cur_box[1] = y
                is_dragging = False
                cv2.imshow('Input', vis_input)
            mask, _, _ = predictor.predict(
                point_coords=np.array(cur_points_labels[0]) if len(cur_points_labels[0]) > 0 else None,
                point_labels=np.array(cur_points_labels[1]) if len(cur_points_labels[1]) > 0 else None,
                box=np.array(cur_box) if len(cur_box) == 4 else None,
                multimask_output=False
            )
            if not is_valid:
                is_valid = True

            mask = mask.reshape((mask.shape[1], mask.shape[2], 1))
            vis_input_ = (mask.astype(np.float) * (color * 0.6 + vis_input.astype(np.float) * 0.4) +
                          (1 - mask.astype(np.float)) * vis_input.astype(np.float)).astype(np.uint8)
            cv2.imshow('Input', vis_input_)
            is_working = False
        elif not is_dragging and event == cv2.EVENT_MBUTTONDOWN:
            is_dragging = True
            cur_box = [x, y]
        elif is_dragging and event == cv2.EVENT_MOUSEMOVE:
            vis_input__ = vis_input.copy()
            cv2.rectangle(vis_input__, (cur_box[0], cur_box[1]), (x, y), (255, 0, 0))
            cv2.imshow('Input', vis_input__)

    while True:
        print('Please Choose an Option for Content Image:')
        print('\t1: Select an Area by SAM')
        print('\t2: Undo Previous Content & Style Selection')
        print('\tOther: Finish!')
        option = input()
        if option == '1':
            vis_input_ = vis_input.copy()
            is_working = False
            is_dragging = False
            is_valid = False
            cur_points_labels = ([], [])
            cur_box = []
            color = (np.random.random(3) * 255).reshape(1, 1, 3)
            mask = np.zeros((h, w, 1)).astype(np.uint8)

            print('\t\tLeft Clik on the Content Image to Set a Foreground Point;')
            print('\t\tRight Clik on the Content Image to Set a Background Point;')
            print('\t\tMiddle Clik on the Content Image and Drag Your Mouse to Specify a Bounding Box;')
            print('\t\tPress Any Key to Finish Your Current Selection')
            cv2.setMouseCallback('Input', mouse_handle)
            cv2.waitKey(0)

            if not is_valid:
                print('\t\tInvalid Selection! Please Re-try:')
            else:
                all_vis_input.append(vis_input_.copy())
                mask = dilate_mask(mask, 15)
                all_mask.append(mask)
                vis_input = vis_input_
                left, up, right, bottom = get_mask_pos(mask)
                all_pos.append((left, up, right, bottom))

                total_mask = np.zeros((h, w))
                for mask in all_mask:
                    total_mask += mask
                total_mask = total_mask > 0
                bg = inpaint_img_with_lama(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), total_mask)
                all_canvas = paint(bg, max_step=20, divide=1, device=device)
                vis_output = show_animation(all_canvas, 'Output')
                for pos, mask in zip(all_pos, all_mask):
                    left, up, right, bottom = pos
                    all_canvas = paint(cv2.cvtColor(img[up:bottom, left:right], cv2.COLOR_BGR2RGB), max_step=40, divide=4, device=device)
                    vis_output = show_animation(all_canvas, 'Output', (left, up), mask, vis_output)
                    cv2.waitKey(1000)
                all_vis_output.append(vis_output)

        elif option == '2':
            if len(all_mask) == 0:
                print('\t\tNo Previous Selection! Please Re-enter:')
            else:
                all_vis_input.pop()
                all_vis_output.pop()
                all_mask.pop()
                all_pos.pop()
                vis_input = all_vis_input[-1]
                vis_output = all_vis_output[-1]
                cv2.imshow("Input", vis_input)
                cv2.waitKey(1000)
                cv2.imshow("Output", cv2.cvtColor(vis_output, cv2.COLOR_RGB2BGR))
                cv2.waitKey(1000)
        else:
            break
    cv2.imwrite(os.path.join(args.output_dir, os.path.basename(args.img_path)), cv2.cvtColor(vis_output, cv2.COLOR_RGB2BGR))
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main(sys.argv[1:])
