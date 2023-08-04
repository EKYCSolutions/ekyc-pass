from argparse import Namespace
import os
import os.path as osp
import time
import cv2
import torch
from football_track_bytetack_config import args
from loguru import logger
import numpy as np
from yolox.exp import get_exp
from yolox.data.data_augment import preproc
from yolox.utils import fuse_model, get_model_info, postprocess
from yolox.utils.visualize import plot_tracking
from yolox.tracker.byte_tracker import BYTETracker
from yolox.tracking_utils.timer import Timer
import matplotlib.pyplot as plt
from utils.helper import detect_color
from tracker.helper import imgByteToNumpy
import torch
from yolox.data import ValTransform
IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]


class Football_bytetrack():
    def __init__(self) -> None:
        self.args = self.load_argument(args)
        self.exp = get_exp(self.args.exp_file, self.args.name)

        if self.args.conf is not None:
            self.exp.test_conf = self.args.conf
        if self.args.nms is not None:
            self.exp.nmsthre = self.args.nms
        if self.args.tsize is not None:
            self.exp.test_size = (self.args.tsize, self.args.tsize)
        if not self.args.experiment_name:
            self.args.experiment_name = self.exp.exp_name

        self.model = self.exp.get_model().to(self.args.device)

        ckpt_file = self.args.ckpt
        ckpt = torch.load(
            ckpt_file, map_location=torch.device(
                'cuda' if torch.cuda.is_available() else 'cpu'))

        self.model.eval()
        self.model.load_state_dict(ckpt["model"])

        self.predictor = Predictor(self.model, self.exp, None, None,
                                   self.args.device, self.args.fp16)

        self.tracker = BYTETracker(self.args, frame_rate=self.args.fps)
        self.preproc = ValTransform(legacy=False)

    def load_argument(self, kwargs):
        args = Namespace(**kwargs)
        return args

    def get_image_list(self, path):
        image_names = []
        for maindir, subdir, file_name_list in os.walk(path):
            for filename in file_name_list:
                apath = osp.join(maindir, filename)
                ext = osp.splitext(apath)[1]
                if ext in IMAGE_EXT:
                    image_names.append(apath)
        return image_names

    def detect_players(self, img):
        # if img != np.ndarray:
        #     img = imgByteToNumpy(img)

        results = []
        timer = Timer()

        outputs, img_info = self.predictor.inference(img, timer)

        if outputs[0] is not None:

            online_targets, labels, bboxes = self.tracker.update(
                outputs[0], [img_info['height'], img_info['width']], self.exp.test_size)

            online_tlwhs = []
            online_ids = []
            online_scores = []

            b = []
            for i, t in enumerate(online_targets):
                tlwh = t.tlwh
                tid = t.track_id

                vertical = tlwh[2] / \
                    tlwh[3] > self.args.aspect_ratio_thresh
                if tlwh[2] * tlwh[3] > self.args.min_box_area and not vertical:
                    online_tlwhs.append(tlwh)
                    online_ids.append(tid)
                    online_scores.append(t.score)
                    # save results
                    results.append({
                        "x": tlwh[0],
                        "y": tlwh[1],
                        "w": tlwh[2],
                        "h": tlwh[3],
                        "label": labels[i]
                    })
                    timer.toc()
        return results

    def image_demo(self, predictor, vis_folder, current_time):

        if osp.isdir(self.args.path):
            files = self.get_image_list(self.args.path)
        else:
            files = [self.args.path]
        files.sort()
        tracker = BYTETracker(self.args, frame_rate=self.args.fps)
        timer = Timer()
        results = []

        for frame_id, img_path in enumerate(files, 0):
            outputs, img_info = predictor.inference(img_path, timer)

            if frame_id == 0:
                bg_ratio = int(np.ceil(img_info["width"]/(3*115)))
                gt_img = cv2.imread('tracker/inference/black.jpg')
                gt_img = cv2.resize(gt_img, (115*bg_ratio, 74*bg_ratio))
                gt_h, gt_w, _ = gt_img.shape

            bg_img = gt_img.copy()

            main_frame = img_info["raw_img"]

            if outputs[0] is not None:
                online_targets, labels, bboxes = tracker.update(
                    outputs[0], [img_info['height'], img_info['width']], self.exp.test_size)

                online_tlwhs = []
                online_ids = []
                online_scores = []

                for i, t in enumerate(online_targets):
                    tlwh = t.tlwh
                    tid = t.track_id
                    size = tlwh[2] * tlwh[3]

                    if size > self.args.min_box_area and size < self.args.max_box_area:
                        online_tlwhs.append(tlwh)
                        online_ids.append(tid)
                        online_scores.append(t.score)

                        x, y, w, h = tlwh
                        x, y, w, h = int(x), int(y), int(w), int(h)
                        x_center, y_center = x + w // 2, y + h // 2

                        print(x, y, w, h)

                        results.append({
                            "x": x,
                            "y": y,
                            "w": w,
                            "h": h,
                            "label": labels[i]
                        })

                        print(i, x_center, y_center)
                        print(main_frame.shape)

                        color, __dict__ = detect_color(
                            main_frame[y: y+h, x: x+w])

                        cv2.imshow("img", main_frame[y: y+h, x: x+w])
                        cv2.waitKey(0)
                        cv2.destroyAllWindows()
                        cv2.waitKey(1)

                        # cv2.circle(bg_img, coords, bg_ratio + 1, color, -1)
                    else:
                        bg_img = cv2.putText(bg_img, 'Too close', (gt_w // 2, gt_h // 2), cv2.FONT_HERSHEY_COMPLEX,
                                             1.4, (255, 255, 255), 2,  cv2.LINE_AA)
                        stack = self.img_vertical_stack(main_frame, bg_img)
                        break

                online_im = plot_tracking(
                    img_info['raw_img'], online_tlwhs, online_ids, frame_id=frame_id, fps=1. /
                    1)

                # stack = self.img_vertical_stack(online_im, bg_img)
                # cv2.imshow("bg", stack)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
                # cv2.waitKey(1)

        if self.args.save_result:
            timestamp = time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
            save_folder = osp.join(vis_folder, timestamp)

            os.makedirs(save_folder, exist_ok=True)
            cv2.imwrite(
                osp.join(save_folder, osp.basename(img_path)), online_im)

        return results

    def imageflow_demo(self, predictor, vis_folder, current_time):
        cap = cv2.VideoCapture(
            self.args.path if self.args.demo == "video" else self.args.camid)

        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float

        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        fps = cap.get(cv2.CAP_PROP_FPS)
        timestamp = time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
        save_folder = osp.join(vis_folder, timestamp)
        os.makedirs(save_folder, exist_ok=True)

        bg_ratio = int(np.ceil(width/(3*115)))
        gt_img = cv2.imread('tracker/inference/black.jpg')
        gt_img = cv2.resize(gt_img, (115*bg_ratio, 74*bg_ratio))

        gt_h, gt_w, _ = gt_img.shape

        if self.args.demo == "video":
            save_path = osp.join(save_folder, self.args.path.split("/")[-1])
        else:
            save_path = osp.join(save_folder, "camera.mp4")

        vid_writer = cv2.VideoWriter(
            save_path, cv2.VideoWriter_fourcc(
                *"mp4v"), fps, (int(width), int(height)))
        logger.info(f"video save_path is {save_path}")

        tracker = BYTETracker(self.args, frame_rate=30)
        timer = Timer()
        frame_id = 0
        results = []

        while True:
            bg_img = gt_img.copy()
            if frame_id % 20 == 0 and frame_id != 0:
                logger.info('Processing frame {} ({:.2f} fps)'.format(
                    frame_id, 1. / max(1e-5, timer.average_time)))

            ret_val, frame = cap.read()
            if ret_val:
                outputs, img_info = predictor.inference(frame, timer)
                main_frame = frame.copy()

                if outputs[0] is not None:
                    online_targets, _, _ = tracker.update(
                        outputs[0], [img_info['height'], img_info['width']], self.exp.test_size)
                    online_tlwhs = []
                    online_ids = []

                    online_scores = []

                    for i, t in enumerate(online_targets):

                        tlwh = t.tlwh
                        tid = t.track_id
                        vertical = tlwh[2] / \
                            tlwh[3] > self.args.aspect_ratio_thresh
                        size = tlwh[2] * tlwh[3]

                        if size > self.args.min_box_area and size < self.args.max_box_area:
                            online_tlwhs.append(tlwh)
                            online_ids.append(tid)
                            online_scores.append(t.score)

                            x = np.maximum(0, int(tlwh[0]))
                            y = np.maximum(0, int(tlwh[1]))
                            w = np.maximum(0, int(tlwh[2]))
                            h = np.maximum(0, int(tlwh[3]))

                            crop = main_frame[y: y+h, x: x+w]

                            if (len(crop) == 0):
                                continue

                            x_center, y_center = x + w // 2, y + h // 2

                            color, key = detect_color(crop)

                            # if t.label == "sports ball":
                            #     cv2.circle(bg_img, coords,
                            #                bg_ratio + 1, (0, 255, 255), -1)
                            # else:
                            #     cv2.circle(bg_img, coords,
                            #                bg_ratio + 1, color, -1)

                            results.append(
                                f"{frame_id},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},{t.score:.2f},{t.label},{x_center},{y_center},{key}.  \n"
                            )
                        else:
                            bg_img = cv2.putText(bg_img, 'Too close', (gt_w // 2, gt_h // 2), cv2.FONT_HERSHEY_COMPLEX,
                                                 1.4, (255, 255, 255), 2,  cv2.LINE_AA)
                            stack = self.img_vertical_stack(main_frame, bg_img)
                            break

                    timer.toc()
                    online_im = plot_tracking(
                        img_info['raw_img'], online_tlwhs, online_ids, frame_id=frame_id + 1, fps=1. / timer.average_time
                    )

                else:
                    online_im = img_info['raw_img']
                if self.args.save_result:
                    vid_writer.write(online_im)
            else:
                break

            frame_id += 1

            print(f"frame id : {frame_id} / {length} ")

        if self.args.save_result:
            res_file = osp.join(vis_folder, f"{timestamp}.txt")
            with open(res_file, 'w') as f:
                f.writelines(results)
            logger.info(f"save results to {res_file}")

    def img_vertical_stack(self, top, bottom):
        pad = np.full(top.shape, fill_value=[255, 255, 255])

        pad[200: 100 + bottom.shape[0],
            200: 200 + bottom.shape[1]] = bottom

        stack = np.vstack((top, pad))
        stack = stack.astype(np.uint8)
        return stack

    def get_image_list(self, path):
        image_names = []
        for maindir, subdir, file_name_list in os.walk(path):
            for filename in file_name_list:
                apath = osp.join(maindir, filename)
                ext = osp.splitext(apath)[1]
                if ext in IMAGE_EXT:
                    image_names.append(apath)
        return image_names

    def write_results(filename, results):
        save_format = '{frame},{id},{x1},{y1},{w},{h},{s},{label},-1,-1\n'
        with open(filename, 'w') as f:
            for frame_id, tlwhs, track_ids, scores in results:
                for tlwh, track_id, score in zip(tlwhs, track_ids, scores):
                    if track_id < 0:
                        continue
                    x1, y1, w, h = tlwh
                    line = save_format.format(frame=frame_id, id=track_id, x1=round(
                        x1, 1), y1=round(y1, 1), w=round(w, 1), h=round(h, 1), s=round(score, 2))
                    f.write(line)
        logger.info('save results to {}'.format(filename))

    def track(self):

        if not self.args.experiment_name:
            self.args.experiment_name = self.exp.exp_name
        output_dir = osp.join(self.exp.output_dir, self.args.experiment_name)
        os.makedirs(output_dir, exist_ok=True)

        if self.args.save_result:
            vis_folder = osp.join(output_dir, "track_vis")
            os.makedirs(vis_folder, exist_ok=True)

        if self.args.trt:
            self.args.device = "gpu"

        logger.info("Args: {}".format(self.args))

        if self.args.conf is not None:
            self.exp.test_conf = self.args.conf
        if self.args.nms is not None:
            self.exp.nmsthre = self.args.nms
        if self.args.tsize is not None:
            self.exp.test_size = (self.args.tsize, self.args.tsize)

        model = self.exp.get_model().to(self.args.device)

        logger.info("Model Summary: {}".format(
            get_model_info(model, self.exp.test_size)))
        model.eval()

        if not self.args.trt:
            if self.args.ckpt is None:
                ckpt_file = osp.join(output_dir, "best_ckpt.pth.tar")
            else:
                ckpt_file = self.args.ckpt
            logger.info("loading checkpoint")
            ckpt = torch.load(ckpt_file, map_location=torch.device(
                'cuda' if torch.cuda.is_available() else 'cpu'))
            # load the model state dict
            model.load_state_dict(ckpt["model"])
            logger.info("loaded checkpoint done.")

        if self.args.fuse:
            logger.info("\tFusing model...")
            model = fuse_model(model)

        if self.args.fp16:
            model = model.half()  # to FP16

        if self.args.trt:
            assert not self.args.fuse, "TensorRT model is not support model fusing!"
            trt_file = osp.join(output_dir, "model_trt.pth")
            assert osp.exists(
                trt_file
            ), "TensorRT model is not found!\n Run python3 tools/trt.py first!"
            model.head.decode_in_inference = False
            decoder = model.head.decode_outputs
            logger.info("Using TensorRT to inference")
        else:
            trt_file = None
            decoder = None

        predictor = Predictor(model, self.exp, trt_file, decoder,
                              self.args.device, self.args.fp16)

        current_time = time.localtime()

        if self.args.demo == "image":
            self.image_demo(predictor, vis_folder, current_time)
        elif self.args.demo == "video" or self.args.demo == "webcam":
            self.imageflow_demo(predictor, vis_folder, current_time)


class Predictor(object):
    def __init__(
        self,
        model,
        exp,
        trt_file=None,
        decoder=None,
        device=torch.device(torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')),
        fp16=False
    ):
        self.model = model
        self.decoder = decoder
        self.num_classes = exp.num_classes
        self.confthre = exp.test_conf
        self.nmsthre = exp.nmsthre
        self.test_size = exp.test_size
        self.device = device
        self.fp16 = fp16
        self.preproc = ValTransform()
        if trt_file is not None:
            from torch2trt import TRTModule
            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(trt_file))

            x = torch.ones(
                (1, 3, exp.test_size[0], exp.test_size[1]), device=device)
            self.model(x)
            self.model = model_trt
        self.rgb_means = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)

    def inference(self, img, timer):

        img_info = {"id": 0}
        if isinstance(img, str):
            img_info["file_name"] = osp.basename(img)
            img = cv2.imread(img)
        else:
            img_info["file_name"] = None

        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img

        # img, ratio = preproc(img, self.test_size, self.rgb_means, self.std)
        img, ratio = self.preproc(img, None,  self.test_size)
        print(img.shape)

        img_info["ratio"] = ratio

        img = torch.from_numpy(img).unsqueeze(0).float().to(self.device)

        if self.fp16:
            img = img.half()  # to FP16

        with torch.no_grad():
            timer.tic()
            outputs = self.model(img)

            if self.decoder is not None:
                outputs = self.decoder(outputs, dtype=outputs.type())
            outputs = postprocess(
                outputs, self.num_classes, self.confthre, self.nmsthre
            )

            return outputs, img_info


if __name__ == "__main__":
    __package__ = ''
    tracker = Football_bytetrack()
    tracker.track()
