from argparse import Namespace
import os
import os.path as osp
import time
import cv2
import torch
from tracker.tracker_config import default_args
from loguru import logger
import numpy as np
from yolox.exp import get_exp
from yolox.utils import fuse_model, get_model_info, postprocess
from yolox.utils.visualize import plot_tracking
from yolox.tracker.byte_tracker import BYTETracker
from yolox.tracking_utils.timer import Timer
import matplotlib.pyplot as plt
from tracker.utils.helper import detect_color
import torch
from tracker.utils.predictor import Predictor

IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]


class TrackerBytetrack():
    def __init__(self, args=None) -> None:
        if args != None:
            print(default_args)
            default_args.update(args)
            self.args = self.load_argument(default_args)
        else:
            self.args = self.load_argument(default_args)

        self.exp = get_exp(self.args.exp_file, self.args.name)
        self.initialize_config()

    def load_argument(self, kwargs):
        args = Namespace(**kwargs)
        return args

    def initialize_config(self):

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

    def imageflow_demo(self, predictor, vis_folder, current_time, show=False):
        cap = cv2.VideoCapture(
            self.args.path if self.args.demo == "video" else self.args.camid)

        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float

        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        fps = cap.get(cv2.CAP_PROP_FPS)
        timestamp = time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
        save_folder = osp.join(vis_folder, timestamp)
        os.makedirs(save_folder, exist_ok=True)

        if self.args.demo == "video":
            save_path = osp.join(save_folder, self.args.path.split("/")[-1])
        else:
            save_path = osp.join(save_folder, "camera.mp4")

        vid_writer = cv2.VideoWriter(
            save_path, cv2.VideoWriter_fourcc(
                *"mp4v"), fps, (int(width), int(height)))
        logger.info(f"video save_path is {save_path}")

        tracker = BYTETracker(self.args, frame_rate=self.args.fps)
        timer = Timer()
        frame_id = 0
        results = []
        results.append(
            f"frame_id,id,x,y,w,h,score,label,x_center, y_center,color,")

        while True:

            if cv2.waitKey(20) & 0xff == ord('q'):
                cv2.destroyAllWindows()
                break

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
                        size = tlwh[2] * tlwh[3]

                        if size > self.args.min_box_area and size < self.args.max_box_area:
                            online_tlwhs.append(tlwh)
                            online_ids.append(tid)
                            online_scores.append(t.score)

                            x, y, w, h = np.maximum(
                                0, [int(tlwh[0]), int(tlwh[2]), int(tlwh[3]), int(tlwh[3])])
                            crop = main_frame[y: y+h, x: x+w]

                            if (len(crop) == 0):
                                continue
                            x_center, y_center = x + w // 2, y + h // 2
                            _, key = detect_color(crop)
                            results.append(
                                f"{frame_id},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},{t.score:.2f},{t.label},{x_center},{y_center},{key}\n"
                            )

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

            if show:
                cv2.imshow("detection", online_im)

        if self.args.save_result:
            res_file = osp.join(vis_folder, f"{timestamp}.txt")
            with open(res_file, 'w') as f:
                f.writelines(results)
            logger.info(f"save results to {res_file}")

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

    def track(self, show=False):

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

        if self.args.demo == "video" or self.args.demo == "webcam":
            self.imageflow_demo(predictor, vis_folder, current_time, show=show)


if __name__ == "__main__":
    __package__ = ''
    tracker = TrackerBytetrack()
    tracker.track()
