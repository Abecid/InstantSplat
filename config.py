"""
Vision CheckOut
"""
import argparse
import os
from types import SimpleNamespace

# PREVIEW_MASKL = (79, 49, 400, 400) #(x,y,w,h)
# PREVIEW_MASKR = (135, 39, 400, 400) #(x,y,w,h)
PREVIEW_MASKL = (0, 0, 640, 480)  # (x,y,w,h)
PREVIEW_MASKR = (0, 0, 640, 480)  # (x,y,w,h)


def tuple_type(strings):
    strings = strings.replace("(", "").replace(")", "")
    mapped_int = map(int, strings.split(","))
    return tuple(mapped_int)


def _to_ns(d):  # tiny helper
    return argparse.Namespace(**d)


def init_cam_configs(valid_camera_view, cam_coords_root=None):
    cam_coords = {}
    for view in valid_camera_view:
        _cam_coord = []
        mask_coords_name = "mask_" + view + ".txt"
        if cam_coords_root:
            cam_coords_path = os.path.join(cam_coords_root, mask_coords_name)
        else:
            cam_coords_path = os.path.join("assets/mask", mask_coords_name)
        try:
            with open(cam_coords_path) as file:
                for line in file:
                    x, y = map(int, line.split())
                    _cam_coord.append([x, y])
            # logger.info(f"Mask file found: {cam_coords_path}")
        except FileNotFoundError:
            print(f"Mask file not found: {cam_coords_path}")
            cam_coords_path = f"assets/mask_{view}.txt"
            try:
                with open(cam_coords_path) as file:
                    print(f"Mask file using: {cam_coords_path}")
                    for line in file:
                        x, y = map(int, line.split())
                        _cam_coord.append([x, y])
            except FileNotFoundError:
                print(f"Mask file not found: {cam_coords_path}")

        cam_coords[view] = _cam_coord
    return cam_coords

def get_config():
    parser = argparse.ArgumentParser(description="Vision checkout configuration")

    parser.add_argument(
        "--app-env",
        default=os.getenv("APP_ENV", "dev"),
        type=str,
        help="App environment (prd, dev), default=environment parameter APP_ENV",
    )
    # TODO: vco-id should be loaded by initial api
    parser.add_argument(
        "--vco-id",
        default="99",
        type=int,
        help="custom VCO_ID",
    )
    parser.add_argument(
        "--company-cd",
        default="FAI",
        type=str,
        help="Company code which will be initilized by station App automatically",
    )
    parser.add_argument(
        "--store-cd",
        default="phase3",
        type=str,
        help="Store code which will be initilized by station App automatically",
    )
    parser.add_argument(
        "--loglevel",
        default="INFO",
        type=str,
        help="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
    )
    parser.add_argument(
        "--logfile",
        default="log/vco.log",
        type=str,
        help="Log file path and filename. default: log/vco.log",
    )
    parser.add_argument(
        "--image-dir",
        default="training_image",
        type=str,
        help="Directory of images for training",
    )
    parser.add_argument(
        "--image-width", default=640, type=int, help="input image width"
    )
    parser.add_argument(
        "--image-height", default=480, type=int, help="input image height"
    )
    parser.add_argument(
        "--sam2-type",
        default="imagepred",
        type=str,
        help="inference_type. imagepred, onlinepred, automask are supported now. if prompt_type is non, automask is only the option",
    )  # use in segmentation
    parser.add_argument(
        "--use-detection-segmentation",
        default="false",
        type=str,
        help="use detection model for segmentation (instead of SAM2 or give point prompts for SAM2). true, false, prompt are supported",
    )
    parser.add_argument(
        "--prompt-type", default="box", type=str, help="none, point or box"
    )  # use in segmentation
    parser.add_argument(
        "--count-start", default=1, type=int, help="countdown to enable capture trigger"
    )  # use only in old pipeline
    """parser.add_argument(
        "--vis-cube", action="store_true", help="check cube with open3d."
    )"""
    """parser.add_argument(
        "--vis3d", action="store_true", help="launch open3d for visualization"
    )  # can remove"""
    parser.add_argument(
        "--cam-type", default="stereo", type=str, help="stereo or mono"
    )  # can remove
    """
    parser.add_argument(
        "--pipeline_type",
        default="main",
        type=str,
        help="main. calibration and capture only deleted.",
    )
    parser.add_argument(
        "--num_cameras", default=7, type=int, help="number of cameras to process"
    )  # can remove
    parser.add_argument(
        "--preview_maskl",
        default=(0, 0, 640, 480),
        type=tuple_type,
        help="mask of plate, left camera",
    )  # can remove
    parser.add_argument(
        "--preview_maskr",
        default=(0, 0, 640, 480),
        type=tuple_type,
        help="mask of plate, right camera",
    )  # can remove"""
    parser.add_argument("--main-cam", default="TB", type=str, help="main camera")
    parser.add_argument("--stream-cam", default="TC", type=str, help="stream camera")
    parser.add_argument("--calib", action="store_true", help="online calibration")
    parser.add_argument("--cam-key", default="TB", help="camera key for mask")
    # parser.add_argument("--machine-1", action="store_true", help="use machine1")
    parser.add_argument(
        "--debug-mode",
        action="store_true",
        help="Debug mode including Output_data store",
    )
    parser.add_argument(
        "--intrinsic-path",
        default="./assets/mask/intrinsic.json",
        type=str,
        help="camera intrinsic file path",
    )
    parser.add_argument(
        "--stereo-path",
        default="./assets/mask/stereo_config_online.json",
        type=str,
        help="stereo config file path",
    )
    # info_file_path = "modules/assets/cam_info.json"
    parser.add_argument(
        "--cam-info-path",
        default="./modules/assets/cam_info.json",
        type=str,
        help="cam info file path",
    )
    parser.add_argument("--log-root", default="", type=str, help="log file path")
    parser.add_argument(
        "--image-prefixs",
        default=[],
        nargs="+",
        type=str,
        help="specific log files path",
    )  # use in log_viewer and evaluation
    parser.add_argument(
        "--back-test", action="store_true", help="process images on log viewer"
    )  # use in log_viewer
    parser.add_argument(
        "--hand-warning-threshold",
        default=40,
        type=int,
        help="How many frames to hand warning trigger",
    )
    parser.add_argument(
        "--stable-threshold",
        default=5,
        type=int,
        help="How many frames to capture trigger",
    )
    parser.add_argument(
        "--idle-check-counter-period",
        default=300,
        type=int,
        help=(
            "How many frames(without any change detection) to check whether "
            "still in idle state or not"
        ),
    )
    parser.add_argument(
        "--video-record",
        default=False,
        type=bool,
        help="Whether record the video or not",
    )
    parser.add_argument("--cache-root", default="", type=str, help="cache data root")
    parser.add_argument(
        "--is-save-cache",
        action="store_true",
        help=(
            "save cache of evaluation. When save cache, cache-root argument must be set"
        ),
    )
    parser.add_argument(
        "--use-lighterglue",
        action="store_true",
        help="Use ligheterglue for denser feature extraction.",
    )
    parser.add_argument(
        "--show-display",
        "-d",
        nargs="+",
        help=(
            "Show display for debug. can select multiple modules to show in "
            "(pipeline, stable_detector, hand_detector, object_detector, "
            "segmentation, multiview_object_matching, stereo_depth, "
            "classification, 3d, cube)"
        ),
    )
    parser.add_argument(
        "--is-save-cls-images",
        action="store_true",
        help="save images for classification input (for debugging).",
    )
    parser.add_argument(
        "--is-save-bbox-images",
        action="store_true",
        help="save bbox images (for debugging).",
    )
    parser.add_argument(
        "--is-save-feature-images",
        action="store_true",
        help="save feature images (for debugging).",
    )
    parser.add_argument(
        "--is-save-segment-images",
        action="store_true",
        help="save segment images (for debugging).",
    )
    parser.add_argument(
        "--is-save-mast3r",
        action="store_true",
        help="save mast3r matching and 3D (for debugging).",
    )
    parser.add_argument(
        "--is-save-intermediate-images",
        action="store_true",
        help="save all intermediate images: bbox, feature, segment, classification (for debugging).",
    )
    parser.add_argument(
        "--eval-ver",
        default=1,
        type=int,
        help="Evaluation dataset version: [1, 2]",
    )
    parser.add_argument(
        "--object-detector",
        default="yolo",
        type=str,
        help="Object detection model: [yolo, dino, dfine]",
    )
    parser.add_argument(
        "--video-save-path",
        default="./log/video_log",
        type=str,
        help="video file path",
    )
    parser.add_argument(
        "--capture-image-path",
        default="./log/capture_images",
        type=str,
        help="capture image path",
    )
    parser.add_argument(
        "--is-save-capture-images",
        action="store_true",
        help="save capture images",
    )
    parser.add_argument(
        "--is-visualize-point-cloud",
        action="store_true",
        help="save capture images",
    )

    parser.add_argument(
        "--is-save-all-images",
        action="store_true",
        help="save all images",
    )

    parser.add_argument(
        "--dataset-name",
        default="vco",
        type=str,
        help="Name of the dataset for gathering",
    )
    parser.add_argument(
        "--num-top-k",
        default=1,
        type=int,
        help="Number of top k for classification",
    )
    parser.add_argument(
        "--use-detection-classification",
        action="store_true",
        help="use detection classification score",
    )

    parser.add_argument(
        "--is-use-mast3r",
        action="store_true",
        help="Use MAST3R for matching and 3D reconstruction.",
    )

    parser.add_argument(
        "--is-use-graph",
        action="store_true",
        help="Use networkx for association",
    )

    # Depth filtering arguments
    parser.add_argument(
        "--enable-depth-filtering",
        action="store_true",
        default=True,
        help="Enable depth filtering to remove floor points (default: True)",
    )

    parser.add_argument(
        "--disable-depth-filtering",
        action="store_true",
        help="Disable depth filtering",
    )

    parser.add_argument(
        "--depth-threshold",
        type=float,
        default=20.0,
        help="Z-coordinate threshold for depth filtering in mm (default: 50.0)",
    )

    parser.add_argument(
        "--depth-min-ratio",
        type=float,
        default=0.2,
        help="Minimum ratio of points to keep after filtering (default: 0.2)",
    )

    parser.add_argument(
        "--empty-board-path",
        default="./assets/mask/empty_board/",
        type=str,
        help="empty board images directory path",
    )
    args = parser.parse_args()

    # Handle depth filtering enable/disable logic
    if args.disable_depth_filtering:
        args.enable_depth_filtering = False

    if args.show_display:
        for item in args.show_display:
            if item not in [
                "pipeline",
                "stable_detector",
                "hand_detector",
                "segmentation",
                "object_detector",
                "multiview_object_matching",
                "stereo_depth",
                "classification",
                "3d",
                "cube",
            ]:
                raise Exception(
                    "Wrong input for --show-display/-d value. Please check usage. $ python main.py -h"
                )
    else:
        args.show_display = []

    return args


def make_vco_args(overrides: dict | None = None, inherit_from: argparse.Namespace | None = None) -> argparse.Namespace:
    """
    Build an argparse.Namespace that matches VCO's get_config() output,
    using VCO defaults and applying optional overrides and/or values taken
    from an existing args (e.g., your InstantSplat args).
    """
    VCO_DEFAULTS = {
        "app_env": "dev",
        "vco_id": 99,
        "company_cd": "FAI",
        "store_cd": "phase3",
        "loglevel": "INFO",
        "logfile": "log/vco.log",
        "image_dir": "training_image",
        "image_width": 640,
        "image_height": 480,
        "sam2_type": "imagepred",
        "use_detection_segmentation": "false",  # "true" | "false" | "prompt"
        "prompt_type": "box",                   # "none" | "point" | "box"
        "count_start": 1,
        "cam_type": "stereo",
        "main_cam": "TB",
        "stream_cam": "TC",
        "calib": False,
        "cam_key": "TB",
        "debug_mode": False,
        "intrinsic_path": "./assets/mask/intrinsic.json",
        "stereo_path": "./assets/mask/stereo_config_online.json",
        "cam_info_path": "./modules/assets/cam_info.json",
        "log_root": "",
        "image_prefixs": [],
        "back_test": False,
        "hand_warning_threshold": 40,
        "stable_threshold": 5,
        "idle_check_counter_period": 300,
        "video_record": False,
        "cache_root": "",
        "is_save_cache": False,
        "use_lighterglue": False,
        "show_display": [],  # e.g. ["pipeline","segmentation","3d"]
        "is_save_cls_images": False,
        "is_save_bbox_images": False,
        "is_save_feature_images": False,
        "is_save_segment_images": False,
        "is_save_mast3r": False,
        "is_save_intermediate_images": False,
        "eval_ver": 1,
        "object_detector": "yolo",  # ["yolo","dino","dfine"] (you mentioned dfine)
        "video_save_path": "./log/video_log",
        "capture_image_path": "./log/capture_images",
        "is_save_capture_images": False,
        "is_visualize_point_cloud": False,
        "is_save_all_images": False,
        "dataset_name": "vco",
        "num_top_k": 1,
        "use_detection_classification": False,
        "is_use_mast3r": False,
        "is_use_graph": False,

        # Depth filtering
        "enable_depth_filtering": True,
        "disable_depth_filtering": False,
        "depth_threshold": 20.0,   # mm
        "depth_min_ratio": 0.2,

        "empty_board_path": "./assets/mask/empty_board/",
    }

    # Start with defaults
    out = VCO_DEFAULTS.copy()

    # If you want to inherit same-named fields from an existing args (e.g., InstantSplat args)
    if inherit_from is not None:
        for k in out.keys():
            if hasattr(inherit_from, k) and getattr(inherit_from, k) is not None:
                out[k] = getattr(inherit_from, k)

    # Apply explicit overrides (e.g., from your launch.json)
    if overrides:
        for k, v in overrides.items():
            if k in out:
                out[k] = v

    # Post-processing to mirror VCO’s boolean logic
    if out.get("disable_depth_filtering"):
        out["enable_depth_filtering"] = False

    # Validate show_display entries if provided (optional strictness)
    valid_show = {
        "pipeline", "stable_detector", "hand_detector", "segmentation",
        "object_detector", "multiview_object_matching", "stereo_depth",
        "classification", "3d", "cube"
    }
    sd = out.get("show_display") or []
    if isinstance(sd, str):
        sd = [sd]
    out["show_display"] = [x for x in sd if x in valid_show]

    return _to_ns(out)
