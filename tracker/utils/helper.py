import cv2
from sklearn.cluster import KMeans
import numpy as np
import stitching
from mplsoccer import Pitch, VerticalPitch
from matplotlib.colors import LinearSegmentedColormap
import warnings
warnings.filterwarnings("ignore")


def imgByteToNumpy(img):
    image_byte = img
    img_np = cv2.imdecode(np.fromstring(
        image_byte, np.uint8), cv2.IMREAD_COLOR)
    return img_np


def background_subtraction(img):

    hh, ww = img.shape[:2]

    # threshold on white
    # Define lower and uppper limits
    lower = np.array([55, 100, 50])
    upper = np.array([65, 220, 255])

    # Create mask to only select black
    thresh = cv2.inRange(img, lower, upper)

    # apply morphology
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # invert morp image
    mask = 255 - morph

    result = cv2.bitwise_and(img, img, mask=mask)

    return result


def find_key(dictionary, value):
    for key, val in dictionary.items():
        if val == value:
            return key
    return None


pallete = {
    'green': (0, 128, 0),
    'red': (200, 16, 46),
    'white': (255, 255, 255),
    'blue': (81, 88, 117),

}

color_for_labels = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)


def find_nearest_coordinate(point, coordinates):
    # Calculate Euclidean distances
    distances = np.linalg.norm(coordinates - point, axis=1)
    nearest_index = np.argmin(distances)  # Find index of the minimum distance
    # Get the nearest coordinate
    nearest_coordinate = coordinates[nearest_index]
    return nearest_coordinate, nearest_index


def xyxy_to_xywh(*xyxy):
    """" Calculates the relative bounding box from absolute pixel values. """
    bbox_left = min([xyxy[0].item(), xyxy[2].item()])
    bbox_top = min([xyxy[1].item(), xyxy[3].item()])
    bbox_w = abs(xyxy[0].item() - xyxy[2].item())
    bbox_h = abs(xyxy[1].item() - xyxy[3].item())
    x_c = (bbox_left + bbox_w / 2)
    y_c = (bbox_top + bbox_h / 2)
    w = bbox_w
    h = bbox_h
    return x_c, y_c, w, h


def xyxy_to_tlwh(bbox_xyxy):
    tlwh_bboxs = []
    for i, box in enumerate(bbox_xyxy):
        x1, y1, x2, y2 = [int(i) for i in box]
        top = x1
        left = y1
        w = int(x2 - x1)
        h = int(y2 - y1)
        tlwh_obj = [top, left, w, h]
        tlwh_bboxs.append(tlwh_obj)
    return tlwh_bboxs


def compute_color_for_labels(label):
    """
    Simple function that adds fixed color depending on the class
    """
    color = [int((p * (label ** 2 - label + 1)) % 255)
             for p in color_for_labels]
    return tuple(color)


def draw_boxes(img, bbox, identities=None, offset=(0, 0)):
    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]
        # box text and bar
        id = int(identities[i]) if identities is not None else 0
        color = compute_color_for_labels(id)
        label = '{}{:d}'.format("", id)
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
        cv2.rectangle(
            img, (x1, y1), (x1 + t_size[0] + 3, y1 + t_size[1] + 4), color, -1)
        cv2.putText(img, label, (x1, y1 +
                                 t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 2, [255, 255, 255], 2)
    return img


def transform_matrix(matrix, p, vid_shape, gt_shape):
    p = (p[0]*1280/vid_shape[1], p[1]*720/vid_shape[0])

    px = (matrix[0][0]*p[0] + matrix[0][1]*p[1] + matrix[0][2]) / \
        ((matrix[2][0]*p[0] + matrix[2][1]*p[1] + matrix[2][2]))
    py = (matrix[1][0]*p[0] + matrix[1][1]*p[1] + matrix[1][2]) / \
        ((matrix[2][0]*p[0] + matrix[2][1]*p[1] + matrix[2][2]))

    p_after = (int(px*gt_shape[1]/115), int(py*gt_shape[0]/74))

    return p_after


def background_subtraction(img):

    hh, ww = img.shape[:2]

    # threshold on white
    # Define lower and uppper limits
    lower = np.array([55, 100, 50])
    upper = np.array([65, 220, 255])

    # Create mask to only select black
    thresh = cv2.inRange(img, lower, upper)

    # apply morphology
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # invert morp image
    mask = 255 - morph

    result = cv2.bitwise_and(img, img, mask=mask)

    return result


def detect_color(img):
    # img = background_subtraction(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.reshape((img.shape[1]*img.shape[0], 3))

    kmeans = KMeans(n_clusters=2, n_init=10)
    s = kmeans.fit(img)

    labels = kmeans.labels_
    centroid = kmeans.cluster_centers_
    labels = list(labels)
    percent = []

    for i in range(len(centroid)):
        j = labels.count(i)
        j = j/(len(labels))
        percent.append(j)

    detected_color = centroid[np.argmin(percent)]

    list_of_colors = list(pallete.values())
    assigned_color = closest_color(list_of_colors, detected_color)[0]
    rgb_assigned_color = (int(assigned_color[0]), int(
        assigned_color[1]), int(assigned_color[2]))

    assigned_color = (int(assigned_color[2]), int(
        assigned_color[1]), int(assigned_color[0]))
    if assigned_color == (0, 0, 0):
        assigned_color = (128, 128, 128)
    k = find_key(pallete, rgb_assigned_color)

    return assigned_color, k


# Find the closest color to the detected one based on the predefined palette
def closest_color(list_of_colors, color):
    colors = np.array(list_of_colors)
    color = np.array(color)
    distances = np.sqrt(np.sum((colors-color)**2, axis=1))
    index_of_shortest = np.where(distances == np.amin(distances))
    shortest_distance = colors[index_of_shortest]

    return shortest_distance


def image_stitch(img_src, show=True,   config={"detector": "sift", "confidence_threshold": 0.3, }):
    stitcher = stitching.Stitcher(**config)
    imgOutput = stitcher.stitch(img_src)
    if show:
        cv2.imshow("stitch", imgOutput)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        cv2.waitKey(1)
    return imgOutput


def fourPointOnClick(image_path: str):
    # Load the image
    if type(image_path) == str:
        img = cv2.imread(image_path)
    else:
        img = image_path

    # Create a window to display the image
    cv2.namedWindow('Image')

    # Initialize a list to store the four points
    points = []

    def click_event(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # Append the clicked point to the list
            points.append((x, y))
            # Draw a circle at the clicked point on the image
            cv2.circle(img, (x, y), 5, (0, 255, 0), -1)
            # Display the image with the clicked point
            cv2.imshow('Image', img)

    # Set the mouse callback function to handle mouse events
    cv2.setMouseCallback('Image', click_event)

    # Display the image and wait for the user to click four points
    cv2.imshow('Image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.waitKey(1)

    # Convert the list of points to a NumPy array and return it
    return np.array(points, dtype=np.float32)


def plot_heatmap(x, y, save=False):

    el_greco_yellow_cmap = LinearSegmentedColormap.from_list("El Greco Yellow - 10 colors",
                                                             ['#fae96b', '#d84e3e'])
    pitch = Pitch(pitch_color='grass', line_zorder=2,  line_color='white',
                  pitch_width=120, pitch_length=80, stripe=True)

    fig, ax = pitch.draw(figsize=(4.4, 6.4))
    kdeplot = pitch.kdeplot(x, y, ax=ax, cmap=el_greco_yellow_cmap, fill=True)
    if save:
        fig.savefig(f"heatmap.png", dpi=300)


def process_df(df):

    # Extract values from columns x_center and y_center and put them in a list of lists
    positions = df[['x_center', 'y_center']].values.tolist()
    labels = df["label"].values
    color = df["color"].values

    return positions, labels, color,
