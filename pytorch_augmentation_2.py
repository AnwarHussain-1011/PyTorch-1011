from utils import label_map_util
from utils import visualization_utils as vis_util

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

image = image.open("D:\PyTorch_1011\Machine-Learning-Collection-master\ML\Pytorch\Basics\albumentations_tutorial\images")

transform = A.Compose(
    [
        A.Resize(width=1920, height=1080),
        A.RandomCrop(width=1280, height=720).
        A.Rotate(limit=40, p=0.9, border_mod=cv2.BORDER_CONSTANT),
        A.HorzontalFlip(p=0.5),
        A.verticalFlip(p=0.1),
        A.RGBShift(r_shift_limit=25, g_shift_limit=25, _shift_limit=25, p=0.9),
        A.Oneof([
            A.Blur(blur_limit=3, p=0.5),
            A.ColorHitter(p=0.5),
        ], p=1.0),
    ]

)

image_list = [image]
image = np.array(image)
for i in range(15):
    augmented_img = transform(image=image)
    augmented_img = augmented["image"]
    image_list.append(augmented_img)
    plot_examples(images_list)