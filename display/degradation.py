import cv2
import numpy as np


def degradate_image(path: str):
    img = cv2.imread(path)
    names = ["median", "bright", "dark", "motionV", "motionH", "gaussian", "contrast"]
    kernels = []
    kernels.append(np.ones((7, 7), np.float32) / 49)
    kernels.append(np.ones((3, 3), np.float32) / 5)
    kernels.append(np.ones((3, 3), np.float32) / 15)
    kernels.append(np.array([[1 / 9, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 1 / 9, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 1 / 9, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 1 / 9, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 1 / 9, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 1 / 9, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 1 / 9, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 1 / 9, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 1 / 9]]))
    kernels.append(np.array([[1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9]]))
    kernels.append(cv2.getGaussianKernel(7, 2))
    kernels.append(np.array([[0, -1, 0],
 [-1, 5, -1],
 [0, -1, 0]]))

    for name, kernel in zip(names, kernels):
        filtered_img = cv2.filter2D(img, -1, kernel)
        cv2.imwrite(f"{name}.png", filtered_img)

    for i in range(10):
        kernel = np.random.uniform(-4, 4, size=(7, 7))
        kernel = kernel / np.sum(kernel)
        filtered_img = cv2.filter2D(img, -1, kernel)
        cv2.imwrite(f"random{i}.png", filtered_img)


def get_residual(path1: str, path2: str):
    img1 = cv2.imread(path1).astype(np.float32)
    img2 = cv2.imread(path2).astype(np.float32)
    img21 = (img2 - img1)
    img21 = np.clip((img21 - img21.min()) / (img21.max() - img21.min()) * 255., 0, 255).astype(np.uint8)
    cv2.imwrite(f"img21.png", img21)


if __name__ == "__main__":
    path = r"D:\Code\basicvsr\BasicVSR_PlusPlus\data\VIMEO\train\00011\0207\im1.png"
    get_residual(r"D:\Code\basicvsr\BasicVSR_PlusPlus\data\VIMEO\train\00011\0207\im3.png",
                 r"D:\Code\basicvsr\BasicVSR_PlusPlus\data\VIMEO\train\00011\0207\im4.png")
