import numpy as np
from matplotlib import pyplot as plt
from PIL import Image

class ImageProcessor:
    def __init__(self, image_path: str):
        self.image_path = image_path
        self.original_shape = None

    def load_image(self, gray_scale: bool=True) -> np.ndarray:
        """Loads an image from the specified path and converts it to grayscale.

        Args:
            gray_scale (bool, optional): whether to convert image to grayscale. Defaults to True

        Returns:
            np.ndarray: flattened image as a numpy array
        """
        img = Image.open(self.image_path)
        img = img.convert("L") if gray_scale else img

        img_arr = np.array(img, dtype=np.float64)
        img_arr = img_arr / 255.0

        self.original_shape = img_arr.shape

        return self._flatten_img(img_arr)

    def gaussian_blur(self, flat_img: np.ndarray, kernel_size: int=9, sigma: float=4.0) -> np.ndarray:
        """Adds a Gaussian blur to a flattened image array
        
        Args:
            flat_img (np.ndarray): flattened image array
            kernel_size (int, optional): size of square grid used for gaussian convolution. Should be an odd number. Defaults to 9
            sigma (float, optional): standard deviation of gaussian distribution. Defaults to 4

        Returns:
            np.ndarray: flattened image after gaussian blur
        """
        img = self._reshape_img(flat_img)

        # gaussian kervel
        kernel = self._gaussian_kernel(kernel_size, sigma)
        kernel = kernel[::-1, ::-1] # flip kernel

        # pad image with relexive boundary conditions
        padded_img = np.pad(
            img, 
            pad_width = kernel_size // 2,
            mode = "reflect"
        )

        # convolution
        output = np.zeros_like(img, dtype=float)

        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                conv_region = padded_img[i:i+kernel_size, j:j+kernel_size]
                output[i,j] = np.sum(conv_region*kernel)

        return output

    def add_noise(self, flat_img: np.ndarray, sigma: float=0.003) -> np.ndarray:
        """Adds random gaussian noise to flattened image

            Args:
                flat_img (np.ndarray): array of flattened image
                sigma (float, optional): standard deviation of normal distribution. Defaults to 0.003

            Returns:
                np.ndarray: flattened image with added noise  
        """
        noise = np.random.normal(
            loc = 0,
            scale = sigma,
            size = self.original_shape
        )

        img = self._reshape_img(flat_img) + noise

        return self._flatten_img(img)

    def save_img(self, flat_img: np.ndarray, path: str):
        """Saves image
        
        Args:
            flat_img (np.ndarray): flattened image array
            path (str): save path
        """
        img = self._reshape_img(flat_img)
        img = np.clip(img, 0, 1)
        img = (img * 255).astype(np.uint8) # rescale to [0, 255] and convert to uint8

        Image.fromarray(img).save(path)
        
    def _flatten_img(self, img):
        return img.flatten(order="F")

    def _gaussian_kernel(self, kernel_size, sigma):
        # create grid centered at 0
        k = kernel_size // 2
        x = np.arange(-k, k+1)
        X,Y = np.meshgrid(x, x)

        # gaussian kernel
        kernel = np.exp(-(X**2 + Y**2) / (2*sigma**2))
        kernel /= np.sum(kernel) # normalize such that sum(.) = 1

        return kernel

    def _reshape_img(self, img_arr):
        return img_arr.reshape(self.original_shape, order="F")



# example usage
if __name__ == "__main__":
    IMAGE_PATH = "data/cameraman.tif"
    processor = ImageProcessor(IMAGE_PATH) # initiate image processor
    flattened_img = processor.load_image() # load and flatten image

    # add gaussian blur
    blurred_img = processor.gaussian_blur(flattened_img, kernel_size=9, sigma=4.0)

    # add noise
    noisy_img = processor.add_noise(blurred_img, sigma=0.003)

    # save images from flattened arrays
    processor.save_img(blurred_img, "data/blurred_image.png")
    processor.save_img(noisy_img, "data/noisy_image.png")


