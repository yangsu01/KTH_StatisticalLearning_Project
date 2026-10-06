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
            mode = "symmetric"
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

    def _forward_haar(self, x):
        a = x[0::2]
        b = x[1::2]

        L = (a + b) / np.sqrt(2)
        H = (a - b) / np.sqrt(2)

        return np.concatenate((L, H))

    def _inverse_haar(self, coefficients):
        # split the coefficients into low and high components
        split = len(coefficients) // 2
        L = coefficients[:split]
        H = coefficients[split:]

        a = (L + H) / np.sqrt(2)
        b = (L - H) / np.sqrt(2)

        x = np.empty(len(coefficients), dtype=np.float64)
        x[0::2] = a # even values
        x[1::2] = b # odd values

        return x

    def _forward_haar_2d(self, img):
        # generates a 2D haar matrix with LL, LH, HL, HH blocks
        temp = np.apply_along_axis(
            self._forward_haar, 
            axis = 0, 
            arr = img
        )

        return np.apply_along_axis(
            self._forward_haar, 
            axis = 1, 
            arr = temp
        )

    def _inverse_haar_2d(self, coefficients):
        temp = np.apply_along_axis(
            self._inverse_haar, 
            axis = 0, 
            arr = coefficients
        )

        return np.apply_along_axis(
            self._inverse_haar, 
            axis = 1, 
            arr = temp
        )

    def apply_W(self, x: np.ndarray, levels: int=3) -> np.ndarray:
        """Applies inverse multi-stage haar transform to the vector of wavelet coefficients
        
        Args:
            x (np.ndarray): Vector of wavelet coefficients
            levels (int, optional): Number of stages for the haar transform. Defaults to 3.
        
        Returns:
            np.ndarray: image after applying the inverse multi-stage haar transform
        """
        coefficients = self._reshape_img(x)

        res = coefficients.copy()
        h, w = res.shape

        # go in reverse order to reconstuct image
        for levels in range(levels-1, -1, -1):
            x = h // (2**levels)
            y = w // (2**levels)

            res[:x, :y] = self._inverse_haar_2d(res[:x, :y])

        return res

    def apply_W_T(self, img: np.ndarray, levels: int=3) -> np.ndarray:
        """Applies multi-stage haar transform to the image
        
        Args:
            img (np.ndarray): flat image array
            levels (int, optional): Number of stages for the haar transform. Defaults to 3.
        
        Returns:
            np.ndarray: wavelet coefficients after applying the multi-stage haar transform
        """
        res = img.copy()
        h, w = img.shape

        for _ in range(levels):
            res[:h, :w] = self._forward_haar_2d(res[:h, :w])

            h //= 2
            w //= 2

        return res

    def apply_R(self, img: np.ndarray, kernel_size: int=9, sigma: float=4.0) -> np.ndarray:
        """Applies the blur to the image 
        
        Args:
            img (np.ndarray): flat image array
            kernel_size (int, optional): Size of the Gaussian kernel. Defaults to 9.
            sigma (float, optional): Standard deviation of the Gaussian kernel. Defaults to 4.0.
        
        Returns:
            np.ndarray: image after applying the blur operator R
        """
        return self.gaussian_blur(img, kernel_size=kernel_size, sigma=sigma)

    def apply_A(self, x: np.ndarray, kernel_size: int=9, sigma: float=4.0, levels: int=3) -> np.ndarray:
        """Applies the A = RW matrix to the flat image
        
        Args:
            x (np.ndarray): flat image array
            kernel_size (int, optional): Size of the Gaussian kernel. Defaults to 9.
            sigma (float, optional): Standard deviation of the Gaussian kernel. Defaults to 4.0.
            levels (int, optional): Number of stages for the haar transform. Defaults to 3.
        
        Returns:
            np.ndarray: wavelet coefficients after applying the blur operator R and the multi-stage haar transform W
        """
        img = self.apply_W(x, levels=levels)
        blur_img = self.apply_R(img, kernel_size=kernel_size, sigma=sigma)

        return self._flatten_img(blur_img)

    def apply_A_T(self, x: np.ndarray, kernel_size: int=9, sigma: float=4.0, levels: int=3) -> np.ndarray:
        """Applies the A^T = W^T R^T matrix to the flat image
        
        Args:
            x (np.ndarray): flat image array
            kernel_size (int, optional): Size of the Gaussian kernel. Defaults to 9.
            sigma (float, optional): Standard deviation of the Gaussian kernel. Defaults to 4.0.
            levels (int, optional): Number of stages for the haar transform. Defaults to 3.
        
        Returns:
            np.ndarray: wavelet coefficients after applying the transpose of the blur operator R and the multi-stage haar transform W
        """
        img = self.apply_R(x, kernel_size=kernel_size, sigma=sigma)
        haar_img = self.apply_W_T(img, levels=levels)

        return self._flatten_img(haar_img)

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

    # apply A matrix ie A = RW
    A = processor.apply_A(flattened_img, kernel_size=9, sigma=4.0, levels=3)

    # apply A^T matrix ie A^T = W^T R^T
    AT = processor.apply_A_T(flattened_img, kernel_size=9, sigma=4.0, levels=3)

    # apply W matrix ie inverse haar transform
    W = processor.apply_W(flattened_img, levels=3)

    # apply W^T matrix ie haar transform. NOTE input should be a 2d image
    WT = processor.apply_W_T(processor._reshape_img(flattened_img), levels=3)

    # apply R matrix and RT matrix are the same (gaussian blur)
    R = processor.apply_R(flattened_img, kernel_size=9, sigma=4.0)