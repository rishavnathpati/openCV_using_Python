import matplotlib.pyplot as plt
import matplotlib.image as mpltimg
import numpy as np


# _______ 3 user defined functions which you have to code their functionality_________


def convolution2D(img, kernel, padding_type):
    # write your code here
    # padding_type can take values 0, 1 or 2
    # 0 - zero padding
    # 1 - duplicate boundary pixels for padding
    # 2 - padding is done by mirroring the pixels

    # should handle kernel of any size but odd values only eg. 5x5, 7x7
    # image is a grayscale image

    filt_image = img  # dummy assignment (to be removed)
    return filt_image


def medianFiltering(img, kernel_size, padding_type):
    # write your code here
    # padding_type can take values 0, 1 or 2
    # 0 - zero padding
    # 1 - duplicate boundary pixels for padding
    # 2 - padding is done by mirroring the pixels

    # should handle kernel of any size but odd values only eg. 5x5, 7x7
    # image is a grayscale image

    med_image = img  # dummy assignment (to be removed)
    return med_image


# You can club the above two functions/write any other functions additionally if you wish

def computePSNR(image1, image2):
    psnr = 0

    return psnr


# _____________________main program begins here___________________

def main():
    # reading a noisy image
    noisy_image = mpltimg.imread('images/noisy_image.jpg')

    original_image = mpltimg.imread('images/original.jpg')


# _____________________________________________________________________
# Average filter kernel
    kernel = 1/9 * np.array([[1, 1, 1],
                             [1, 1, 1],
                             [1, 1, 1]])

    low_pass_filtered_image = convolution2D(noisy_image, kernel, 1)

    avg_psnr = computePSNR(original_image, low_pass_filtered_image)

    med_filtered_image = medianFiltering(noisy_image, 4, 1)

    med_psnr = computePSNR(original_image, med_filtered_image)


# _____________________________________________________________________
# reading a blurry image
    blurry_image = mpltimg.imread('images/blurry_image.jpg')

    # Laplacian filter kernel
    kernel = np.array([[1, 1, 1],
                       [1, -8, 1],
                       [1, 1, 1]])

    laplacian_filtered_image = convolution2D(blurry_image, kernel, 1)

# perform the addition as in Eqn. 3.6.7 to obtain the sharpened image

    sharpened_image = blurry_image  # dummy assignment (to be removed)


# _____________________________________________________________________
# Code to display the images

    fig, axes = plt.subplots(nrows=2, ncols=3)

    ax = axes.ravel()

    ax[0].imshow(noisy_image, cmap='gray')
    ax[0].set_title("Noisy image")
    ax[0].set_axis_off()

    ax[1].imshow(low_pass_filtered_image, cmap='gray')
    ax[1].set_title("Low Pass Filter Output")
    ax[1].set_axis_off()
    ax[1].text(x=40, y=230, s="PSNR = %1.2f db" % avg_psnr)

    ax[2].imshow(med_filtered_image, cmap='gray')
    ax[2].set_title("Median Filter Output")
    ax[2].set_axis_off()
    ax[2].text(x=40, y=230, s="PSNR = %1.2f db" % med_psnr)

    ax[3].imshow(blurry_image, cmap='gray')
    ax[3].set_title("Blurry Input Image")
    ax[3].set_axis_off()

    ax[4].imshow(laplacian_filtered_image, cmap='gray')
    ax[4].set_title("Laplacian Filter Output")
    ax[4].set_axis_off()

    ax[5].imshow(sharpened_image, cmap='gray')
    ax[5].set_title("Sharpened Image")
    ax[5].set_axis_off()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
