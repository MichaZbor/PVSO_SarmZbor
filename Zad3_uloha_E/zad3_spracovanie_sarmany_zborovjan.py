import time
import matplotlib.pyplot as plt
import numpy as np
import cv2

# 1. Vytvorenie Gaussovho jadra
def gaussian_kernel(size, sigma):
    """Vytvorí 2D Gaussov filter manuálne."""
    ax = np.linspace(-(size // 2), size // 2, size)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    return kernel / kernel.sum()  # Normalizácia

# 2. Výpočet Difference of Gaussians (DoG)
def difference_of_gaussians(size, sigma1, sigma2):
    """Vypočíta rozdiel dvoch Gaussových filtrov (DoG)."""
    g1 = gaussian_kernel(size, sigma1)
    g2 = gaussian_kernel(size, sigma2)
    return g1 - g2  # Rozdiel Gaussových filtrov

# 3. Ručne aplikovaná 2D konvolúcia
def convolve2d(image, kernel):
    """Aplikuje 2D konvolúciu na obrázok bez použitia np.pad."""
    start_time = time.time()
    kh, kw = kernel.shape
    ih, iw = image.shape

    pad_h, pad_w = kh // 2, kw // 2  # Určuje, o koľko sa zmenší výstup

    # Nové rozmery výstupu bez okrajov
    output_h, output_w = ih - 2 * pad_h, iw - 2 * pad_w
    output = np.zeros((output_h, output_w))

    # Aplikácia konvolúcie len na vnútorné pixely
    for i in range(pad_h, ih - pad_h):
        for j in range(pad_w, iw - pad_w):
            region = image[i - pad_h: i + pad_h + 1, j - pad_w: j + pad_w + 1]
            output[i - pad_h, j - pad_w] = np.sum(region * kernel)

    print(f"Elapsed time: {time.time() - start_time}")
    return output

def imread_grayscale(image):
    if image is not None and len(image.shape) == 3:
        height, width, _ = image.shape
        grayscale_image = np.zeros((height, width), dtype=np.uint8)

        for y in range(height):
            for x in range(width):
                b, g, r = image[y, x, 0:3]
                grayscale_image[y, x] = int(0.299 * r + 0.587 * g + 0.114 * b)
    else:
        # Ak je už obrázok grayscale, prekonvertujeme ho na numpy.ndarray
        grayscale_image = np.array(image, dtype=np.uint8)

    return grayscale_image

if __name__ == '__main__':
    # 4. Načítanie obrázka v odtieňoch sivej
    image1 = cv2.imread("obrazok.jpg")
    image = cv2.imread("obrazok.jpg")
    image = imread_grayscale(image)

    # Nastavenie hodnôt sigma
    sigma1, sigma2 = 4, 8
    size = 5  # Veľkosť jadra filtra

    # Výpočet DoG filtra
    dog_kernel = difference_of_gaussians(size, sigma1, sigma2)

    # Aplikácia DoG filtra na obrázok
    dog_image = convolve2d(image, dog_kernel)
    
    # 5. Zobrazenie výsledkov pomocou OpenCV
    image = cv2.resize(image, (360, 360))
    dog_image = cv2.resize(dog_image, (360, 360))
    cv2.imshow("Pôvodný obrázok", image)
    cv2.imshow("DoG aplikovaný na obrázok", cv2.normalize(dog_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8))

    # Inicializácia počítadiel pre B, G, R kanály
    intensity_bins = 26
    color_histogram = np.zeros((intensity_bins, 3), dtype=int)

    # Spracovanie obrazu s rozmermi 240x240
    for y in range(240):
        for x in range(240):
            pixel = image1[y, x]  # pixel = [B, G, R]
            for channel in range(3):  # pre každý farebný kanál
                value = pixel[channel]
                bin_index = min(value // 10, intensity_bins - 1)
                color_histogram[bin_index, channel] += 1

    # Vykreslenie histogramu farebných intenzít
    plt.figure(figsize=(10, 5))
    colors = ['red', 'green', 'blue']
    labels = ['Red', 'Green', 'Blue']

    for i in range(3):
        plt.plot(color_histogram[:, i], color=colors[i], label=labels[i])

    plt.title("Distribúcia intenzít RGB farieb v obrázku (240x240)")
    plt.xlabel("Interval intenzity (10 pixelov)")
    plt.ylabel("Pixely")
    plt.legend()
    plt.grid()
    plt.show()

    # Vykreslenie histogramu farebných intenzít
    cv2.waitKey()
    cv2.destroyAllWindows()
