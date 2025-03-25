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
    """Aplikuje 2D konvolúciu na obrázok."""
    kh, kw = kernel.shape
    ih, iw = image.shape

    pad_h, pad_w = kh // 2, kw // 2
    image_padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='reflect')

    output = np.zeros((ih, iw))
    for i in range(ih):
        for j in range(iw):
            region = image_padded[i:i+kh, j:j+kw]
            output[i, j] = np.sum(region * kernel)

    return output

# 4. Načítanie obrázka v odtieňoch sivej
image = cv2.imread("obrazok.jpg", cv2.IMREAD_GRAYSCALE)

if image is None:
    raise ValueError("Obrázok sa nepodarilo načítať. Skontroluj cestu k súboru.")

# Nastavenie hodnôt sigma
sigma1, sigma2 = 1, 2
size = 13  # Veľkosť jadra filtra

# Výpočet DoG filtra
dog_kernel = difference_of_gaussians(size, sigma1, sigma2)

# Aplikácia DoG filtra na obrázok
dog_image = convolve2d(image, dog_kernel)

# 5. Zobrazenie výsledkov pomocou OpenCV
cv2.imshow("Pôvodný obrázok", image)
cv2.imshow("DoG aplikovaný na obrázok", cv2.normalize(dog_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8))

cv2.waitKey( )
cv2.destroyAllWindows()


