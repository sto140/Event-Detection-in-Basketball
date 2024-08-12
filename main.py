import math


def sd_slow(im):
    "Return the s.d. (standard deviation) of image im"
    ny, nx, nc = im.shape  # Extract properties: height (ny), width (nx), and number of channels(colour) (nc)
    sum = 0  # Initialize sum to zero
    ave = mean(im)  # Compute the mean using the previous 'mean' function

    # Iterate over each pixel (y, x) and each color channel (c) in the image
    for y in range(0, ny):
        for x in range(0, nx):
            for c in range(0, nc):
                v = im[y, x, c] - ave  # Calculate deviation from the mean
                sum = sum + v * v  # represents the squared difference between the mean and sum

    # Calculate the variance (mean of squared differences) and then take the square root to get the standard deviation
    variance = sum / (ny * nx * nc)  # Calculate variance as sum of squared deviations divided by total elements
    standard_deviation = math.sqrt(variance)  # Calculate standard deviation as square root of variance

    return standard_deviation  # Return the computed standard deviation of the image
