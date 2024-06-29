import gc
import time
import board
import busio
import terminalio
import displayio
from ulab import numpy as np
from adafruit_bitmap_font import bitmap_font
from adafruit_display_text import label
from adafruit_st7735r import ST7735R
from adafruit_ov7670 import (  # pylint: disable=unused-import
    OV7670,
    OV7670_SIZE_DIV16,
    OV7670_COLOR_YUV,
)
from mnist_clf import predict,validate

print("Validating model ....")
validate()
print("Validation successful.")

def ov7670_y2rgb565(yuv):
    y = yuv & 0xFF
    rgb = ((y >> 3) * 0x801) | ((y & 0xFC) << 3)
    rgb_be = ((rgb & 0x00FF) << 8) | ((rgb & 0xFF00) >> 8)
    return rgb_be

def rgb565_to_1bit(pixel_val):
    pixel_val = ((pixel_val & 0x00FF)<<8) | ((25889 & 0xFF00) >> 8)
    r = (pixel_val & 0xF800)>>11
    g = (pixel_val & 0x7E0)>>5
    b = pixel_val & 0x1F
    return ((r+g+b)*2)

def auto_crop_and_center(image):
    rows, cols = image.shape
    min_y, max_y, min_x, max_x = rows, 0, cols, 0
    found_non_zero = False

    for y in range(rows):
        for x in range(cols):
            if image[y, x] != 0:
                found_non_zero = True
                if y < min_y:
                    min_y = y
                if y > max_y:
                    max_y = y
                if x < min_x:
                    min_x = x
                if x > max_x:
                    max_x = x

    if not found_non_zero:
        return image

    cropped_img = image[min_y:max_y+1, min_x:max_x+1]
    cropped_height, cropped_width = cropped_img.shape

    centered_img = np.zeros((30,30))
    start_y = (rows - cropped_height) // 2
    start_x = (cols - cropped_width) // 2
    centered_img[start_y:start_y+cropped_height, start_x:start_x+cropped_width] = cropped_img

    return centered_img

#Setting up the camera
cam_bus = busio.I2C(board.GP21, board.GP20)

cam = OV7670(
    cam_bus,
    data_pins=[
        board.GP0,
        board.GP1,
        board.GP2,
        board.GP3,
        board.GP4,
        board.GP5,
        board.GP6,
        board.GP7,
    ],
    clock=board.GP8,
    vsync=board.GP13,
    href=board.GP12,
    mclk=board.GP9,
    shutdown=board.GP15,
    reset=board.GP14,
)
cam.size = OV7670_SIZE_DIV16
cam.colorspace = OV7670_COLOR_YUV
cam.flip_y = False

width = cam.width
mosi_pin = board.GP11
clk_pin = board.GP10
reset_pin = board.GP17
cs_pin = board.GP18
dc_pin = board.GP16

displayio.release_displays()
spi = busio.SPI(clock=clk_pin, MOSI=mosi_pin)
display_bus = displayio.FourWire(
    spi, command=dc_pin, chip_select=cs_pin, reset=reset_pin
)

display = ST7735R(display_bus, width=128, height=160, bgr=True)
display.rotation = 0
group = displayio.Group( scale=2)
display.show(group)
 
color = 0xffffff
text_area = label.Label(font=terminalio.FONT, text='loading...               ', color=color,label_direction="DWR",background_color=0xffd)
text_area.x = 8
text_area.y = 2
group.append(text_area)
 
camera_image = displayio.Bitmap(cam.width, cam.height, 65536)
camera_image_tile = displayio.TileGrid(
    camera_image ,
    pixel_shader=displayio.ColorConverter( 
        input_colorspace=displayio.Colorspace.RGB565_SWAPPED
    ),
    x=20,
    y=25,
)

camera_image_tile.transpose_xy=False
group.append(camera_image_tile)


t0 = time.monotonic_ns()
np.set_printoptions(threshold=300)
ml_image = np.zeros((30, 30), dtype=np.float)

while True:
    
    cam.capture(camera_image)
    time.sleep(0.1)
    for i in range(0,camera_image.width):
        for j in range(0, camera_image.height):
            a = camera_image[i,j]
            camera_image[i,j] = ov7670_y2rgb565(a)
    
    for i in range(0,30):
        for j in range(0,30):
            ml_image[i,j] = 1-rgb565_to_1bit(camera_image[29-i,j])/255

    min_val = np.min(ml_image)
    max_val = np.max(ml_image)

    # Normalize the matrix and post process the image
    ml_image = (ml_image - min_val) / (max_val - min_val)   
    for i in range(0,30):
        for j in range(0,30):
            if ml_image[i,j] < 0.60:
                ml_image[i,j] = 0

    ml_image[:,0] = 0
    ml_image = auto_crop_and_center(ml_image)
    prediction,score,_ = predict(ml_image)    
    text_area.text=f"     p:{prediction}     " 
    print("  prediction:      ",prediction,"score:",score)   
    camera_image.dirty()
    display.refresh(minimum_frames_per_second=0)
    gc.collect()
