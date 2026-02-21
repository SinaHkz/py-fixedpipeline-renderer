from OpenGL.GL import glReadPixels, GL_RGB, GL_UNSIGNED_BYTE
from PIL import Image


def save_framebuffer(width, height, filename):
    data = glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE)
    image = Image.frombytes("RGB", (width, height), data)
    image = image.transpose(Image.FLIP_TOP_BOTTOM)
    image.save(filename)
