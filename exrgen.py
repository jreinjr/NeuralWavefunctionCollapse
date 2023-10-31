import os
import numpy as np
import OpenEXR
import Imath
import matplotlib.pyplot as plt


class EXRUtils:
    @staticmethod
    def write_exr(filename, red, green, blue):
        """Function to save EXR images given R, G, B arrays."""
        header = OpenEXR.Header(red.shape[1], red.shape[0])
        header['channels'] = {
            'R': Imath.Channel(Imath.PixelType(OpenEXR.FLOAT)),
            'G': Imath.Channel(Imath.PixelType(OpenEXR.FLOAT)),
            'B': Imath.Channel(Imath.PixelType(OpenEXR.FLOAT)),
        }
        out = OpenEXR.OutputFile(filename, header)
        out.writePixels({'R': red.tobytes(), 'G': green.tobytes(), 'B': blue.tobytes()})
        out.close()

    @staticmethod
    def load_exr(filename):
        """Function to load EXR images and return R, G, B arrays."""
        file = OpenEXR.InputFile(filename)
        dw = file.header()['dataWindow']
        size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)

        def extract_channel(channel_name):
            channel_str = file.channel(channel_name, Imath.PixelType(OpenEXR.FLOAT))
            channel_data = np.frombuffer(channel_str, dtype=np.float32)
            channel_data.shape = (size[1], size[0])
            return channel_data

        return extract_channel('R'), extract_channel('G'), extract_channel('B')

    @staticmethod
    def create_random_exr_images(num_images, width, height, folder_name):
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)

        for i in range(num_images):
            red = np.random.random((height, width)).astype(np.float32)
            green = np.random.random((height, width)).astype(np.float32)
            blue = np.random.random((height, width)).astype(np.float32)

            filename = os.path.join(folder_name, f'image_{i}.exr')
            EXRUtils.write_exr(filename, red, green, blue)

    @staticmethod
    def display_exr_sequence(folder_name, num_images, loop_count=1, delay=0.1):
        plt.ion()  # Turn on interactive mode
        fig, ax = plt.subplots()

        for _ in range(loop_count):
            for i in range(num_images):
                red, green, blue = EXRUtils.load_exr(os.path.join(folder_name, f'image_{i}.exr'))
                ax.imshow(np.stack([red, green, blue], axis=-1))
                plt.pause(delay)
                ax.cla()  # Clear the axis for the next image

        plt.ioff()  # Turn off interactive mode
        plt.show()


# Example usage:
folder = "exr_images"
EXRUtils.create_random_exr_images(num_images=30, width=256, height=256, folder_name=folder)
EXRUtils.display_exr_sequence(folder_name=folder, num_images=30, loop_count=5, delay=0.1)
