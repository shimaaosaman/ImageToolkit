# main.py
import cv2
import os
import utils
import matplotlib.pyplot as plt


def show(img, title='Image'):
    """عرض الصورة عبر matplotlib"""
    if len(img.shape) == 3:
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    else:
        plt.imshow(img, cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.show()


def list_images(folder='images'):
    files = [f for f in os.listdir(folder)
             if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    return files


def menu():
    print("""
========================
    IMAGE TOOLKIT
========================
1.  Load Image
2.  Resize
3.  Crop
4.  Rotate
5.  Flip
6.  Color Conversion
7.  Blur
8.  Edge Detection
9.  Thresholding
10. Histogram
11. Draw Shapes
12. Bonus Features
13. Save Current Image
14. Exit
""")


def bonus_menu(img):
    print("""
--- Bonus Features ---
1. Brightness
2. Contrast
3. Gamma Correction
4. Sharpen
5. Cartoon Effect
6. Pencil Sketch
7. Watermark
""")
    choice = input('Choice: ')
    if choice == '1':
        v = int(input('Brightness (-100..100): '))
        return utils.adjust_brightness(img, v)
    elif choice == '2':
        a = float(input('Alpha (1.0-3.0): '))
        return utils.adjust_contrast(img, alpha=a)
    elif choice == '3':
        g = float(input('Gamma (0.1-3.0): '))
        return utils.gamma_correction(img, g)
    elif choice == '4':
        return utils.sharpen_image(img)
    elif choice == '5':
        return utils.cartoon_effect(img)
    elif choice == '6':
        return utils.pencil_sketch(img)
    elif choice == '7':
        t = input('Watermark text: ')
        return utils.add_watermark(img, t)
    return img


def main():
    current = None
    folder = 'images'

    while True:
        menu()
        choice = input('Enter your choice: ').strip()

        if choice == '1':
            files = list_images(folder)
            for i, f in enumerate(files, 1):
                print(f"{i}. {f}")
            idx = int(input('Select number: ')) - 1
            current = utils.load_image(os.path.join(folder, files[idx]))
            w, h, c = utils.get_image_info(current)
            print(f"Loaded: {files[idx]} | {w}x{h}x{c}")
            show(current, 'Original')

        elif choice == '2' and current is not None:
            print('1.256  2.512  3.1024')
            sizes = {'1':256, '2':512, '3':1024}
            s = sizes[input('Size: ')]
            current = utils.resize_image(current, (s, s))
            show(current, f'Resized {s}x{s}')
            utils.save_image(current, prefix='resize')

        elif choice == '3' and current is not None:
            print('1.Center  2.Random')
            t = input('Type: ')
            size = (200, 200)
            if t == '1':
                current = utils.center_crop(current, size)
            else:
                current = utils.random_crop(current, size)
            show(current, 'Cropped')
            utils.save_image(current, prefix='crop')

        elif choice == '4' and current is not None:
            print('1.90  2.180  3.270  4.45')
            angles = {'1':90,'2':180,'3':270,'4':45}
            a = angles[input('Angle: ')]
            current = utils.rotate_image(current, a)
            show(current, f'Rotated {a}')

        elif choice == '5' and current is not None:
            print('1.Horizontal  2.Vertical  3.Both')
            modes = {'1':'horizontal','2':'vertical','3':'both'}
            m = modes[input('Mode: ')]
            current = utils.flip_image(current, m)
            show(current, f'Flip {m}')

        elif choice == '6' and current is not None:
            print('1.RGB  2.Gray  3.HSV  4.LAB')
            opts = {'1':'rgb','2':'gray','3':'hsv','4':'lab'}
            t = opts[input('Target: ')]
            current = utils.convert_color(current, t)
            show(current, f'Color: {t}')

        elif choice == '7' and current is not None:
            print('1.Gaussian 2.Median 3.Average 4.Bilateral')
            kinds = {'1':'gaussian','2':'median','3':'average','4':'bilateral'}
            k = kinds[input('Type: ')]
            ks = int(input('Kernel size (odd): '))
            current = utils.apply_blur(current, k, ks)
            show(current, f'Blur: {k}')

        elif choice == '8' and current is not None:
            print('1.Canny 2.Sobel 3.Laplacian 4.All')
            t = input('Type: ')
            if t == '1':
                show(utils.canny_edge(current), 'Canny')
            elif t == '2':
                show(utils.sobel_edge(current), 'Sobel')
            elif t == '3':
                show(utils.laplacian_edge(current), 'Laplacian')
            elif t == '4':
                fig, ax = plt.subplots(1, 3, figsize=(15, 5))
                ax[0].imshow(utils.canny_edge(current), cmap='gray'); ax[0].set_title('Canny')
                ax[1].imshow(utils.sobel_edge(current), cmap='gray'); ax[1].set_title('Sobel')
                ax[2].imshow(utils.laplacian_edge(current), cmap='gray'); ax[2].set_title('Laplacian')
                for a in ax: a.axis('off')
                plt.show()

        elif choice == '9' and current is not None:
            print('1.Binary 2.Adaptive 3.Otsu')
            methods = {'1':'binary','2':'adaptive','3':'otsu'}
            m = methods[input('Method: ')]
            current = utils.threshold_image(current, m)
            show(current, f'Threshold: {m}')

        elif choice == '10' and current is not None:
            hist = utils.compute_histogram(current)
            if isinstance(hist, dict):
                for c, h in hist.items():
                    plt.plot(h, color=c)
            else:
                plt.plot(hist, color='black')
            plt.title('Histogram')
            plt.show()

        elif choice == '11' and current is not None:
            current = utils.draw_shapes(current)
            show(current, 'Shapes')

        elif choice == '12' and current is not None:
            current = bonus_menu(current)
            show(current, 'Bonus Result')

        elif choice == '13' and current is not None:
            path = utils.save_image(current, prefix='manual_save')
            print(f"Saved to: {path}")

        elif choice == '14':
            print('Goodbye!')
            break

        else:
            print('Invalid choice or no image loaded.')


if __name__ == '__main__':
    main()