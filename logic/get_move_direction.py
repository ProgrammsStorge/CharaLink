import cv2
from PIL import Image

def get_direction(image_path):
    go_to = "forward"
    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    face_img = cv2.imread(image_path).copy()
    face_rect = face_cascade.detectMultiScale(face_img, scaleFactor=1.2, minNeighbors=5)
    if len(face_rect) == 0:
        image = Image.open(image_path)
        globals = []
        old_color = 0
        edit_count = 0
        all_colors = []
        rescale = 5
        all_rects = []
        for i in range(image.size[0] // rescale):
            colors = []
            global_color = 0
            for j in range(image.size[1] // rescale):
                colors.append(min(image.getpixel((i * rescale, j * rescale))) * 2)
                round_num = 100
                global_color = (sum(colors) // len(colors)) // round_num * round_num
                if global_color < 0: global_color = 0
                if global_color > 254: global_color = 254
                all_colors.append(global_color)
            if old_color != global_color:
                edit_count += 1
                try:
                    if edit_count % 2 == 0:
                        all_rects.append([i])
                    else:
                        all_rects[-1].append(i)
                except:
                    pass
            old_color = global_color
            globals.append(global_color)
        try:
            all_rects[-1].append(image.size[0])
        except:
            pass
        good_path = [0, 0]
        for rect in all_rects:
            if rect[1] - rect[0] > good_path[1] - good_path[0]:
                good_path = rect
        good_pos = (good_path[0] + (good_path[1] - good_path[0]) // 2) * rescale

        if good_pos > image.size[0] // 3:
            go_to = "forward"
        if good_pos > image.size[0] // 3 * 2:
            go_to = "right"
    else:
        go_to = None

    return go_to