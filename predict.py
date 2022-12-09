import time

import cv2
import numpy as np

from retinaface import Retinaface

if __name__ == "__main__":
    retinaface = Retinaface()
    # ----------------------------------------------------------------------------------------------------------#
    # ----------------------------------------------------------------------------------------------------------#
    mode = "predict"
    # ----------------------------------------------------------------------------------------------------------#
    # video_path is used to specify the path of the video, when video_path = 0 means detect the camera
    # video_path = "xxx.mp4" to detect the video, it means read out the xxx.mp4 file in the root directory.
    # video_save_path means the path to save the video, when video_save_path = "" means no save
    # video_save_path = "yyyy.mp4" to save the video as yyyy.mp4 file in the root directory.
    # video_fps is used for the fps of the saved video
    # video_path, video_save_path and video_fps are only valid when mode='video'
    # save video requires ctrl+c to exit or run to the last frame to complete the full save step.
    # ----------------------------------------------------------------------------------------------------------#
    video_path = 0
    video_save_path = "yyy.mp4"
    video_fps = 25.0
    # -------------------------------------------------------------------------#
    # test_interval is used to specify the number of times the image will be detected when measuring fps
    # Theoretically, the larger the test_interval, the more accurate the fps.
    # -------------------------------------------------------------------------#
    test_interval = 100
    # -------------------------------------------------------------------------#
    # dir_origin_path specifies the path of the folder used for the detected images
    # dir_save_path specifies the path where the detected images are saved
    # dir_origin_path and dir_save_path are only valid when mode='dir_predict'
    # -------------------------------------------------------------------------#
    dir_origin_path = "img/"
    dir_save_path = "img_out/"

    if mode == "predict":
        '''
        predict.py has several points to note
        1, can not be batch prediction, if you want to batch prediction, you can use os.listdir() to traverse the folder, use cv2.imread to open the image file for prediction.
        2、If you want to save it, use cv2.imwrite("img.jpg", r_image) to save it.
        3、If you want to get the coordinates of the box, you can enter the detect_image function and read the four values of (b[0], b[1]), (b[2], b[3]).
        4, if you want to intercept the next target, you can use the obtained (b[0], b[1]), (b[2], b[3]) these four values on the original map using the matrix of the way to intercept.
        5, after replacing the facenet network be sure to re-encode the face, run encoding.py.
        '''
        #while True:
            #img = input('Input image filename:')
            #print(img)
        image = cv2.imread('img/ben9.jpg')
        if image is None:
            print('Open Error! Try again!')
            #continue
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            r_image = retinaface.detect_image(image)[0]
            r_image = cv2.cvtColor(r_image, cv2.COLOR_RGB2BGR)
            cv2.imshow("after", r_image)
            cv2.imwrite('img/bb.jpg', r_image)
            cv2.waitKey(0)

    elif mode == "video":
        capture = cv2.VideoCapture('C:/Users/taoqi/Desktop/test.mp4')
        if video_save_path != "":
            fourcc = cv2.VideoWriter_fourcc(*'FMP4')
            size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            out = cv2.VideoWriter(video_save_path, fourcc, video_fps, size)

        ref, frame = capture.read()
        if not ref:
            raise ValueError("Failure to read the camera (video) correctly, please note whether the camera is installed correctly (whether the video path is filled in correctly).")

        fps = 0.0
        num = 0
        n = []
        #q = ''
        res = False
        left = False
        temp = []
        # [100, 200]
        while (True):
            t1 = time.time()
            # Read a frame
            ref, frame = capture.read()
            if not ref:
                break
            # Format transformation, BGRtoRGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # detect
            frame, q = retinaface.detect_image(frame)
            #print(q)
            frame = np.array(frame)
            if 'CR' in q or 'ZCR' in q:
                res = True
            else:
                res = False
            #temp.append(res)
            #print(res)
            if res and not left:
                left = True
                n.append(num)
            elif not res and left:
                n.append(num - 1)
                left = False
            '''
            elif res and left:
                if num == int(capture.get(7)) - 2:
                    n.append(num)
            '''

            #print(name)
            # RGBtoBGR meets the opencv display format
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            # Calculate FPS
            # Make sure you don't divide by 0
            #print(time.time() - t1)
            a = time.time() - t1
            if a == 0:
                a = 0.0001
            fps = (fps + (1. / a)) / 2
            #print("fps= %.2f" % (fps))
            frame = cv2.putText(frame, "fps= %.2f" % (fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow("video", frame)

            #print(num)
            c = cv2.waitKey(1) & 0xff
            if video_save_path != "":
                out.write(frame)

            if c == 27:
                capture.release()
                break
            num += 1
        print(num)
        if res and left:
            n.append(int(capture.get(7)) - 2)
        lis = []
        print(n)
        #print(int(capture.get(7)) - 1)
        left, right = 0, 1
        while left < len(n) - 1 and right < len(n):
            #lis.append(n[left])
            while right < len(n) - 2 and n[right] + 10 > n[right + 1]:
                right += 2
            lis.append([n[left], n[right]])
            left = right + 1
            right = left + 1
        print(lis)
        #print(len(temp))
        '''
        left = 0
        right = 0
        while left < len(n) and right < len(n):
            while left < len(n) - 1 and n[left] + 10 < n[left + 1]:
                left += 1
            if left == len(n) - 1:
                break
            lis.append(n[left])
            right = left + 1
            while right < len(n) - 1 and n[right] + 10 > n[right + 1]:
                right += 1
            # if right == len(n) - 1:
            # break
            lis.append(n[right])
            left = right + 1
            # right += 1
        print(lis)
        '''
        print("Video Detection Done!")
        capture.release()
        if video_save_path != "":
            print("Save processed video to the path :" + video_save_path)
            out.release()
        cv2.destroyAllWindows()

    elif mode == "fps":
        img = cv2.imread('img/obama.jpg')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        tact_time = retinaface.get_FPS(img, test_interval)
        print(str(tact_time) + ' seconds, ' + str(1 / tact_time) + 'FPS, @batch_size 1')
    elif mode == "dir_predict":
        import os

        from tqdm import tqdm

        img_names = os.listdir(dir_origin_path)
        for img_name in tqdm(img_names):
            if img_name.lower().endswith(
                    ('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
                image_path = os.path.join(dir_origin_path, img_name)
                image = cv2.imread(image_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                r_image = retinaface.detect_image(image)[0]
                r_image = cv2.cvtColor(r_image, cv2.COLOR_RGB2BGR)
                if not os.path.exists(dir_save_path):
                    os.makedirs(dir_save_path)
                cv2.imwrite(os.path.join(dir_save_path, img_name), r_image)
    else:
        raise AssertionError("Please specify the correct mode: 'predict', 'video', 'fps' or 'dir_predict'.")