import cv2
import numpy as np

if __name__ == '__main__':

    chroma = cv2.VideoCapture('videos/confused-travolta.mp4')
    background = cv2.VideoCapture('videos/bar.mp4')

    ret, frame = background.read()

    grava = True

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('output/out.mp4', fourcc, 20.0, (frame.shape[1], int((frame.shape[0] / 2) + frame.shape[0])))

    frameCount = 0
    font = cv2.FONT_HERSHEY_SIMPLEX

    while chroma and background:

        frameCount += 1

        ret1, frame1 = chroma.read()
        ret2, frame2 = background.read()

        if not ret1 and not ret2:
            exit()

        if not ret1:
            notMask = True
            frame1 = np.zeros((frame.shape[1], frame.shape[0], 3), np.uint8)

        frame1 = cv2.resize(frame1, (frame.shape[1], frame.shape[0]))
        frame2 = cv2.resize(frame2, (frame.shape[1], frame.shape[0]))

        notMask = False

        lower = np.array([0, 100, 0], dtype = np.uint8)
        upper = np.array([50, 255, 75], dtype = np.uint8)

        mask = cv2.inRange(frame1, lower, upper)

        backgroundProcess = cv2.bitwise_and(frame2, frame2, mask = mask)

        invertMask = np.invert(mask)

        chromaProcess = cv2.bitwise_and(frame1, frame1, mask = invertMask)

        makingOff = cv2.hconcat([frame1, frame2])
        makingOff = cv2.resize(makingOff, (frame.shape[1], (int(frame.shape[0] / 2))))

        txtFrame1 = cv2.putText(makingOff, 'Chroma-key', (25, 25), font, .5, [255, 255, 255], 1, cv2.LINE_AA)
        txtFrame2 = cv2.putText(makingOff, 'Background', (665, 25), font, .5, [255, 255, 255], 1, cv2.LINE_AA)

        if notMask:
            final = frame2
        else:
            final = cv2.addWeighted(backgroundProcess, 1, chromaProcess, 1, 0)

        count = cv2.putText(final, 'Frame Count: ' + str(frameCount), (25, 25), font, .5, [255, 255, 255], 1, cv2.LINE_AA)
        count = cv2.putText(final, 'Result', (25, 50), font, .5, [255, 255, 255], 1, cv2.LINE_AA)

        final = cv2.vconcat([makingOff, final])

        cv2.imshow('final', final)

        if grava == True:
            out.write(final)

        ch = cv2.waitKey(15)

        if ch == ord('q'):
            out.release()
            cv2.destroyAllWindows()
            break

    out.release()
    chroma.release()
    background.release()
    cv2.destroyAllWindows()
