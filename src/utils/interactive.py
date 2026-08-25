import sys

import cv2
import matplotlib.pyplot as plt


def show_frame(video_path, title="Video Frame", frame_idx=0):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    if ret:
        plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        plt.title(title)
        plt.axis("off")
        plt.show()
        plt.pause(0.1)
    else:
        print("⚠️ Could not read frame.")


def show_image(image, title="Image"):
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis("off")
    plt.show()
    plt.pause(0.1)


def ask_user(prompt="Continue? (y/n): "):
    print("\n" + "=" * 50)
    print("⏳ WAITING FOR YOUR INPUT")
    print(prompt)
    sys.stdout.flush()
    while True:
        ans = input().strip().lower()
        if ans in ("y", "yes", "n", "no"):
            return ans in ("y", "yes")
        print("Please answer 'y' or 'n'.")
