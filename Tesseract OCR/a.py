# import the necessary packages
from pytesseract import Output
import pytesseract
import argparse
import cv2
from PIL import ImageColor



ap = argparse.ArgumentParser()
ap.add_argument(
    "-i", "--image", type=str, required=True, help="Path to input image to be OCR"
)
ap.add_argument(
    "-t",
    "--tesseract",
    type=str,
    default="C:/Program Files/Tesseract-OCR/tesseract.exe",
    help="Path to tesseract.exe",
)
ap.add_argument(
    "-c",
    "--min-conf",
    type=int,
    default=0,
    help="Minium confidence value to filter weak text detection",
)

args = vars(ap.parse_args())

print(args["image"])
print(args["tesseract"])


pytesseract.pytesseract.tesseract_cmd = args["tesseract"]
image_path = args["image"]


# load the input image, convert it from BGR to RGB channel ordering,
# and use Tesseract to localize each area of text in the input image
image = cv2.imread(r"C:\Users\quock\Desktop\test2.jpg")
rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
results = pytesseract.image_to_data(rgb, output_type=Output.DICT)

# loop over each of the individual text localizations
for i in range(0, len(results["text"])):
    # extract the bounding box coordinates of the text region from
    # the current result
    x = results["left"][i]
    y = results["top"][i]
    w = results["width"][i]
    h = results["height"][i]
    # extract the OCR text itself along with the confidence of the
    # text localization
    text = results["text"][i]
    conf = int(results["conf"][i])

    # filter out weak confidence text localizations
    if conf > args["min_conf"]:
        if not (text and text.strip()):
            continue

        # display the confidence and text to our terminal
        print("Confidence: {}".format(conf))
        print("Text: {}".format(text))
        print(f"Local: {(x, y), (x + w, y + h)}")
        print("")
        # strip out non-ASCII text so we can draw the text on the image
        # using OpenCV, then draw a bounding box around the text along
        # with the text itself
        text = "".join([c if ord(c) < 128 else "" for c in text]).strip()
        cv2.rectangle(
            img=image,
            pt1=(x, y),
            pt2=(x + w, y + h),
            color=ImageColor.getcolor("#001e51", "RGB"),
            thickness=2,
        )
        # cv2.putText(
        #     image, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3
        # )
# show the output image
cv2.imshow("Image", image)
cv2.waitKey(0)