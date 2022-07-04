from sklearn.neighbors import KNeighborsClassifier
from skimage import exposure
from skimage import feature
from imutils import paths
import argparse
import imutils
import cv2
import time
start_time = time.time()

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--training", required=True, help="Path de entrenamiento")
ap.add_argument("-t", "--test", required=True, help="Path de testeo")
args = vars(ap.parse_args())
data = []
labels = []

#training set
for imagePath in paths.list_images(args["training"]):
    imagePath = imagePath.replace("\\","/")

    make = imagePath.split("/")[-2]

    image = cv2.imread(imagePath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edged = imutils.auto_canny(gray)

    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    c = max(cnts, key=cv2.contourArea)
    (x, y, w, h) = cv2.boundingRect(c)
    face = gray[y:y + h, x:x + w]
    face = cv2.resize(face, (200, 100))
    H = feature.hog(face, orientations=9, pixels_per_cell=(10, 10),
    cells_per_block=(2, 2), transform_sqrt=True, block_norm="L1")
    data.append(H)
    labels.append(make)

    model = KNeighborsClassifier(n_neighbors=1)
    model.fit(data, labels)

#testing set
for (i, imagePath) in enumerate(paths.list_images(args["test"])):

    image = cv2.imread(imagePath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face = cv2.resize(gray, (200, 100))
    (H, hogImage) = feature.hog(face, orientations=9, pixels_per_cell=(10, 10),cells_per_block=(2, 2), transform_sqrt=True, block_norm="L1",visualize=True)
    pred = model.predict(H.reshape(1, -1))[0]
    hogImage = exposure.rescale_intensity(hogImage, out_range=(0, 255))
    hogImage = hogImage.astype("uint8")
    cv2.imshow("HOG Image #{}".format(i + 1), hogImage)
    cv2.putText(image, pred.title(), (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
    (0, 255, 0), 3)
    cv2.imshow("Test Image #{}".format(i + 1), image)
    cv2.waitKey(0)
    
print("--- %s seconds ---" % (time.time() - start_time))
