from agents import Agent, ClassifierAgent
import numpy as np
import cv2 as cv


image = cv.imread("trainingSample/img_569.jpg")

#dim = image.shape
dimResized = (512, 512)

a = Agent(image, 6)
b = Agent(image, 6)
mainDude = ClassifierAgent(image)

a.capture()
b.capture()

cv.imshow("Sample Image", cv.resize(a.refImg, dimResized, interpolation=cv.INTER_AREA))
cv.imshow("Agent a's view", cv.resize(a.view, dimResized, interpolation=cv.INTER_AREA))
cv.imshow("Agent b's view", cv.resize(b.view, dimResized, interpolation=cv.INTER_AREA))
cv.waitKey()

## Test Traverse Function. Should carry the agent through every possible pixel location it can
## occupy based on its window/view size
"""
a.pos = (0, 0)
print("Test Traverse Function\n", a.pos)
for i in range(a.maxBounds[0]*a.maxBounds[1] + 5):
    a.traverse()

    positionalImg = image.copy()
    positionalImg[a.pos[0], a.pos[1]] = (0, 0, 255)
    cv.imshow("Sample Image", cv.resize(positionalImg, dimResized, interpolation=cv.INTER_AREA))
    print(a.pos)
    cv.waitKey(30)
"""

## Test Classifier's reconstruction based on agent(s)
"""
image[27, 27] = (255, 0, 0)
a.pos = (0, 0)
b.pos = (0, 15)
mainDude.update(a, b)
print("Test Classifier's reconstruction function\n", a.pos)
for i in range(int((a.maxBounds[0]+1)*(a.maxBfunds[1]+1)/2)):
    a.traverse()
    b.traverse()
    mainDude.update(a, b)

    positionalImg = image.copy()
    positionalImg[a.pos[0], a.pos[1]] = (0, 0, 255)

    cv.imshow("Positional Image", cv.resize(positionalImg, dimResized, interpolation=cv.INTER_AREA))
    cv.imshow("Reconstructed Image", cv.resize(mainDude.newImg, dimResized, interpolation=cv.INTER_AREA))

    print(f"Agent a position: {a.pos} | Agent b position: {b.pos}")
    cv.waitKey(30)

cv.imshow("Reconstructed Image", cv.resize(mainDude.newImg, dimResized, interpolation=cv.INTER_AREA))
cv.waitKey()

"""

