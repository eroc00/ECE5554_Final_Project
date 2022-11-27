from agents import Agent, ClassifierAgent
import numpy as np
import cv2 as cv


image = cv.imread("trainingSample/img_569.jpg")

#dim = image.shape
dimResized = (512, 512)
mainDude = ClassifierAgent(image, numAgents=2, windowLen=2)

# Test agents 
"""

mainDude.capture()

#a.capture()
#b.capture()

cv.imshow("Sample Image", cv.resize(image, dimResized, interpolation=cv.INTER_AREA))
cv.imshow("Agent a's view", cv.resize(mainDude.agents[0].view, dimResized, interpolation=cv.INTER_AREA))
cv.imshow("Agent b's view", cv.resize(mainDude.agents[1].view, dimResized, interpolation=cv.INTER_AREA))
cv.waitKey()
"""

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
mainDude.agents[0].pos = (0, 0)
mainDude.agents[1].pos = (0, 14)

mainDude.capture()
mainDude.traverseAgents()
print("Test Classifier's reconstruction function\n", mainDude.agents[0].pos)
for i in range(int(28*28/len(mainDude.agents))):
    mainDude.capture()
    mainDude.traverseAgents()

    positionalImg = image.copy()

    for agent in mainDude.agents:
        positionalImg[agent.pos[0], agent.pos[1]] = (0, 0, 255)
    

    cv.imshow("Positional Image", cv.resize(positionalImg, dimResized, interpolation=cv.INTER_AREA))
    cv.imshow("Reconstructed Image", cv.resize(mainDude.newImg, dimResized, interpolation=cv.INTER_AREA))
    cv.imshow("Information Space", cv.resize(mainDude.infoSpace, dimResized, interpolation=cv.INTER_AREA))

    print(f"Agent a position: {mainDude.agents[0].pos} | Agent b position: {mainDude.agents[1].pos}")
    cv.waitKey(30)

cv.imshow("Reconstructed Image", cv.resize(mainDude.newImg, dimResized, interpolation=cv.INTER_AREA))
cv.waitKey()
"""

# Test Agent movement functions

classifier = ClassifierAgent(image/255.0, numAgents=4, windowLen=3)

a = classifier.agents[0]
b = classifier.agents[1]
c = classifier.agents[2]
d = classifier.agents[3]

a.pos = (0, 0)
b.pos = (0, 25)
c.pos = (25, 0)
d.pos = (25, 25)

for agent in classifier.agents:
    print(f"Agent {agent}' position: {agent.pos}")

print("Test Horizontal Bounds")
a.moveLeft()
b.moveRight()
c.moveLeft()
d.moveRight()

for agent in classifier.agents:
    print(f"Agent {agent}' in bounds?: {agent.pos}, {agent.pos == agent.prevPos}")

print("Test Vertical Bounds")
a.moveUp()
b.moveUp()
c.moveDown()
d.moveDown()

for agent in classifier.agents:
    print(f"Agent {agent}' in bounds?: {agent.pos}, {agent.pos == agent.prevPos}")

a.moveDown()
a.moveRight()
print(f"Agent a moved diagonally?: {a.pos}, {a.pos == (1, 1)}")

d.moveUp()
d.moveLeft()
print(f"Agent d moved diagonally?: {d.pos}, {d.pos == (24, 24)}")


# Tests the Sampling function by the Classifier Agent
"""
image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

classifier = ClassifierAgent(image/255.0, numAgents=4, windowLen=6)

i = classifier.sampleImage(image)

print(f"Input image dimensions: {image.shape}, data type: {image.dtype} | \
        Output image dimensions: {i.shape}, data type: {i.dtype}")

cv.imshow("Sampled Image", cv.resize(i, dimResized, interpolation=cv.INTER_AREA))

cv.waitKey()
"""