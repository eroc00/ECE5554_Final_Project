import numpy as np
import random

class Agent:
    f = None
    pos = (0, 0)
    view = None
    refImg = None
    maxBounds = None
    moveHorizontally = False

    # traversal member objects
    __increment = 1

    def __init__(self, referenceImage, fieldOfView):
        self.f = fieldOfView
        self.refImg = referenceImage

        shape = referenceImage.shape
        self.maxBounds = (shape[0] - self.f, shape[1] - self.f)

        self.pos = (random.randint(0, self.maxBounds[0]), 
                random.randint(0, self.maxBounds[1]))

    def __call__(self):
        pass

    def capture(self):        
        self.view = self.refImg[self.pos[0]:self.pos[0]+self.f, #x
                                self.pos[1]:self.pos[1]+self.f] #y

    def move(self):
        pass

    def traverse(self):

        self.prevPos = self.pos
        if self.moveHorizontally:
            self.pos = (self.pos[0], self.pos[1] + 1)
            self.moveHorizontally = False
        else:
            self.pos = (self.pos[0]+self.__increment, self.pos[1])

        if self.pos[1] > self.maxBounds[1]:
            self.pos = (self.pos[0], self.pos[1]-1)
            self.moveHorizontally = True
            return

        movementVector = (self.pos[0] - self.prevPos[0], self.pos[0] - self.prevPos[0])

        topBounds = self.pos[0] == 0
        bottomBounds = self.pos[0] == self.maxBounds[0]
        
        if (topBounds or bottomBounds) and (np.abs(movementVector[0]) == 1):
            self.moveHorizontally = True
            self.__increment = -self.__increment

class ClassifierAgent:
    newImg = None

    def __init__(self, image):
        self.newImg = np.zeros(image.shape, dtype="uint8")

    def update(self, *args):
        for agent in args:
            agent.capture()
            pos = agent.pos
            self.newImg[pos[0]:pos[0]+agent.f, pos[1]:pos[1]+agent.f, :] = agent.view

    


