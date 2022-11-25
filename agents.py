import numpy as np
import random
import models
import cv2

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

        self.randomizePosition()

    def __call__(self, img):
        self.changeImage(img)
        self.randomizePosition()
        self.capture()

    def changeImage(self, image):
        self.refImg = image
    
    # Samples the reference image based on its view window
    def capture(self):        
        self.view = self.refImg[self.pos[0]:self.pos[0]+self.f, #x
                                self.pos[1]:self.pos[1]+self.f] #y

    def move(self):
        pass

    def randomizePosition(self):
        self.pos = (random.randint(0, self.maxBounds[0]), 
                random.randint(0, self.maxBounds[1]))

    # Scans through a photo in a snake-like pattern (from top to bottom).
    # Respects agent's view window boundaries to not access an out of bounds 
    # pixel value in an image
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

    model = None

    agents = None

    def __init__(self, image=np.zeros((28, 28, 1)), numAgents=2, windowLen=2):
        self.newImg = np.zeros(image.shape)

        self.agents = [Agent(image, windowLen) for i in range(numAgents)]

        print(f"Number of agents: {len(self.agents)}")
        self.initModel()

    def capture(self):
        for agent in self.agents:
            agent.capture()
            pos = agent.pos
            self.newImg[pos[0]:pos[0]+agent.f, pos[1]:pos[1]+agent.f, :] = agent.view/255.0

    def sampleImage(self, image):
        #samImg = np.zeros(image.shape)

        self.newImg = np.zeros(image.shape)
        for agent in self.agents:
            agent(image)
            pos = agent.pos
            self.newImg[pos[0]:pos[0]+agent.f, pos[1]:pos[1]+agent.f] = agent.view/255.0

        return self.newImg


    # Function should move all agents across an image appropriately
    def moveAgents(self):
        pass


    def initModel(self):
        self.model = models.create_CNN_model(self.newImg.shape, 10)

    # Loads a saved model. CURRENTLY DOES NOT WORK
    def loadModel(self):
        self.initModel()
        self.model.load_weights(models.checkpoint_path)
    
    

#classifier = ClassifierAgent()

#classifier.model.summary()



#classifier.loadModel()

#classifier.model.summary()
    

    
    


